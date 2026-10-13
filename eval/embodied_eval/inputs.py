"""Compile only allowlisted public fields; keep gold in a separate scoring file."""
import hashlib
import io
import os
import re
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from . import JUDGE_PROTOCOL_VERSION
from .common import digest, file_hash, lock, read_json, read_jsonl, write_json, write_jsonl, code_hashes


def normalize(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text).strip()).casefold()


def options(texts):
    result = {}
    for text in texts:
        for match in re.finditer(r'^\s*([A-Z])(?:[.、:)]\s*(.*)|\s*)$', text, re.M):
            if match[1] in result:
                raise ValueError('duplicate option letter')
            result[match[1]] = (match[2] or match[1]).strip()
    if len(result) < 2:
        raise ValueError('fewer than two options')
    return result


def gold(record, kind):
    answer = record['text_list'][-1]
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError('empty reference')
    canonical = None
    if kind == 'single_choice':
        opts = options(record['text_list'][:-1])
        if answer.strip() in opts:
            canonical = answer.strip()
        else:
            matches = [k for k, v in opts.items() if normalize(v) == normalize(answer)]
            if len(matches) != 1:
                raise ValueError('reference does not map to exactly one option')
            canonical = matches[0]
    elif kind == 'yes_no':
        if normalize(answer) not in ('yes', 'no'):
            raise ValueError('invalid Yes/No reference')
        canonical = normalize(answer).title()
    return dict(reference_answer=answer, canonical_gt=canonical)


def inspect_image(path):
    path = str(Path(path).resolve())
    data = Path(path).read_bytes()
    with Image.open(io.BytesIO(data)) as im:
        im.load()
        if getattr(im, 'n_frames', 1) != 1:
            raise ValueError('animated images are not supported')
        mime = Image.MIME.get(im.format)
        if mime not in ('image/jpeg', 'image/png', 'image/webp'):
            raise ValueError(f'unsupported image encoding: {mime}')
        width, height = im.size
    return dict(path=path, sha256=hashlib.sha256(data).hexdigest(), mime=mime,
                width=width, height=height, bytes=len(data))


def label(record, index, media):
    task = record['task']
    anchor = media['matched_text_index']
    prefix = f'image_index={index}'
    if task in ('frame_completion', 'action_frame_matching'):
        if task == 'frame_completion' and anchor == 0:
            return f'{prefix}; context sequence position={index + 1} (1-based)'
        if anchor < 1:
            raise ValueError('candidate lacks preceding option label')
        preceding = record['text_list'][anchor - 1]
        match = re.fullmatch(r'\s*([A-Z])(?:[.、]\s*(\S.*?)|)\s*', preceding)
        if not match:
            raise ValueError('unsupported candidate image/text binding')
        return f'{prefix}; candidate option={match[1]}; original label={match[2] or match[1]}'
    if task == 'step_frame_alignment':
        return f'{prefix}; strip label={chr(65 + index)}'
    if task == 'attribute_matching':
        return f'{prefix}; frame {index}'
    if task == 'cross_view_matching':
        return f'{prefix}; frame label={chr(97 + index)}'
    if task in ('progress_comparison', 'progress_ordering', 'progress_anomaly_detection'):
        return f'{prefix}; Image {index + 1}'
    return prefix


def compile_record(record, line, kind, prompts, images, inspected):
    texts = record['text_list']
    if len(texts) < 2 or not all(isinstance(t, str) for t in texts):
        raise ValueError('invalid text_list')
    if len(record['role_list']) != len(texts) + len(record['image_list']):
        raise ValueError('role/media count mismatch')
    positions = defaultdict(list)
    for i, media in enumerate(images):
        anchor = media['matched_text_index']
        if type(anchor) is not int or not 0 <= anchor < len(texts):
            raise ValueError('invalid media anchor')
        info = inspected[str(Path(media['path']).resolve())]
        if media.get('sha256') and media['sha256'] != info['sha256']:
            raise ValueError('image hash mismatch')
        positions[anchor].append((i, media, info))
    content = []
    if record['task'] == 'frame_completion':
        content.append(dict(type='text', text='Context sequence positions are 1-based. Candidate frame names are labels, not sequence positions or image_index. Keep the supplied candidate bindings.'))
    for anchor in range(len(texts)):
        for i, media, info in positions[anchor]:
            if media.get('text_before'):
                content.append(dict(type='text', text=media['text_before']))
            content.append(dict(type='text', text=label(record, i, media)))
            content.append(dict(type='image', image_index=i, **info))
        if anchor < len(texts) - 1:
            content.append(dict(type='text', text=texts[anchor]))
    instruction = prompts['output_instructions'][record['task'] if kind == 'open' else kind]
    item = dict(id=record['id'], line=line, task=record['task'], answer_format=kind,
                option_letters=list(options(texts[:-1])) if kind == 'single_choice' else [],
                system=prompts['system'], content=content, output_instruction=instruction)
    item['input_sha256'] = digest(item)
    return item


def prepare(dataset, catalog, prompts_path, output, tasks=None, per_task=None, workers=8):
    output = Path(output)
    dataset = Path(dataset).resolve()
    media_root = dataset.parent
    with lock(output):
        if (output / 'manifest.json').exists():
            raise ValueError('prepared corpus already exists; use a new output directory')
        compiler_version = code_hashes()
        records = read_jsonl(dataset)
        ids = [r.get('id') for r in records]
        if any(not isinstance(sid, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', sid) for sid in ids) or len(ids) != len(set(ids)):
            raise ValueError('question IDs must be unique, nonempty filename-safe strings')
        task_defs = read_json(catalog)['tasks']
        prompts = read_json(prompts_path)
        if tasks and set(tasks) - set(task_defs):
            raise ValueError('unknown task filter')
        counts = Counter()
        selected = []
        for line, record in enumerate(records, 1):
            task = record['task']
            if task not in task_defs:
                raise ValueError(f'unknown task at line {line}')
            if tasks and task not in tasks or per_task and counts[task] >= per_task:
                continue
            kind = task_defs[task]['answer_format']
            private = gold(record, kind)
            if set(record) - {'id', 'task', 'text_list', 'image_list', 'role_list'}:
                raise ValueError(f'unsupported question fields at line {line}')
            media = [dict(m, path=m['image']) for m in record['image_list']]
            for m in media:
                path = Path(m['path'])
                if path.is_absolute() or len(path.parts) != 2 or path.parts[0] != 'images' or '..' in path.parts or '\\' in m['path']:
                    raise ValueError('media must use relative images/ paths inside the dataset directory')
                m['path'] = str((media_root / m['path']).resolve())
                if not Path(m['path']).is_relative_to(media_root):
                    raise ValueError('media must be inside the dataset directory')
            selected.append((line, record, kind, private, media))
            counts[task] += 1
        if not selected:
            raise ValueError('empty selection')
        paths = sorted({str(Path(m['path']).resolve()) for _, _, _, _, media in selected for m in media})
        print(f'Verifying {len(paths)} unique images for {len(selected)} questions...', flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            inspected = dict(zip(paths, pool.map(inspect_image, paths)))
        for info in inspected.values():
            info['path'] = Path(info['path']).relative_to(media_root).as_posix()
        public, answers = [], []
        for line, record, kind, private, media in selected:
            item = compile_record(record, line, kind, prompts, media, inspected)
            public.append(item)
            answers.append(dict(id=item['id'], task=record['task'], input_sha256=item['input_sha256'], **private))
        write_jsonl(output / 'items.jsonl', public)
        write_jsonl(output / 'private/answers.jsonl', answers)
        write_json(output / 'prompts.json', prompts)
        names = ['items.jsonl', 'private/answers.jsonl', 'prompts.json']
        manifest = dict(schema_version=2, dataset_sha256=file_hash(dataset), catalog_sha256=file_hash(catalog),
                        total=len(public), full_dataset=len(public) == len(records), tasks=dict(counts),
                        selection=dict(tasks=tasks, per_task=per_task), unique_images=len(paths),
                        media_root=Path(os.path.relpath(media_root, output.resolve())).as_posix(),
                        media_storage='relative_paths_original_bytes_verified_on_each_request',
                        files={name: file_hash(output / name) for name in names}, compiler=compiler_version,
                        open_scores='binary_0_or_1', judge_protocol_version=JUDGE_PROTOCOL_VERSION)
        write_json(output / 'manifest.json', manifest)
        return manifest
