"""Validate the standalone dataset without inference or external storage."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

from .common import file_hash, read_json, read_jsonl, write_json
from .inputs import gold, inspect_image


def validate_release(dataset, metadata_dir, workers=8, output=None):
    dataset = Path(dataset).resolve()
    root = dataset.parent
    metadata_dir = Path(metadata_dir)
    manifest = read_json(metadata_dir / 'release_manifest.json')
    rows = read_jsonl(dataset)
    catalog = read_json(Path(__file__).parent / 'task_catalog.json')['tasks']
    if file_hash(dataset) != manifest['dataset_sha256']:
        raise ValueError('dataset checksum mismatch')
    if file_hash(metadata_dir / 'images.jsonl') != manifest['images_manifest_sha256']:
        raise ValueError('image manifest checksum mismatch')
    if file_hash(metadata_dir / 'video_frames.jsonl') != manifest['video_frames_sha256']:
        raise ValueError('video frame manifest checksum mismatch')
    if len(rows) != manifest['questions']:
        raise ValueError('question count mismatch')
    images = read_jsonl(metadata_dir / 'images.jsonl')
    by_path = {m['path']: m for m in images}
    if len(by_path) != len(images):
        raise ValueError('duplicate media manifest entries')
    referenced, counts, formats = set(), Counter(), Counter()
    for line, row in enumerate(rows, 1):
        if row.get('id') != f'q{line:06d}' or 'source' in row or row.get('video_list'):
            raise ValueError(f'invalid released question metadata at line {line}')
        task = row['task']
        if task not in catalog:
            raise ValueError(f'unknown task at line {line}')
        kind = catalog[task]['answer_format']
        gold(row, kind)
        texts, media = row['text_list'], row['image_list']
        if not media or len(row['role_list']) != len(texts) + len(media):
            raise ValueError(f'invalid text/media layout at line {line}')
        if any(not isinstance(t, str) for t in texts):
            raise ValueError('text_list must contain strings')
        counts[task] += 1
        formats[kind] += 1
        for m in media:
            path = PurePosixPath(m['image'])
            if (path.is_absolute() or len(path.parts) != 2 or path.parts[0] != 'images'
                    or '..' in path.parts or '\\' in str(path) or m['image'] not in by_path):
                raise ValueError(f'invalid media reference in {row["id"]}')
            if type(m['matched_text_index']) is not int or not 0 <= m['matched_text_index'] < len(texts):
                raise ValueError('invalid image/text anchor')
            expected = by_path[m['image']]
            for key in ('sha256', 'width', 'height'):
                if m[key] != expected[key]:
                    raise ValueError('question media metadata mismatch')
            referenced.add(m['image'])
    if referenced != set(by_path) or dict(counts) != manifest['tasks']:
        raise ValueError('media coverage or task counts differ from release manifest')
    if dict(formats) != manifest['answer_formats'] or len(images) != manifest['unique_images']:
        raise ValueError('release statistics mismatch')
    if sum(len(r['image_list']) for r in rows) != manifest['media_references']:
        raise ValueError('media reference count mismatch')
    frame_rows = read_jsonl(metadata_dir / 'video_frames.jsonl')
    frame_ids = {r['id'] for r in frame_rows}
    if len(frame_rows) != len(frame_ids) or len(frame_rows) != manifest['video_questions_as_fixed_frames']:
        raise ValueError('video question count mismatch')
    by_id = {r['id']: r for r in rows}
    for record in frame_rows:
        media = by_id[record['id']]['image_list']
        expected = [dict(image_index=i, image=m['image'], source_frame_index=m['source_frame_index'],
                         pts_sec=m.get('pts_sec')) for i, m in enumerate(media)]
        indices = [f['source_frame_index'] for f in record['frames']]
        if record['frames'] != expected or indices != sorted(set(indices)):
            raise ValueError('video frame order or positions changed')
    actual_paths = {p.relative_to(root).as_posix() for p in (root / 'images').iterdir()}
    if actual_paths != referenced:
        raise ValueError('images directory contains missing or unlisted entries')

    def check(m):
        path = root / m['path']
        if path.is_symlink() or not path.is_file():
            raise ValueError('release media must be standalone regular files')
        actual = inspect_image(path)
        for key in ('sha256', 'mime', 'width', 'height', 'bytes'):
            if actual[key] != m[key]:
                raise ValueError(f'media {key} mismatch: {m["path"]}')
        return actual['bytes']

    with ThreadPoolExecutor(max_workers=workers) as pool:
        total_bytes = sum(pool.map(check, images))
    if total_bytes != manifest['media_bytes']:
        raise ValueError('media byte count mismatch')
    result = dict(status='passed', questions=len(rows), tasks=dict(counts),
                  answer_formats=dict(formats), unique_images=len(images), media_bytes=total_bytes,
                  dataset_sha256=file_hash(dataset), all_media_hashes_verified=True,
                  all_images_decoded=True, relative_paths_only=True, source_field_absent=True,
                  standalone_regular_files=True)
    if output:
        write_json(output, result)
    return result
