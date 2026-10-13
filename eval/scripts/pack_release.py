"""Create separate code and complete dataset archives with portable entry names."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tarfile
import zipfile


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def code_files(root):
    root = Path(root)
    files = [root/name for name in ('LICENSE', 'README.md', 'README_zh.md', 'pyproject.toml', 'requirements.txt', '.gitignore', 'evaluate.sh')]
    for folder in ('eval', 'docs', '.github'):
        files += [p for p in (root/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts
                  and p.suffix in ('.py', '.json', '.md', '.yml')]
    files += [p for p in (root/'docs/figures').glob('*') if p.is_file() and p.suffix in ('.png', '.pdf')]
    files += list((root/'configs').glob('*.example.json'))
    for folder in ('models', 'prompts'):
        files += list((root/'configs'/folder).glob('*.json'))
    files += [p for p in (root/'deploy').glob('*') if p.suffix in ('.py', '.sh', '.json', '.md')]
    for folder, extensions in [('requirements', ('.txt',)), ('patches', ('.patch', '.json')),
                               ('templates', ('.jinja',)), ('licenses', ('.txt',))]:
        files += [p for p in (root/'deploy'/folder).glob('*') if p.suffix in extensions]
    # The code checkout contains only a placeholder for the separate data download.
    files += [root/'data/.gitkeep']
    files += list((root/'metadata').glob('*.json')) + list((root/'metadata').glob('*.jsonl'))
    return sorted(set(files))


def pack_code(root, code):
    root, code = Path(root).resolve(), Path(code)
    with zipfile.ZipFile(code, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in code_files(root):
            if path.is_symlink() or not path.is_file():
                raise ValueError('archives require regular files')
            archive.write(path, path.relative_to(root).as_posix())


def pack_data(root, data):
    root, data = Path(root).resolve(), Path(data)
    data_root = root / 'data'
    dataset_files = [data_root / 'benchmark_data.jsonl']
    media = sorted((data_root / 'images').iterdir())
    with data.open('xb') as stream, gzip.GzipFile(filename='', mode='wb', fileobj=stream, mtime=0, compresslevel=1) as compressed:
        with tarfile.open(fileobj=compressed, mode='w|', format=tarfile.PAX_FORMAT) as archive:
            for count, path in enumerate(dataset_files + media, 1):
                if path.is_symlink() or not path.is_file():
                    raise ValueError('archives require regular files')
                info = archive.gettarinfo(str(path), arcname=path.relative_to(data_root).as_posix())
                info.uid = info.gid = 0
                info.uname = info.gname = ''
                info.mtime = 0
                info.mode = 0o644
                info.pax_headers = {}
                with path.open('rb') as source:
                    archive.addfile(info, source)
                if count % 10000 == 0:
                    print(f'Packed {count} files', flush=True)
    return len(media)


def pack(root, output, code_only=False):
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    version = json.loads((root/'metadata/release_manifest.json').read_text())['version']
    code = output/f'embodied-benchmark-code-v{version}.zip'
    data = output/f'embodied-benchmark-data-v{version}.tar.gz'
    if code.exists() or (not code_only and data.exists()):
        raise FileExistsError('release archives already exist')
    pack_code(root, code)
    if code_only:
        (output/(code.name + '.sha256')).write_text(f'{sha256(code)}  {code.name}\n')
        return {'code_archive': code.name}
    media_files = pack_data(root, data)
    (output/f'SHA256SUMS-v{version}').write_text(''.join(f'{sha256(p)}  {p.name}\n' for p in (code, data)))
    return {'code_archive': code.name, 'data_archive': data.name, 'media_files': media_files}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default=str(Path(__file__).resolve().parents[2]))
    p.add_argument('--output', required=True)
    p.add_argument('--code-only', action='store_true', help='package code without rebuilding the media archive')
    print(pack(**vars(p.parse_args())))
