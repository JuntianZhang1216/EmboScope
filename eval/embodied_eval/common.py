import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temp.open('wb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def write_json(path, value):
    atomic_bytes(path, json.dumps(value, ensure_ascii=False, indent=2).encode() + b'\n')


def write_jsonl(path, rows):
    atomic_bytes(path, b''.join(encoded(row) + b'\n' for row in rows))


@contextlib.contextmanager
def lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.lock').open('a') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(f'directory already in use: {directory}') from None
        yield


def code_hashes():
    return {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


class CorpusItem(dict):
    """A frozen item with a runtime-only media root, excluded from JSON and hashes."""

    def __init__(self, value, media_root):
        super().__init__(value)
        self.media_root = Path(media_root)


def load_corpus(directory, private=False):
    directory = Path(directory).resolve()
    manifest = read_json(directory / 'manifest.json')
    if manifest['schema_version'] != 2 or Path(manifest['media_root']).is_absolute():
        raise ValueError('corpus requires schema 2 and a relative media_root')
    names = ['items.jsonl', 'prompts.json'] + (['private/answers.jsonl'] if private else [])
    for name in names:
        if file_hash(directory / name) != manifest['files'][name]:
            raise ValueError(f'corpus changed: {name}')
    items = read_jsonl(directory / 'items.jsonl')
    for item in items:
        if digest({k: v for k, v in item.items() if k != 'input_sha256'}) != item['input_sha256']:
            raise ValueError('input hash mismatch')
    media_root = (directory / manifest['media_root']).resolve()
    return manifest, [CorpusItem(item, media_root) for item in items]
