"""Apply a compatibility patch only to files with the expected upstream hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

PATCHES = Path(__file__).resolve().parent / 'patches'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply_patch(name, root):
    spec = json.loads((PATCHES / 'manifest.json').read_text())[name]
    root = Path(root).resolve()
    states = [sha(root / path) for path in spec['files']]
    if states == [entry['after'] for entry in spec['files'].values()]:
        return
    if states != [entry['before'] for entry in spec['files'].values()]:
        raise ValueError(f'{name}: upstream file hashes differ; refusing to patch a different version')
    command = ['patch', '--batch', '--fuzz=0', '-p1', '-i', str(PATCHES / spec['patch'])]
    subprocess.run(command + ['--dry-run'], cwd=root, check=True)
    subprocess.run(command, cwd=root, check=True)
    for path, entry in spec['files'].items():
        if sha(root / path) != entry['after']:
            raise ValueError(f'{name}: patched hash mismatch for {path}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('name', choices=['phi4mm-crops', 'hy-vlm10-vllm'])
    p.add_argument('root', type=Path)
    args = p.parse_args()
    apply_patch(args.name, args.root)
