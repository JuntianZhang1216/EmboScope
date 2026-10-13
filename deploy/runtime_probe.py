"""Check inference-critical package versions and compatibility patch hashes without loading weights."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import sys


def inspect(environment):
    if sys.version_info[:2] != (3, 12):
        raise ValueError('the reference deployment requires Python 3.12')
    expected = {'transformers': '4.57.6', 'torch': '2.10.0+cu129',
                'vllm': '0.16.1rc1.dev167+gc21d0039e', 'triton': '3.6.0', 'tokenizers': '0.22.2'}
    if environment == 'glm':
        expected.update(transformers='5.2.0', huggingface_hub='1.4.1')
    if environment == 'phi':
        expected['scipy'] = '1.17.1'
    if environment == 'hy-vlm':
        expected['hy-embodied-vllm-plugin'] = '0.1.0'
    if environment == 'hy-mot':
        expected = dict(torch='2.6.0', transformers='4.57.0', flash_attn='2.8.3',
                        timm='1.0.26', accelerate='1.13.0', tokenizers='0.22.2',
                        fastapi='0.141.1', uvicorn='0.53.0')
    versions = {name: metadata.version(name) for name in expected}
    for name, version in expected.items():
        actual = versions[name].split('+')[0] if environment == 'hy-mot' and name == 'torch' else versions[name]
        if actual != version:
            raise ValueError(f'{name}: expected {version}, found {versions[name]}')
    patches = {}
    if environment in {'phi', 'hy-vlm'}:
        specs = json.loads((Path(__file__).parent / 'patches/manifest.json').read_text())
        name, package = ('phi4mm-crops', 'vllm') if environment == 'phi' else ('hy-vlm10-vllm', 'hy-embodied-vllm-plugin')
        dist = metadata.distribution(package)
        for path, record in specs[name]['files'].items():
            actual = hashlib.sha256(Path(dist.locate_file(path)).read_bytes()).hexdigest()
            if actual != record['after']:
                raise ValueError(f'{path}: compatibility patch missing or different')
            patches[path] = actual
    if environment == 'hy-mot':
        direct = json.loads(metadata.distribution('transformers').read_text('direct_url.json') or '{}')
        commit = direct.get('vcs_info', {}).get('commit_id')
        if commit != '9293856c419762ebf98fbe2bd9440f9ce7069f1a':
            raise ValueError('install the pinned HY Transformers git revision with deploy/install.sh')
    # Record the complete resolved environment, including unpinned transitive dependencies.
    installed = sorted((d.metadata['Name'], d.version) for d in metadata.distributions() if d.metadata['Name'])
    return dict(environment=environment, python=sys.version, critical_packages=versions,
                compatibility_hashes=patches, packages=dict(installed))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--environment', choices=['vllm', 'glm', 'phi', 'hy-vlm', 'hy-mot'], required=True)
    args = p.parse_args()
    try:
        print(json.dumps(inspect(args.environment), indent=2))
    except (ValueError, OSError, metadata.PackageNotFoundError) as exc:
        p.exit(2, f'Error: {exc}\n')
