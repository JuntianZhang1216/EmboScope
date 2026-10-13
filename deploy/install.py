"""Build an isolated Python 3.12 model environment from pinned source revisions."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess

from apply_patches import apply_patch

HERE = Path(__file__).resolve().parent
VLLM_COMMIT = 'c21d0039ecc85b06617034ff2166a1fb79309d53'
VLLM_VERSION = '0.16.1rc1.dev167+gc21d0039e'
HY_COMMIT = '16289e99111ee362a23f93d5b3a463ddc844decc'


def install(environment, target, python, dry_run=False):
    target = Path(target).expanduser().resolve()
    if target.exists():
        raise ValueError('--venv must be a new directory; existing environments are never modified')
    interpreter = str(target / 'bin/python')
    pip = [interpreter, '-m', 'pip']
    requirements = HERE / 'requirements' / (environment + '.txt')
    build_env = dict(os.environ, VLLM_TARGET_DEVICE='cuda')
    build_env.setdefault('MAX_JOBS', '8')
    build_env.setdefault('NVCC_THREADS', '2')

    def run(command, capture=False, env_overrides=None):
        if env_overrides:
            print('Build environment: ' + json.dumps(env_overrides), flush=True)
        print(shlex.join(map(str, command)), flush=True)
        if dry_run:
            return ''
        return subprocess.run(list(map(str, command)), check=True, text=True,
                              capture_output=capture, env=dict(build_env, **(env_overrides or {}))).stdout

    def checkout(repo, commit, destination):
        run(['git', 'clone', '--filter=blob:none', '--no-checkout', repo, destination])
        run(['git', '-C', destination, 'checkout', '--detach', commit])
        actual = run(['git', '-C', destination, 'rev-parse', 'HEAD'], capture=True)
        if not dry_run and actual.strip() != commit:
            raise ValueError('upstream checkout revision mismatch')

    run([python, '-c', 'import sys; assert sys.version_info[:2] == (3, 12), "Use Python 3.12"'])
    run([python, '-m', 'venv', target])
    run(pip + ['install', '--upgrade', 'pip', 'setuptools>=77.0.3,<81', 'wheel', 'packaging', 'ninja'])
    if environment == 'hy-mot':
        run(pip + ['install', 'torch==2.6.0', 'torchvision==0.21.0', 'torchaudio==2.6.0',
                   '--index-url', 'https://download.pytorch.org/whl/cu124'])
        run(pip + ['install', '-r', requirements])
        run(pip + ['install', '--no-build-isolation', 'flash-attn==2.8.3'])
    else:
        run(pip + ['install', 'torch==2.10.0', 'torchvision==0.25.0', 'torchaudio==2.10.0',
                   '--index-url', 'https://download.pytorch.org/whl/cu129'])
        source = target / 'src/vllm'
        if not dry_run:
            source.parent.mkdir(parents=True)
        checkout('https://github.com/vllm-project/vllm.git', VLLM_COMMIT, source)
        run(pip + ['install', '-r', source / 'requirements/build.txt'])
        if environment == 'glm':
            # GLM-4.6V needs the 5.2 processor with this vLLM implementation.
            # Declare that tested combination in this dedicated build's metadata.
            print('Set the dedicated vLLM build dependency to transformers==5.2.0', flush=True)
            if not dry_run:
                common = source / 'requirements/common.txt'
                content = common.read_text()
                old = 'transformers >= 4.56.0, < 5'
                if content.count(old) != 1:
                    raise ValueError('unexpected vLLM Transformers dependency declaration')
                common.write_text(content.replace(old, 'transformers == 5.2.0'))
        if environment == 'phi':
            print('Apply and verify phi4mm-crops in the isolated source tree', flush=True)
            if not dry_run:
                apply_patch('phi4mm-crops', source)
        # Only this source build receives the vLLM version override. Building
        # dependencies under that override could assign them the wrong version.
        run(pip + ['install', '--no-build-isolation', '--no-deps', source],
            env_overrides={'SETUPTOOLS_SCM_PRETEND_VERSION': VLLM_VERSION})
        run(pip + ['install', '-c', requirements, '-r', requirements, 'vllm==' + VLLM_VERSION])
        if environment == 'hy-vlm':
            hy = target / 'src/HY-Embodied'
            checkout('https://github.com/Tencent-Hunyuan/HY-Embodied.git', HY_COMMIT, hy)
            plugin = hy / 'Hy-Embodied-VLM-1.0/inference/vllm'
            print('Apply and verify hy-vlm10-vllm in the isolated plugin source', flush=True)
            if not dry_run:
                apply_patch('hy-vlm10-vllm', plugin)
            run(pip + ['install', '--no-deps', plugin])
    run(pip + ['check'])
    probe = run([interpreter, HERE / 'runtime_probe.py', '--environment', environment], capture=True)
    freeze = run(pip + ['freeze'], capture=True)
    if not dry_run:
        (target / 'emboscope-environment.json').write_text(json.dumps(json.loads(probe), indent=2) + '\n')
        (target / 'emboscope-freeze.txt').write_text(freeze)
        print('Ready: ' + interpreter)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--environment', choices=['vllm', 'glm', 'phi', 'hy-vlm', 'hy-mot'], required=True)
    p.add_argument('--venv', type=Path, required=True)
    p.add_argument('--python', default='python3.12', help='base Python 3.12 interpreter')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    try:
        install(args.environment, args.venv, args.python, args.dry_run)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        p.exit(2, f'Error: {exc}\n')
