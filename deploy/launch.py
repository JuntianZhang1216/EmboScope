"""Launch a paper model and generate a configuration for evaluate.sh."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--list', action='store_true', help='list the 24 paper profiles')
    p.add_argument('--profile', help='profile from deploy/models.json')
    p.add_argument('--model-path', type=Path, help='downloaded checkpoint directory')
    p.add_argument('--python', default=sys.executable, help='interpreter of the model environment')
    p.add_argument('--output', type=Path, help='new directory for model.json, judge.json and deployment records')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--endpoint', help='client-visible chat/completions URL when different from the bind address')
    p.add_argument('--api-key-env', help='environment variable containing the service API key')
    p.add_argument('--tensor-parallel', type=int, help='override the reference GPU count')
    p.add_argument('--max-model-len', type=int, help='override vLLM context capacity; does not truncate inputs')
    p.add_argument('--gpu-memory-utilization', type=float, default=0.85)
    p.add_argument('--max-num-seqs', type=int, default=2)
    p.add_argument('--dry-run', action='store_true', help='print the plan without writing files or loading models')
    return p


def build_plan(args):
    catalog = read(ROOT / 'deploy/models.json')
    if args.profile not in catalog['profiles']:
        raise ValueError('unknown --profile; use --list')
    if args.model_path is None or args.output is None:
        raise ValueError('--model-path and --output are required')
    profile = catalog['profiles'][args.profile]
    model = catalog['models'][profile['model']]
    if not 1 <= args.port <= 65535 or not 0 < args.gpu_memory_utilization <= 1 or args.max_num_seqs < 1:
        raise ValueError('invalid port, GPU memory fraction or sequence count')
    tp = args.tensor_parallel if args.tensor_parallel is not None else model['tensor_parallel']
    context = args.max_model_len if args.max_model_len is not None else model['max_model_len']
    if tp < 1 or context < 1:
        raise ValueError('tensor parallelism and context must be positive')
    if model['backend'] == 'transformers' and (tp != 1 or args.max_model_len is not None):
        raise ValueError('HY MoT uses one GPU and the native checkpoint context')
    weights = args.model_path.expanduser().resolve()
    output = args.output.expanduser().resolve()
    config = read(ROOT / profile['config'])
    if context <= config['generation']['max_tokens']:
        raise ValueError('context must exceed the generation budget')
    host = '127.0.0.1' if args.host == '0.0.0.0' else ('::1' if args.host == '::' else args.host)
    host = f'[{host}]' if ':' in host else host
    endpoint = args.endpoint or f'http://{host}:{args.port}/v1/chat/completions'
    url = urlparse(endpoint)
    if (url.scheme not in {'http', 'https'} or not url.netloc or url.username or url.password
            or url.query or url.fragment or not url.path.endswith('/chat/completions')):
        raise ValueError('--endpoint must be a chat/completions URL without credentials, query or fragment')
    config.update(endpoint=endpoint)
    if args.api_key_env:
        if not args.api_key_env.isidentifier():
            raise ValueError('--api-key-env must name an environment variable')
        config['api_key_env'] = args.api_key_env
    if model['backend'] == 'vllm':
        command = [args.python, '-m', 'vllm.entrypoints.cli.main', 'serve', str(weights),
                   '--served-model-name', profile['model'], '--host', args.host, '--port', str(args.port),
                   '--tensor-parallel-size', str(tp), '--max-model-len', str(context),
                   '--max-num-seqs', str(args.max_num_seqs), '--gpu-memory-utilization', str(args.gpu_memory_utilization),
                   '--limit-mm-per-prompt', json.dumps(model['limit_mm_per_prompt']), '--enforce-eager']
        for arg in model['extra_args'] + profile.get('extra_args', []):
            # JSON-escape only paths embedded in a JSON argument (e.g. Phi's LoRA).
            replacement = str(weights) if not arg.startswith('{"') else json.dumps(str(weights))[1:-1]
            command.append(arg.replace('{model_path}', replacement).replace('{repo_root}', str(ROOT)))
    else:
        command = [args.python, str(ROOT / 'deploy/hy_mot_server.py'), '--model', profile['model'],
                   '--weights', str(weights), '--host', args.host, '--port', str(args.port),
                   '--runtime-dir', str(output / 'server')]
        if args.api_key_env:
            command += ['--api-key-env', args.api_key_env]
    prompts = ROOT / profile.get('prompts', 'eval/embodied_eval/prompts.json')
    return dict(profile=args.profile, model_repo=model['repo'], environment=model['environment'],
                backend=model['backend'],
                command=command, config=config, output=str(output), model_path=str(weights),
                prompts_path=str(prompts), prompts_sha256=hashlib.sha256(prompts.read_bytes()).hexdigest())


def main():
    p = parser()
    args = p.parse_args()
    if args.list:
        catalog = read(ROOT / 'deploy/models.json')
        for key, profile in catalog['profiles'].items():
            model = catalog['models'][profile['model']]
            print(f"{key:35} {model['environment']:8} TP={model['tensor_parallel']}  {model['repo']}")
        return
    try:
        plan = build_plan(args)
        if args.dry_run:
            print(json.dumps(plan, indent=2))
            return
        executable = shutil.which(args.python)
        if not executable:
            raise ValueError('model Python interpreter not found')
        # Keep the venv executable path: resolving its symlink would bypass the venv.
        executable = os.path.abspath(executable)
        plan['command'][0] = executable
        weights = Path(plan['model_path'])
        if not (weights / 'config.json').is_file():
            raise ValueError('--model-path must contain config.json and the complete checkpoint')
        if args.profile == 'phi4-multimodal' and not (weights / 'vision-lora').is_dir():
            raise ValueError('Phi-4 requires the checkpoint vision-lora directory')
        if args.api_key_env and not os.environ.get(args.api_key_env):
            raise ValueError('missing API key environment variable: ' + args.api_key_env)
        probe = subprocess.run([executable, str(ROOT / 'deploy/runtime_probe.py'), '--environment', plan['environment']],
                               check=True, text=True, capture_output=True)
        plan['runtime'] = json.loads(probe.stdout)
        plan['checkpoint_config_sha256'] = hashlib.sha256((weights / 'config.json').read_bytes()).hexdigest()
        plan['deployment_code_sha256'] = {
            path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((ROOT / 'deploy').rglob('*'))
            if path.is_file() and '__pycache__' not in path.parts
            and path.suffix in {'.py', '.sh', '.json', '.patch', '.jinja', '.txt'}}
        config = plan['config']
        plan['deployment_sha256'] = fingerprint(plan)
        config['model_revision'] = 'local-deployment:' + plan['deployment_sha256']
        config['notes'] = 'deployment_sha256=' + plan['deployment_sha256']
        output = Path(plan['output'])
        output.mkdir(parents=True, exist_ok=False)
        for name, value in [('model.json', config), ('deployment.json', plan)]:
            (output / name).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
        if args.profile == 'qwen35-a17b-fp8':
            judge = dict(config, generation=read(ROOT / 'configs/judge.local.example.json')['generation'])
            (output / 'judge.json').write_text(json.dumps(judge, indent=2) + '\n')
        command = ['bash', str(ROOT / 'evaluate.sh'), '--model-config', str(output / 'model.json'),
                   '--prompts-path', plan['prompts_path'], '--output', 'outputs/' + args.profile,
                   '--judge-config', 'PATH_TO_JUDGE_CONFIG']
        (output / 'evaluation-command.txt').write_text(shlex.join(command) + '\n')
        print('Evaluation config: ' + str(output / 'model.json'), flush=True)
        print('Launch: ' + shlex.join(plan['command']), flush=True)
        env = dict(os.environ)
        if plan['backend'] == 'vllm':
            # vLLM reads its key from the environment; never put a secret in argv or records.
            env.pop('VLLM_API_KEY', None)
            if args.api_key_env:
                env['VLLM_API_KEY'] = os.environ[args.api_key_env]
        os.execve(executable, plan['command'], env)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
            print(exc.stderr, file=sys.stderr)
        p.exit(2, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
