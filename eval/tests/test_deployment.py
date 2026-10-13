"""CPU checks for the paper deployment profiles and native-server HTTP contract."""
import base64
from contextlib import nullcontext, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from jinja2 import Environment
from PIL import Image

from embodied_eval.answer_adapters import adapt_answer
from embodied_eval.common import read_json
from embodied_eval.scoring import closed_score
from embodied_eval.serving import Client, validate_config
import test_framework as framework

ROOT = Path(__file__).resolve().parents[2]


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


launch = module('deployment_launch', 'deploy/launch.py')
hy = module('deployment_hy', 'deploy/hy_mot_server.py')
patcher = module('deployment_patches', 'deploy/apply_patches.py')
packer = module('deployment_pack', 'eval/scripts/pack_release.py')


class DeploymentTests(unittest.TestCase):
    def args(self, profile, *extra):
        return launch.parser().parse_args([
            '--profile', profile, '--model-path', 'weights with spaces/"checkpoint"',
            '--output', 'outputs/service', '--dry-run', *extra])

    def test_all_paper_profiles_generate_current_client_configs(self):
        catalog = read_json(ROOT / 'deploy/models.json')
        self.assertEqual(len(catalog['models']), 20)
        self.assertEqual(len(catalog['profiles']), 24)
        thinking = {'minicpm-v45-thinking': 8192, 'qwen35-27b-thinking': 8192,
                    'glm46v-thinking': 8192, 'internvl35-14b-thinking': 7168}
        for profile in catalog['profiles']:
            with self.subTest(profile=profile):
                plan = launch.build_plan(self.args(profile))
                cfg = validate_config(plan['config'])
                self.assertEqual(cfg['generation'], dict(temperature=0, top_p=1, seed=42,
                    max_tokens=thinking.get(profile, 8192 if profile == 'mimo-embodied-7b' else 1024)))
                self.assertEqual(cfg['reasoning_mode'] == 'enabled', profile in thinking)
                command = plan['command']
                if plan['backend'] == 'vllm':
                    self.assertEqual(command.count('--limit-mm-per-prompt'), 1)
                    limits = json.loads(command[command.index('--limit-mm-per-prompt') + 1])
                    self.assertEqual(limits['image'], 32)
                self.assertNotIn('{model_path}', json.dumps(command))
                self.assertNotIn('{repo_root}', json.dumps(command))
        phi = launch.build_plan(self.args('phi4-multimodal'))['command']
        lora = json.loads(phi[phi.index('--default-mm-loras') + 1])
        self.assertEqual(lora['image'], str(Path('weights with spaces/"checkpoint"').resolve() / 'vision-lora'))

    def test_local_launch_plan_and_resource_guards(self):
        for profile in ['qwen35-4b', 'hy-embodied-05-mot-2b']:
            args = self.args(profile)
            args.dry_run = False
            validate_config(launch.build_plan(args)['config'])
        for extra in [('--tensor-parallel', '0'), ('--max-model-len', '500'),
                      ('--port', '65536'), ('--gpu-memory-utilization', '0')]:
            with self.assertRaises(ValueError):
                launch.build_plan(self.args('qwen35-4b', *extra))
        with self.assertRaisesRegex(ValueError, 'one GPU'):
            launch.build_plan(self.args('hy-embodied-05-mot-2b', '--tensor-parallel', '2'))

    def test_local_startup_generates_usable_configs_and_distinct_deployment_ids(self):
        with tempfile.TemporaryDirectory(prefix='deployment path ') as folder:
            weights = Path(folder) / 'weights'
            weights.mkdir()
            (weights / 'config.json').write_text('{}\n')
            identities = set()
            for index, profile in enumerate(['qwen35-a17b-fp8', 'hy-embodied-05-mot-2b',
                                             'qwen35-a17b-fp8']):
                with self.subTest(profile=profile, index=index):
                    output = Path(folder) / f'service-{index}'
                    argv = ['serve', '--profile', profile, '--model-path', str(weights),
                            '--output', str(output), '--python', sys.executable]
                    probe = SimpleNamespace(stdout='{"environment": "mock"}')
                    with patch.object(sys, 'argv', argv), \
                         patch.object(launch.subprocess, 'run', return_value=probe), \
                         patch.object(launch.os, 'execve') as execute, redirect_stdout(io.StringIO()):
                        launch.main()
                        config = validate_config(read_json(output / 'model.json'))
                        plan = read_json(output / 'deployment.json')
                        self.assertEqual(config['model_revision'],
                                         'local-deployment:' + plan['deployment_sha256'])
                        self.assertNotIn(config['model_revision'], identities)
                        identities.add(config['model_revision'])
                        self.assertEqual(plan['model_path'], str(weights.resolve()))
                        self.assertIn('checkpoint_config_sha256', plan)
                        command = execute.call_args.args[1]
                        self.assertIn(str(weights.resolve()), command)
                        self.assertEqual(command, plan['command'])
                        if profile == 'qwen35-a17b-fp8':
                            judge = validate_config(read_json(output / 'judge.json'))
                            self.assertEqual(judge['model_revision'], config['model_revision'])
                            self.assertEqual(judge['generation']['max_tokens'], 2048)
                        before = (output / 'deployment.json').read_bytes()
                        with self.assertRaises(SystemExit) as stopped:
                            launch.main()
                        self.assertEqual(stopped.exception.code, 2)
                        self.assertEqual((output / 'deployment.json').read_bytes(), before)
                        self.assertEqual(execute.call_count, 1)

    def test_endpoint_and_auth_are_portable_and_do_not_serialize_secrets(self):
        plan = launch.build_plan(self.args('qwen35-4b', '--host', '0.0.0.0', '--api-key-env', 'MODEL_KEY'))
        self.assertEqual(plan['config']['endpoint'], 'http://127.0.0.1:8000/v1/chat/completions')
        self.assertEqual(plan['config']['api_key_env'], 'MODEL_KEY')
        self.assertNotIn('--api-key', plan['command'])
        with self.assertRaises(ValueError):
            launch.build_plan(self.args('qwen35-4b', '--endpoint', 'https://user:secret@host/v1/chat/completions'))

    def test_shell_dry_run_from_other_directory_creates_nothing(self):
        with tempfile.TemporaryDirectory(prefix='deployment path ') as folder:
            result = subprocess.run(['bash', str(ROOT / 'deploy/serve.sh'), '--profile', 'internvl35-14b-thinking',
                '--model-path', 'missing weights', '--output', 'missing output', '--dry-run'],
                cwd=folder, capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads(result.stdout)
            self.assertIn('--reasoning-parser', plan['command'])
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_thinking_bundle_preserves_current_judge_and_other_prompts(self):
        default = read_json(ROOT / 'eval/embodied_eval/prompts.json')
        thinking = read_json(ROOT / 'configs/prompts/internvl35-14b-thinking.json')
        self.assertNotEqual(default['system'], thinking['system'])
        self.assertIn('<think>', thinking['system'])
        self.assertEqual({k: v for k, v in thinking.items() if k != 'system'},
                         {k: v for k, v in default.items() if k != 'system'})
        judge = validate_config(read_json(ROOT / 'configs/judge.local.example.json'))
        self.assertEqual(judge['generation']['max_tokens'], 2048)
        self.assertEqual(judge['reasoning_mode'], 'disabled')

    def test_internvl_adapter_obeys_final_boundary_and_strict_scoring(self):
        cases = {'analysis </think> A': 'A', '<think>first</think>reconsider</think>B': 'B',
                 '<think>unfinished': '', 'done</think><think>unfinished': '',
                 ' B ': 'B', 'B because it is closer': 'B because it is closer'}
        for raw, expected in cases.items():
            self.assertEqual(adapt_answer(raw, 'internvl_think_v1'), expected)
        scored = closed_score(dict(answer_format='single_choice', option_letters=['A', 'B']),
                              adapt_answer('analysis</think>B because...', 'internvl_think_v1'),
                              dict(canonical_gt='B'))
        self.assertEqual(scored['score'], 0)
        fixture = framework.FrameworkTests()
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        self.addCleanup(fixture.doCleanups)
        raw = 'reasoning</think>A'
        endpoint, _ = fixture.server([(200, fixture.response(raw))])
        cfg = read_json(ROOT / 'configs/models/internvl35-14b-thinking.json')
        cfg.update(endpoint=endpoint)
        result = Client(cfg).call([], fixture.root / 'response.json', {})
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['answer'], 'A')
        self.assertEqual(result['raw_answer'], raw)

    def test_hy_template_retains_system_and_32_ordered_images(self):
        template = Environment().from_string((ROOT / 'deploy/templates/hy-embodied-vlm-10.jinja').read_text())
        blocks = []
        for i in range(32):
            blocks.extend([{'type': 'text', 'text': f'frame {i}: '},
                           {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AA=='}}])
        messages = [{'role': 'system', 'content': 'System instruction.'}, {'role': 'user', 'content': blocks}]
        rendered = template.render(messages=messages, enable_thinking=False)
        self.assertEqual(rendered.count('<｜hy_image▁pad｜>'), 32)
        self.assertEqual(rendered.count('System instruction.'), 1)
        self.assertLess(rendered.index('frame 0:'), rendered.index('frame 31:'))
        self.assertIn('reasoning_effort:no_think', rendered)
        self.assertTrue(rendered.endswith('<think></think>'))
        self.assertTrue(template.render(messages=messages, enable_thinking=True).endswith('<think>'))

    def test_patch_hash_guard_rejects_unknown_sources_without_modification(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'vllm/model_executor/models/phi4mm.py'
            path.parent.mkdir(parents=True)
            path.write_text('different upstream source\n')
            with self.assertRaisesRegex(ValueError, 'hashes differ'):
                patcher.apply_patch('phi4mm-crops', folder)
            self.assertEqual(path.read_text(), 'different upstream source\n')

    def test_code_package_contains_deployment_but_excludes_runtime_assets(self):
        names = {p.relative_to(ROOT).as_posix() for p in packer.code_files(ROOT)}
        required = {'deploy/serve.sh', 'deploy/install.py', 'deploy/hy_mot_server.py',
                    'deploy/patches/hy-vlm10-vllm.patch', 'deploy/templates/hy-embodied-vlm-10.jinja',
                    'configs/models/phi4-multimodal.json', 'configs/prompts/internvl35-14b-thinking.json',
                    'deploy/licenses/Apache-2.0.txt', 'evaluate.sh', 'data/.gitkeep'}
        self.assertFalse(required - names)
        self.assertFalse(any(n.startswith(('images/', 'weights/', 'outputs/', '.venvs/')) for n in names))
        self.assertEqual({n for n in names if n.startswith('data/')}, {'data/.gitkeep'})
        self.assertNotIn('benchmark_data.jsonl', names)


class NativeServerTests(unittest.TestCase):
    def test_base64_conversion_preserves_order_pixels_and_native_template_call(self):
        blocks = []
        for color in [10, 90]:
            stream = io.BytesIO()
            Image.new('L', (4, 3), color).save(stream, format='PNG')
            blocks.append(dict(type='image_url', image_url=dict(url='data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode())))
        original = [dict(role='system', content='instruction'), dict(role='user', content=blocks)]
        messages, images = hy.convert_messages(original)
        self.assertEqual([b['image'].getpixel((0, 0)) for b in messages[1]['content']], [10, 90])
        self.assertEqual([b['image'].mode for b in messages[1]['content']], ['L', 'L'])
        self.assertEqual([i['size'] for i in images], [[4, 3], [4, 3]])
        calls = []
        class Processor:
            def apply_chat_template(self, value, **kwargs):
                calls.append((value, kwargs))
                return 'rendered'
            def __call__(self, **kwargs):
                return kwargs
        inputs = hy.prepare_inputs(Processor(), messages)
        self.assertFalse(calls[0][1]['tokenize'])
        self.assertFalse(calls[0][1]['enable_thinking'])
        self.assertFalse(inputs['add_special_tokens'])
        self.assertEqual(len(inputs['images']), 2)
        with self.assertRaisesRegex(ValueError, '32 images'):
            hy.convert_messages([dict(role='user', content=[blocks[0]] * 33)])
        with self.assertRaisesRegex(ValueError, 'base64'):
            hy.convert_messages([dict(role='user', content=[dict(type='image_url', image_url=dict(url='https://host/image.png'))])])

    def test_native_http_auth_generation_and_overflow_without_gpu(self):
        # Substitute only tensor/model computation. Exercise the real FastAPI routes,
        # request validation, template processing, response serialization and audits.
        class Tokens:
            shape = (1, 3)
            def __eq__(self, value):
                return SimpleNamespace(sum=lambda: 0)
            def numpy(self):
                return SimpleNamespace(tobytes=lambda: b'input token bytes')
        class Inputs(dict):
            def to(self, device):
                return self
        class Output:
            def __getitem__(self, key):
                return SimpleNamespace(tolist=lambda: [20, 2])
        class Processor:
            image_token_id = 999
            image_processor = SimpleNamespace(to_dict=lambda: {}, merge_size=2)
            def apply_chat_template(self, messages, **kwargs):
                return 'native prompt'
            def __call__(self, **kwargs):
                return Inputs(input_ids=Tokens())
            def decode(self, tokens, skip_special_tokens):
                return '<answer>A</answer>' if skip_special_tokens else '<answer>A</answer><eos>'
        model = SimpleNamespace(config=SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=4096)),
                                generation_config=SimpleNamespace(eos_token_id=2), device='cpu',
                                generate=lambda **kwargs: Output())
        torch = SimpleNamespace(__version__='mock', inference_mode=nullcontext, manual_seed=lambda seed: None)
        with tempfile.TemporaryDirectory() as folder, patch.dict(sys.modules, torch=torch,
                                                                 transformers=SimpleNamespace(__version__='mock')):
            app = hy.create_app(model, Processor(), 'hy-embodied-05-mot-2b', folder, api_key='test-key')
            client = TestClient(app)
            cfg = read_json(ROOT / 'configs/models/hy-embodied-05-mot-2b.json')
            payload = dict(model=cfg['model'], messages=[dict(role='user', content='question')],
                           stream=False, **cfg['generation'], **cfg['request_extra'])
            endpoint = '/v1/chat/completions'
            headers = {'Authorization': 'Bearer test-key'}
            self.assertEqual(client.post(endpoint, json=payload).status_code, 401)
            response = client.post(endpoint, json=payload, headers=headers)
            self.assertEqual(response.status_code, 200, response.text)
            data = response.json()
            self.assertEqual(data['choices'][0]['finish_reason'], 'stop')
            self.assertEqual(adapt_answer(data['choices'][0]['message']['content'], cfg['answer_adapter']), 'A')
            self.assertEqual(data['usage']['total_tokens'], 5)
            for extra in [dict(max_tokens=2048), dict(stream=True), dict(temperature=0.7), dict(tools=[])]:
                self.assertEqual(client.post(endpoint, json=dict(payload, **extra), headers=headers).status_code, 400)
            # Context is frozen when creating the app; build a small-context backend.
            model.config.text_config.max_position_embeddings = 1025
            tiny = TestClient(hy.create_app(model, Processor(), cfg['model'], folder))
            response = tiny.post(endpoint, json=payload)
            self.assertEqual(response.status_code, 400)
            self.assertIn('without truncation', response.text)
            self.assertTrue(list((Path(folder) / 'request_audits').glob('*.json')))


if __name__ == '__main__':
    unittest.main()
