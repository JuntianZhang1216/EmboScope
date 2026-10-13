"""Exercise the shell workflow against the real CLI and a localhost mock service."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest

from embodied_eval.common import read_json, write_json, write_jsonl
import test_framework as framework


ENTRYPOINT = Path(__file__).resolve().parents[2] / 'evaluate.sh'


class ShellEntrypointTests(unittest.TestCase):
    def setUp(self):
        self.fixture = framework.FrameworkTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        self.checkout = f.root / 'checkout with spaces'
        (self.checkout / 'data/images').mkdir(parents=True)
        shutil.copytree(ENTRYPOINT.parent / 'eval/embodied_eval', self.checkout / 'eval/embodied_eval',
                        ignore=shutil.ignore_patterns('__pycache__'))
        for name in ('evaluate.sh', 'pyproject.toml'):
            shutil.copyfile(ENTRYPOINT.parent / name, self.checkout / name)
        shutil.copyfile(f.image, self.checkout / 'data/images/frame.png')
        self.dataset = self.checkout / 'data/benchmark_data.jsonl'
        self.output = f.root / 'workflow with spaces'
        self.model_config = f.root / 'model config.json'
        self.judge_config = f.root / 'judge config.json'
        self.cwd = f.root / 'another working directory'
        self.cwd.mkdir()
        rows = [f.record(), f.record('action_sequence_description', 'Lift the cup.')]
        write_jsonl(self.dataset, [dict(row, id=f'q{i:06d}') for i, row in enumerate(rows, 1)])

    def service(self, responses):
        endpoint, received = self.fixture.server(responses)
        config = dict(self.fixture.config(endpoint), workers=1, max_attempts=1)
        write_json(self.model_config, config)
        write_json(self.judge_config, config)
        return received

    def invoke(self, *args):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        env.pop('EMBOSCOPE_TEST_MISSING_KEY', None)
        return subprocess.run(
            ['bash', str(self.checkout / 'evaluate.sh'), '--python', sys.executable,
             '--output', str(self.output), *map(str, args)],
            cwd=self.cwd, env=env, capture_output=True, text=True, timeout=45)

    def configs(self):
        return ['--model-config', self.model_config, '--judge-config', self.judge_config]

    def test_full_workflow_and_resume_from_another_directory(self):
        f = self.fixture
        # A different working directory cannot redirect the fixed dataset.
        (self.cwd / 'data').mkdir()
        (self.cwd / 'data/benchmark_data.jsonl').write_text('not the benchmark\n')
        received = self.service([(200, f.response('A')), (200, f.response('Lift the cup.')),
                                 (200, f.response(json.dumps(f.decision())))])
        for _ in range(2):
            result = self.invoke(*self.configs())
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(received), 3)

        report = read_json(self.output / 'scores/report.json')
        self.assertEqual(report['status'], 'complete')
        self.assertEqual(report['overall']['accuracy'], 1)
        self.assertEqual(set(p.name for p in (self.output / 'logs').iterdir()),
                         {'prepare.log', 'run.log', 'judge.log', 'score.log'})
        result = self.invoke('--stage', 'score')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(received), 3)

    def test_missing_downloads_fail_before_requests(self):
        received = self.service([(200, self.fixture.response())])
        self.dataset.rename(self.dataset.with_suffix('.saved'))
        result = self.invoke(*self.configs())
        self.assertEqual(result.returncode, 2)
        self.assertIn('missing data/benchmark_data.jsonl', result.stderr)
        self.dataset.with_suffix('.saved').rename(self.dataset)
        images = self.checkout / 'data/images'
        images.rename(self.checkout / 'data/images.saved')
        result = self.invoke(*self.configs())
        self.assertEqual(result.returncode, 2)
        self.assertIn('missing data/images/', result.stderr)
        self.assertEqual(received, [])
        self.assertFalse(self.output.exists())

    def test_dataset_override_is_rejected_by_both_entrypoints(self):
        result = self.invoke('--stage', 'prepare', '--dataset', self.dataset)
        self.assertEqual(result.returncode, 2)
        self.assertIn('unknown argument: --dataset', result.stderr)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(self.checkout / 'eval'))
        result = subprocess.run([sys.executable, '-m', 'embodied_eval', 'prepare',
                                 '--dataset', str(self.dataset), '--output', str(self.output)],
                                cwd=self.cwd, env=env, text=True, capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 2)
        self.assertIn('unrecognized arguments: --dataset', result.stderr)
        self.assertFalse(self.output.exists())

    def test_incomplete_inference_stops_before_judging_and_can_resume(self):
        f = self.fixture
        write_jsonl(self.dataset, [f.record('action_sequence_description', 'Lift the cup.')])
        received = self.service([(503, {'error': 'temporary failure'}),
                                 (200, f.response('Lift the cup.')),
                                 (200, f.response(json.dumps(f.decision())))])
        result = self.invoke(*self.configs())
        self.assertEqual(result.returncode, 3, result.stdout + result.stderr)
        self.assertEqual(len(received), 1)
        self.assertFalse((self.output / 'judge/run.json').exists())
        self.assertEqual(read_json(self.output / 'scores/report.json')['status'], 'incomplete')
        result = self.invoke(*self.configs(), '--retry-pending')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(received), 3)
        self.assertEqual(read_json(self.output / 'scores/report.json')['status'], 'complete')

    def test_judge_failure_returns_incomplete_without_resampling(self):
        f = self.fixture
        received = self.service([(200, f.response('A')), (200, f.response('Lift the cup.')),
                                 (200, f.response('invalid JSON'))])
        for _ in range(2):
            result = self.invoke(*self.configs())
            self.assertEqual(result.returncode, 3, result.stdout + result.stderr)
        self.assertEqual(len(received), 3)
        report = read_json(self.output / 'scores/report.json')
        self.assertEqual(report['open_statuses'], {'judge_protocol_error': 1})
        self.assertEqual(report['overall']['pending'], 1)
        self.assertIsNone(report['official_all_accuracy'])

    def test_selection_changes_are_rejected_before_inference(self):
        f = self.fixture
        received = self.service([(200, f.response())])
        result = self.invoke('--stage', 'prepare', '--per-task', '1')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        result = self.invoke(*self.configs())
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Prepared corpus mismatch (selection)', result.stderr)
        self.assertEqual(received, [])

    def test_custom_prompt_is_sent_and_must_match_on_resume(self):
        f = self.fixture
        prompts = ENTRYPOINT.parent / 'configs/prompts/internvl35-14b-thinking.json'
        received = self.service([(200, f.response('A')), (200, f.response('Lift the cup.'))])
        args = ['--model-config', self.model_config, '--skip-judge', '--prompts-path', prompts]
        result = self.invoke(*args)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(received[0]['messages'][0]['content'], read_json(prompts)['system'])
        result = self.invoke('--model-config', self.model_config, '--skip-judge')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Prepared corpus mismatch (prompts)', result.stderr)
        self.assertEqual(len(received), 2)
        result = self.invoke(*args)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(received), 2)

    def test_judge_configuration_is_checked_before_model_requests(self):
        received = self.service([(200, self.fixture.response())])
        config = read_json(self.judge_config)
        config['api_key_env'] = 'EMBOSCOPE_TEST_MISSING_KEY'
        write_json(self.judge_config, config)
        result = self.invoke(*self.configs())
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('missing API key environment variable', result.stderr)
        self.assertEqual(received, [])
        self.assertFalse(self.output.exists())

    def test_explicit_closed_scoring_does_not_require_a_judge(self):
        f = self.fixture
        received = self.service([(200, f.response('A')), (200, f.response('Lift the cup.'))])
        result = self.invoke('--model-config', self.model_config, '--skip-judge')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = read_json(self.output / 'scores/report.json')
        self.assertEqual(report['closed_micro']['accuracy'], 1)
        self.assertEqual(report['status'], 'incomplete')
        self.assertIsNone(report['official_all_accuracy'])
        self.assertIn('Closed-question report:', result.stdout)
        self.assertFalse((self.output / 'judge').exists())
        self.assertEqual(len(received), 2)


if __name__ == '__main__':
    unittest.main()
