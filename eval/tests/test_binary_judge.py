"""The released judge only accepts binary verdicts and enforces its protocol contract."""
import json
import unittest

from embodied_eval import JUDGE_PROTOCOL_VERSION
from embodied_eval.common import file_hash, read_json, read_jsonl, write_json
from embodied_eval.scoring import judge, parse_judgment, score
from embodied_eval.serving import run
import test_framework as framework


class BinaryJudgeTests(unittest.TestCase):
    def setUp(self):
        self.fixture = framework.FrameworkTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.addCleanup(self.fixture.doCleanups)

    def test_only_binary_integer_scores_are_accepted(self):
        decision = self.fixture.decision()
        self.assertEqual(parse_judgment(json.dumps(decision), 1)['score'], 1)
        decision.update(verdict='incorrect', score=0, incorrect_claims=['Wrong object.'])
        self.assertEqual(parse_judgment(json.dumps(decision), 1)['score'], 0)
        for verdict, value in [('unsupported', None), ('unsupported', 0),
                               ('correct', None), ('incorrect', None), ('correct', True),
                               ('incorrect', False), ('correct', 1.0), ('correct', '1')]:
            with self.subTest(verdict=verdict, score=value), self.assertRaises(ValueError):
                parse_judgment(json.dumps(dict(decision, verdict=verdict, score=value)), 1)

    def completed_run(self):
        f = self.fixture
        corpus = f.corpus([f.record('action_sequence_description', 'Lift the cup.'),
                           f.record('adjacent_action', 'Lift the cup.')])
        correct = f.decision()
        incorrect = f.decision(verdict='incorrect', score=0, incorrect_claims=['Wrong object.'])
        endpoint, received = f.server([(200, f.response('Lift the cup.')),
                                       (200, f.response('Lift the bowl.')),
                                       (200, f.response(json.dumps(correct))),
                                       (200, f.response(json.dumps(incorrect)))])
        config = f.root/'config.json'
        write_json(config, dict(f.config(endpoint), workers=1))
        run(corpus, f.root/'run', config)
        judge(corpus, f.root/'run', f.root/'judge', config)
        return corpus, received

    def test_complete_binary_report_has_final_accuracy(self):
        f = self.fixture
        corpus, received = self.completed_run()
        report = score(corpus, f.root/'run', f.root/'scores', f.root/'judge')
        self.assertEqual(report['status'], 'complete')
        self.assertEqual(report['open_micro']['accuracy'], 0.5)
        self.assertEqual(report['official_all_accuracy'], 0.5)
        self.assertEqual(report['open_statuses'], {'scored': 2})
        self.assertEqual(report['judge']['judge_protocol_version'], JUDGE_PROTOCOL_VERSION)
        self.assertEqual([r['score'] for r in read_jsonl(f.root/'scores/scores.jsonl')], [1, 0])
        self.assertEqual(read_jsonl(f.root/'scores/judge_errors.jsonl'), [])
        self.assertEqual(report['overall']['accuracy'], 0.5)
        self.assertEqual(len(received), 4)

    def test_mismatched_judge_protocol_is_rejected(self):
        f = self.fixture
        corpus, _ = self.completed_run()
        path = f.root/'judge/run.json'
        manifest = read_json(path)
        manifest['identity']['judge_protocol_version'] = 'invalid-protocol'
        write_json(path, manifest)
        with self.assertRaisesRegex(ValueError, 'judge protocol mismatch'):
            score(corpus, f.root/'run', f.root/'scores', f.root/'judge')

    def test_missing_prompt_protocol_fails_before_judge_request(self):
        f = self.fixture
        corpus = f.corpus([f.record('action_sequence_description', 'Lift the cup.')])
        prompts = read_json(corpus/'prompts.json')
        prompts.pop('judge_protocol_version')
        write_json(corpus/'prompts.json', prompts)
        manifest = read_json(corpus/'manifest.json')
        manifest['files']['prompts.json'] = file_hash(corpus/'prompts.json')
        write_json(corpus/'manifest.json', manifest)
        endpoint, received = f.server([(200, f.response('Lift the cup.'))])
        config = f.root/'config.json'
        write_json(config, f.config(endpoint))
        run(corpus, f.root/'run', config)
        with self.assertRaisesRegex(ValueError, 'different judge protocol'):
            judge(corpus, f.root/'run', f.root/'judge', config)
        self.assertEqual(len(received), 1)

    def test_invalid_verdict_is_an_unscored_protocol_error(self):
        f = self.fixture
        corpus = f.corpus([f.record('action_sequence_description', 'Lift the cup.')])
        invalid = f.decision(verdict='unsupported', score=None)
        endpoint, received = f.server([(200, f.response('Lift the cup.')),
                                       (200, f.response(json.dumps(invalid)))])
        config = f.root/'config.json'
        write_json(config, f.config(endpoint))
        run(corpus, f.root/'run', config)
        judge(corpus, f.root/'run', f.root/'judge', config)
        report = score(corpus, f.root/'run', f.root/'scores', f.root/'judge')
        self.assertEqual(report['status'], 'incomplete')
        self.assertEqual(report['open_statuses'], {'judge_protocol_error': 1})
        self.assertEqual(report['open_micro']['scored'], 0)
        self.assertIsNone(report['official_all_accuracy'])
        self.assertEqual(len(read_jsonl(f.root/'scores/judge_errors.jsonl')), 1)
        self.assertEqual(len(received), 2)


if __name__ == '__main__':
    unittest.main()
