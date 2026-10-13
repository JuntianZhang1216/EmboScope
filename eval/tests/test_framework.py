import base64
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image
from embodied_eval.common import read_json, read_jsonl, write_json, write_jsonl, digest
from embodied_eval.inputs import prepare, compile_record, gold, inspect_image
from embodied_eval.serving import run, validate_config, solve_messages, Client
from embodied_eval.scoring import closed_score, parse_yes_no, parse_judgment, score, judge, validate_facts
from embodied_eval.answer_adapters import adapt_answer

ROOT = Path(__file__).resolve().parents[1]


class FrameworkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root/'images').mkdir()
        self.image = self.root/'images/frame.png'
        Image.new('RGB', (8, 8), (40, 60, 90)).save(self.image)
        self.prompts = read_json(ROOT/'embodied_eval/prompts.json')

    def tearDown(self):
        self.temp.cleanup()

    def record(self, task='relative_distance', reference='first'):
        return dict(id='q000001', task=task, text_list=['Choose.\nA. first\nB. second', reference],
                    image_list=[dict(image='images/frame.png', matched_text_index=0)], role_list=['human', 'human', 'gpt'])

    def corpus(self, records=None):
        records = [dict(record, id=f'q{i:06d}') for i, record in enumerate(records or [self.record()], 1)]
        path = self.root/'dataset.jsonl'
        write_jsonl(path, records)
        corpus = self.root/'corpus'
        prepare(path, ROOT/'embodied_eval/task_catalog.json', ROOT/'embodied_eval/prompts.json', corpus)
        return corpus

    def config(self, endpoint):
        return dict(endpoint=endpoint, model='test', model_revision='test-v1', reasoning_mode='disabled',
                    generation=dict(temperature=0, max_tokens=1024), workers=2, requests_per_second=1000,
                    request_extra=dict(chat_template_kwargs=dict(enable_thinking=False)), max_attempts=2)

    def server(self, responses):
        received = []
        class Handler(BaseHTTPRequestHandler):
            def do_POST(inner):
                received.append(json.loads(inner.rfile.read(int(inner.headers['Content-Length']))))
                status, data = responses[min(len(received)-1, len(responses)-1)]
                inner.send_response(status)
                inner.send_header('Content-Type', 'application/json')
                inner.send_header('Retry-After', '0')
                inner.end_headers()
                inner.wfile.write(json.dumps(data).encode())
            def log_message(*args):
                pass
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return 'http://127.0.0.1:%d/v1/chat/completions' % server.server_port, received

    def response(self, answer='A', finish='stop'):
        return dict(model='test-v1', choices=[dict(message=dict(content=answer), finish_reason=finish)], usage={})

    def test_model_version_is_optional_for_inference_and_frozen_when_supplied(self):
        endpoint, received = self.server([(200, self.response())])
        config = self.config(endpoint)
        del config['model_revision']
        corpus = self.corpus()
        output = self.root / 'run'
        config_path = self.root / 'config.json'
        write_json(config_path, config)
        run(corpus, output, config_path)
        run(corpus, output, config_path)
        self.assertEqual(len(received), 1)
        changed = dict(config, model_revision='deployment-v2')
        write_json(config_path, changed)
        with self.assertRaises(ValueError):
            run(corpus, output, config_path)
        self.assertEqual(len(received), 1)
        self.assertEqual(validate_config(changed)['model_revision'], 'deployment-v2')
        for value in ['', ' ', None, 2]:
            with self.assertRaisesRegex(ValueError, 'model_revision'):
                validate_config(dict(config, model_revision=value))

    def test_strict_choice_and_gt_mapping(self):
        record = self.record(reference='second')
        ref = gold(record, 'single_choice')
        self.assertEqual(ref['canonical_gt'], 'B')
        item = dict(answer_format='single_choice', option_letters=['A', 'B'])
        for answer in ['b', 'B.', '(B)', 'Answer: B', 'second', 'B because...', 'B A', 'Ｂ', '']:
            result = closed_score(item, answer, ref)
            self.assertEqual(result['score'], 0, answer)
            self.assertFalse(result['format_valid'], answer)
        self.assertEqual(closed_score(item, ' B\n', ref)['score'], 1)
        record['text_list'][0] = 'A. same\nB. same'
        record['text_list'][-1] = 'same'
        with self.assertRaises(ValueError):
            gold(record, 'single_choice')

    def test_yes_no_allowlist(self):
        for text in ['yes', ' Yes. ', 'Answer: NO！']:
            self.assertIsNotNone(parse_yes_no(text))
        for text in ['True', '1', 'Yes because it is done', 'Yes/No', 'No!!']:
            self.assertIsNone(parse_yes_no(text))

    def test_glm_adapter_only_unwraps_one_complete_envelope(self):
        box = lambda text: '<|begin_of_box|>' + text + '<|end_of_box|>'
        for text in ['B', 'Yes', 'First pick up the cup.\nThen place it down.']:
            self.assertEqual(adapt_answer(box(text), 'glm_box_v1'), text)
            self.assertEqual(adapt_answer(box(text)), box(text))
        for text in [box('A') + box('B'), 'Explanation ' + box('B'),
                     box(box('B')), '<|begin_of_box|>B', 'B<|end_of_box|>']:
            self.assertEqual(adapt_answer(text, 'glm_box_v1'), text)
        item = dict(answer_format='single_choice', option_letters=['A', 'B'])
        self.assertFalse(closed_score(item, adapt_answer(box('B because...'), 'glm_box_v1'),
                                      {'canonical_gt': 'B'})['format_valid'])

    def test_glm_online_adapter_preserves_raw_response(self):
        raw = '<|begin_of_box|>A<|end_of_box|>'
        endpoint, received = self.server([(200, self.response(raw))])
        config = self.config(endpoint)
        config['answer_adapter'] = 'glm_box_v1'
        result = Client(config).call([], self.root/'response.json', {})
        self.assertEqual(result['answer'], 'A')
        self.assertEqual(result['raw_answer'], raw)
        self.assertEqual(result['raw_response']['choices'][0]['message']['content'], raw)
        self.assertEqual(len(received), 1)

    def test_embodied_online_adapters_preserve_raw_and_strict_scoring(self):
        cases = [('robobrain_answer_v1', '<think></think><answer>A</answer>', 'A'),
                 ('hy_answer_v1', '<answer>A</answer>', 'A'),
                 ('hy_answer_v1', '<answer>A because...</answer>', 'A because...'),
                 ('mimo_think_v1', '<think>Compare both images.</think>\nA', 'A'),
                 ('mimo_think_v1', '<think>Unfinished reasoning says A', ''),
                 ('robobrain_answer_v1', '<answer>A because...</answer>', 'A because...')]
        for index, (adapter, raw, expected) in enumerate(cases):
            with self.subTest(adapter=adapter, raw=raw):
                endpoint, received = self.server([(200, self.response(raw))])
                config = self.config(endpoint)
                config.update(reasoning_mode='provider_default', request_extra={}, answer_adapter=adapter)
                result = Client(config).call([], self.root/f'adapter_{index}.json', {})
                self.assertEqual(result['answer'], expected)
                self.assertEqual(result['raw_answer'], raw)
                self.assertEqual(result['raw_response']['choices'][0]['message']['content'], raw)
                self.assertEqual(len(received), 1)
                item = dict(answer_format='single_choice', option_letters=['A', 'B'])
                graded = closed_score(item, result['answer'], {'canonical_gt': 'A'})
                self.assertEqual(graded['format_valid'], expected == 'A')

    def test_last_anchor_preserved_without_gold(self):
        record = dict(id='q000001', task='action_frame_matching', text_list=['Question', 'A', 'B', 'PRIVATE_GOLD_SENTINEL'],
                      image_list=[dict(image=str(self.image), matched_text_index=a) for a in [2, 3]],
                      role_list=['human']*5+['gpt'])
        media = [dict(m, path=m['image']) for m in record['image_list']]
        info = {str(self.image): inspect_image(self.image)}
        item = compile_record(record, 1, 'single_choice', self.prompts, media, info)
        payload = solve_messages(item)
        self.assertNotIn('PRIVATE_GOLD_SENTINEL', json.dumps(payload))
        self.assertNotIn(str(self.image), json.dumps(payload))
        content = payload[1]['content']
        self.assertEqual(sum(b['type']=='image_url' for b in content), 2)
        labels = [b['text'] for b in content if b['type']=='text']
        self.assertIn('image_index=1; candidate option=B; original label=B', labels)

    def test_end_to_end_resume_and_fixed_denominator(self):
        records = [self.record(reference='first'), self.record(reference='second')]
        corpus = self.corpus(records)
        endpoint, received = self.server([(200, self.response('A'))])
        config = self.root/'config.json'
        write_json(config, self.config(endpoint))
        output = self.root/'run'
        run(corpus, output, config, limit=1)
        partial = score(corpus, output, self.root/'scores')
        self.assertEqual(partial['closed_micro']['accuracy_lower'], .5)
        self.assertEqual(partial['closed_micro']['accuracy_upper'], 1)
        self.assertIsNone(partial['closed_micro']['accuracy'])
        run(corpus, output, config)
        run(corpus, output, config)
        self.assertEqual(len(received), 2)
        report = score(corpus, output, self.root/'scores')
        self.assertEqual(report['closed_micro']['accuracy'], .5)
        self.assertFalse(received[0]['chat_template_kwargs']['enable_thinking'])
        self.assertNotIn('reference_answer', json.dumps(received))
        self.assertEqual(base64.b64decode(received[0]['messages'][1]['content'][1]['image_url']['url'].split(',')[1]), self.image.read_bytes())
        modified = self.config(endpoint); modified['generation']['temperature'] = .5
        write_json(config, modified)
        with self.assertRaises(ValueError):
            run(corpus, output, config)

    def test_http_retry_and_format_failure_no_retry(self):
        corpus = self.corpus()
        endpoint, received = self.server([(429, {}), (200, self.response('Answer: A'))])
        config = self.root/'config.json';write_json(config, self.config(endpoint))
        run(corpus, self.root/'run', config)
        self.assertEqual(len(received), 2)
        report = score(corpus, self.root/'run', self.root/'scores')
        self.assertEqual(report['closed_micro']['accuracy'], 0)
        self.assertEqual(report['closed_micro']['format_failures'], 1)
        run(corpus, self.root/'run', config, retry_pending=True)
        self.assertEqual(len(received), 2)

    def test_media_mutation_is_pending_not_model_failure(self):
        corpus = self.corpus()
        self.image.write_bytes(b'changed')
        endpoint, received = self.server([(200, self.response())])
        config = self.root/'config.json';write_json(config, self.config(endpoint))
        run(corpus, self.root/'run', config)
        self.assertEqual(len(received), 0)
        report = score(corpus, self.root/'run', self.root/'scores')
        self.assertEqual(report['closed_micro']['pending'], 1)

    def test_unfinished_request_never_silently_resent(self):
        corpus = self.corpus()
        endpoint, received = self.server([(200, self.response())])
        config = self.root/'config.json';write_json(config, self.config(endpoint))
        out = self.root/'run'
        run(corpus, out, config)
        path = out/'responses/q000001.json'; old = read_json(path);old['status']='in_flight';write_json(path, old)
        run(corpus, out, config, retry_pending=True)
        self.assertEqual(len(received), 1)
        run(corpus, out, config, retry_unknown=True)
        self.assertEqual(len(received), 2)
        self.assertEqual(len(list((out/'retry_history/q000001').glob('*.json'))), 1)

    def test_judge_schema_and_anonymity(self):
        record = self.record('action_sequence_description', 'Pick up the cup.')
        record['text_list'][0] = 'Describe the action sequence.'
        corpus = self.corpus([record])
        decision = dict(verdict='incorrect', score=0, reason='Required action is not established.', missing_required_events=['pick up the cup'],
                        incorrect_claims=[], relation_errors=[], evidence_refs=[])
        endpoint, received = self.server([(200, self.response('Pick up the cup.')), (200, self.response(json.dumps(decision)))])
        config = self.root/'config.json';write_json(config, self.config(endpoint))
        run(corpus, self.root/'run', config)
        judge(corpus, self.root/'run', self.root/'judge', config)
        payload = received[1]['messages']
        self.assertNotIn('requested_model', json.dumps(payload))
        self.assertIn('reference_answer', json.dumps(payload))
        report = score(corpus, self.root/'run', self.root/'scores', self.root/'judge')
        self.assertEqual(report['overall']['pending'], 0)
        self.assertEqual(report['official_all_accuracy'], 0)
        decision.update(verdict='correct', score=1, missing_required_events=[], evidence_refs=[dict(image_index=8, observation='wrong index')])
        with self.assertRaises(ValueError):
            parse_judgment(json.dumps(decision), 1)
        decision['evidence_refs'][0]['image_index'] = 0
        self.assertEqual(parse_judgment(json.dumps(decision), 1)['score'], 1)
        decision['score'] = True
        with self.assertRaises(ValueError):
            parse_judgment(json.dumps(decision), 1)

    def test_task_specific_frame_labels(self):
        from embodied_eval.inputs import label
        self.assertEqual(label({'task': 'cross_view_matching'}, 2, {'matched_text_index': 0}), 'image_index=2; frame label=c')
        self.assertEqual(label({'task': 'attribute_matching'}, 2, {'matched_text_index': 0}), 'image_index=2; frame 2')
        self.assertEqual(label({'task': 'step_frame_alignment'}, 2, {'matched_text_index': 0}), 'image_index=2; strip label=C')

    def test_canary_covers_each_task_then_resumes(self):
        record = self.record('step_completion', 'Yes')
        record['text_list'][0] = 'Is the step complete?'
        corpus = self.corpus([self.record(), self.record(), record])
        endpoint, received = self.server([(200, self.response('A'))])
        config = self.root/'config.json';write_json(config, self.config(endpoint))
        run(corpus, self.root/'run', config, canary_per_task=1)
        ids = {p.stem for p in (self.root/'run/responses').glob('*.json')}
        self.assertEqual(ids, {'q000001', 'q000003'})
        run(corpus, self.root/'run', config)
        self.assertEqual(len(received), 3)

    def test_thinking_disabled_violation_stops_runner(self):
        corpus = self.corpus()
        response = self.response('A')
        response['choices'][0]['message']['reasoning_content'] = 'unexpected thinking'
        endpoint, received = self.server([(200, response)])
        config = self.root/'config.json';write_json(config, self.config(endpoint))
        result = run(corpus, self.root/'run', config)
        self.assertTrue(result['aborted'])
        self.assertEqual(result['statuses'], {'protocol_error': 1})

    def test_null_final_content_is_empty_model_answer(self):
        corpus = self.corpus()
        endpoint, received = self.server([(200, self.response(None))])
        config = self.root/'config.json';write_json(config, self.config(endpoint))
        run(corpus, self.root/'run', config)
        report = score(corpus, self.root/'run', self.root/'scores')
        self.assertEqual(report['closed_micro']['accuracy'], 0)
        self.assertEqual(report['closed_micro']['pending'], 0)
        self.assertTrue((self.root/'run/code_snapshot/serving.py').exists())

    def test_null_message_is_durable_error_and_runner_continues(self):
        malformed = self.response()
        malformed['choices'][0]['message'] = None
        endpoint, received = self.server([(200, malformed), (200, self.response('B'))])
        config = dict(self.config(endpoint), workers=1)
        cfg = self.root/'config.json'; write_json(cfg, config)
        corpus = self.corpus([self.record(), self.record()])
        output = self.root/'run'
        progress = run(corpus, output, cfg)
        self.assertEqual(progress['statuses'], {'protocol_error': 1, 'success': 1})
        self.assertFalse(progress['aborted'])
        bad = read_json(output/'responses/q000001.json')
        self.assertEqual(bad['raw_response'], malformed)
        self.assertEqual(bad['error_detail'], 'expected message object')
        self.assertEqual(len(bad['attempts']), 1)
        self.assertIn('finished_at', bad)
        run(corpus, output, cfg)
        self.assertEqual(len(received), 2)  # No implicit resampling of protocol errors.

    def test_malformed_response_shapes_are_saved_without_crashing(self):
        cases = [None, [], {'choices': None}, {'choices': {}}, {'choices': []},
                 {'choices': [None]}, {'choices': [{'message': []}]},
                 {'choices': [{'message': 'invalid'}]}, {'choices': [{}]}]
        endpoint, received = self.server([(200, raw) for raw in cases])
        for idx, raw in enumerate(cases):
            with self.subTest(raw=raw):
                path = self.root/f'malformed_{idx}.json'
                result = Client(self.config(endpoint)).call([], path, {})
                self.assertEqual(result['status'], 'protocol_error')
                self.assertEqual(read_json(path)['raw_response'], raw)
                self.assertEqual(len(result['attempts']), 1)
        self.assertEqual(len(received), len(cases))

    def test_message_override_rejected(self):
        c = self.config('http://localhost/v1/chat/completions')
        c['request_extra']['messages'] = []
        with self.assertRaises(ValueError):
            validate_config(c)

    def test_replica_round_robin_and_attempt_identity(self):
        first, a = self.server([(200, self.response('A'))])
        second, b = self.server([(200, self.response('B'))])
        cfg = self.config(first)
        cfg['endpoints'] = [first, second]
        client = Client(cfg)
        results = [client.call([], self.root / f'r{i}.json', dict(id=str(i))) for i in range(4)]
        self.assertEqual([r['answer'] for r in results], ['A', 'B', 'A', 'B'])
        self.assertEqual((len(a), len(b)), (2, 2))
        self.assertEqual([r['attempts'][0]['endpoint'] for r in results], [first, second, first, second])

    def test_explicit_http_retry_can_use_another_replica(self):
        first, a = self.server([(429, {})])
        second, b = self.server([(200, self.response('B'))])
        cfg = self.config(first)
        cfg['endpoints'] = [first, second]
        result = Client(cfg).call([], self.root/'retry.json', dict(id='retry'))
        self.assertEqual(result['answer'], 'B')
        self.assertEqual([x['endpoint'] for x in result['attempts']], [first, second])
        self.assertEqual((len(a), len(b)), (1, 1))

    def test_unknown_transport_is_not_replayed_on_replica(self):
        from unittest.mock import patch
        import requests
        cfg = self.config('http://localhost:12345/v1/chat/completions')
        cfg['endpoints'] = [cfg['endpoint'], 'http://localhost:12346/v1/chat/completions']
        with patch('requests.Session.post', side_effect=requests.Timeout) as post:
            result = Client(cfg).call([], self.root/'timeout.json', dict(id='timeout'))
        self.assertEqual(result['status'], 'transport_unknown')
        self.assertEqual(post.call_count, 1)

    def test_invalid_replica_lists_rejected(self):
        cfg = self.config('http://localhost:12345/v1/chat/completions')
        for endpoints in [[], 'not-a-list', [cfg['endpoint'], cfg['endpoint']],
                          ['http://another/v1/chat/completions'], [cfg['endpoint'], 'http://user:pass@host/v1/chat/completions']]:
            with self.subTest(endpoints=endpoints), self.assertRaises(ValueError):
                validate_config(dict(cfg, endpoints=endpoints))

    def decision(self, **changes):
        d = dict(verdict='correct', score=1, reason='The required action is visible.',
                 missing_required_events=[], incorrect_claims=[], relation_errors=[],
                 evidence_refs=[dict(image_index=0, observation='The cup is lifted.')])
        d.update(changes)
        return d

    def test_judge_rejects_contradictory_and_malformed_verdicts(self):
        changes = [dict(verdict=[]), dict(score=True), dict(score=1.0),
                   dict(missing_required_events=['missing action']),
                   dict(incorrect_claims=['wrong object']), dict(relation_errors=['wrong order']),
                   dict(evidence_refs=[]), dict(reason=' '),
                   dict(evidence_refs=[dict(image_index=-1, observation='x')]),
                   dict(evidence_refs=[dict(image_index=True, observation='x')]),
                   dict(evidence_refs=[dict(image_index=0, observation=' ')])]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                parse_judgment(json.dumps(self.decision(**change)), 1)
        for text in ['```json\n' + json.dumps(self.decision()) + '\n```',
                     json.dumps(self.decision())[:-1] + ', "score": 1}']:
            with self.assertRaises(ValueError):
                parse_judgment(text, 1)

    def test_judge_e2e_valid_scores_resume_and_frozen_snapshot(self):
        r = self.record('action_sequence_description', 'Lift the cup.')
        r['text_list'][0] = 'Describe the action sequence.'
        corpus = self.corpus([r, r])
        endpoint, received = self.server([(200, self.response('Pick up the cup.')),
                                         (200, self.response(json.dumps(self.decision())))])
        config = self.root/'config.json'; write_json(config, self.config(endpoint))
        run(corpus, self.root/'run', config, limit=1)
        judge(corpus, self.root/'run', self.root/'judge', config)
        judge(corpus, self.root/'run', self.root/'judge', config)
        self.assertEqual(len(received), 2)
        request = received[1]
        self.assertFalse(request['chat_template_kwargs']['enable_thinking'])
        data = json.loads(request['messages'][-1]['content'])
        self.assertEqual(data['candidate_answer'], 'Pick up the cup.')
        self.assertEqual(data['reference_answer'], 'Lift the cup.')
        self.assertEqual(data['image_count'], 1)
        identity = read_json(self.root/'judge/run.json')['identity']
        self.assertEqual(identity['judge_protocol_version'], 'semantic_judge_v3_binary_sequence_alignment')
        self.assertEqual(data['task_rubric'], identity['task_rubrics']['action_sequence_description'])
        self.assertIn('one-to-many and many-to-one', data['task_rubric'])
        self.assertNotIn('test-v1', json.dumps(request['messages']))
        report = score(corpus, self.root/'run', self.root/'scores', self.root/'judge')
        self.assertEqual(report['open_micro']['accuracy_lower'], .5)
        self.assertEqual(report['open_statuses'], {'scored': 1, 'pending_prediction': 1})
        self.assertIsNone(report['official_all_accuracy'])
        run(corpus, self.root/'run', config)
        with self.assertRaises(ValueError):
            judge(corpus, self.root/'run', self.root/'judge', config)
        with self.assertRaises(ValueError):
            score(corpus, self.root/'run', self.root/'scores', self.root/'judge')

    def test_judge_task_selection_is_isolated_and_frozen(self):
        corpus = self.corpus([self.record('action_sequence_description', 'Lift the cup.'),
                              self.record('adjacent_action', 'Lift the cup.')])
        endpoint, received = self.server([(200, self.response('Lift the cup.')),
                                         (200, self.response('Lift the cup.')),
                                         (200, self.response(json.dumps(self.decision())))])
        config = self.root/'config.json'; write_json(config, self.config(endpoint))
        run(corpus, self.root/'run', config)
        progress = judge(corpus, self.root/'run', self.root/'judge', config,
                         tasks=['action_sequence_description'])
        self.assertEqual(progress['total'], 1)
        self.assertEqual(len(received), 3)
        identity = read_json(self.root/'judge/run.json')['identity']
        self.assertEqual(identity['eligible_ids'], ['q000001'])
        self.assertEqual(identity['selected_tasks'], ['action_sequence_description'])
        score(corpus, self.root/'run', self.root/'scores', self.root/'judge')
        rows = read_jsonl(self.root/'scores/scores.jsonl')
        self.assertEqual(rows[0]['score'], 1)
        self.assertEqual(rows[1]['status'], 'pending_judge')
        with self.assertRaises(ValueError):
            judge(corpus, self.root/'run', self.root/'judge', config, tasks=['adjacent_action'])
        for tasks in ([], ['unknown'], ['relative_distance']):
            with self.subTest(tasks=tasks), self.assertRaises(ValueError):
                judge(corpus, self.root/'run', self.root/'other', config, tasks=tasks)
        self.assertEqual(len(received), 3)

    def test_judge_protocol_error_persisted_without_resampling(self):
        r = self.record('action_sequence_description', 'Lift the cup.')
        corpus = self.corpus([r])
        endpoint, received = self.server([(200, self.response('Lift the cup.')),
                                         (200, self.response(json.dumps(self.decision(incorrect_claims=['wrong cup']))))])
        config = self.root/'config.json'; write_json(config, self.config(endpoint))
        run(corpus, self.root/'run', config)
        progress = judge(corpus, self.root/'run', self.root/'judge', config)
        self.assertEqual(progress['statuses'], {'judge_protocol_error': 1})
        judge(corpus, self.root/'run', self.root/'judge', config, retry_pending=True, retry_unknown=True)
        self.assertEqual(len(received), 2)
        report = score(corpus, self.root/'run', self.root/'scores', self.root/'judge')
        self.assertEqual(report['open_statuses'], {'judge_protocol_error': 1})
        self.assertEqual(report['overall']['pending'], 1)
        self.assertEqual(len(read_jsonl(self.root/'scores/judge_errors.jsonl')), 1)

    def test_judge_truncation_and_refusal_are_pending(self):
        from embodied_eval.scoring import validate_judge_response
        item = dict(content=[dict(type='image')])
        for extra in [dict(finish_reason='length'), dict(refusal='refused')]:
            response = dict(answer=json.dumps(self.decision()), finish_reason='stop')
            response.update(extra)
            with self.assertRaises(ValueError):
                validate_judge_response(item, response)

    def test_empty_candidate_scores_zero_without_judge_call(self):
        corpus = self.corpus([self.record('action_sequence_description', 'Lift the cup.')])
        endpoint, received = self.server([(200, self.response(''))])
        config = self.root/'config.json'; write_json(config, self.config(endpoint))
        run(corpus, self.root/'run', config)
        progress = judge(corpus, self.root/'run', self.root/'judge', config)
        self.assertEqual(progress['total'], 0)
        self.assertEqual(len(received), 1)
        report = score(corpus, self.root/'run', self.root/'scores', self.root/'judge')
        self.assertEqual(report['open_micro']['accuracy'], 0)

    def test_facts_validate_before_serving(self):
        item = dict(id='q000001', answer_format='open', task='adjacent_action', content=[dict(type='image')])
        fact = dict(required_events=['lift cup'], required_relations=[], optional_details=[],
                    admissible_groupings=[], evidence_refs=[dict(image_index=0, observation='cup lifted')],
                    anchor_instance='first reach', direction='after')
        validate_facts({'q000001': fact}, [item])
        for changes in [dict(direction='earlier'), dict(anchor_instance=''),
                        dict(required_events=[]), dict(evidence_refs=[dict(image_index=1, observation='x')])]:
            with self.assertRaises(ValueError):
                validate_facts({'q000001': dict(fact, **changes)}, [item])
        with self.assertRaises(ValueError):
            validate_facts({'unknown': fact}, [item])

    def test_wait_then_judge_entry_point(self):
        corpus = self.corpus([self.record('action_sequence_description', 'Lift the cup.')])
        endpoint, received = self.server([(200, self.response('Lift the cup.')),
                                         (200, self.response(json.dumps(self.decision())))])
        config = self.root/'config.json'; write_json(config, self.config(endpoint))
        run(corpus, self.root/'run', config)
        subprocess.run([sys.executable, str(ROOT/'scripts/judge_and_score.py'),
                        '--corpus', str(corpus), '--run-dir', str(self.root/'run'),
                        '--config', str(config), '--output', str(self.root/'judge'),
                        '--scores-output', str(self.root/'scores'), '--wait-for-run'],
                       check=True, capture_output=True, timeout=30)
        self.assertEqual(len(received), 2)
        self.assertEqual(read_json(self.root/'judge/controller.json')['status'], 'completed')
        self.assertEqual(read_json(self.root/'scores/report.json')['open_micro']['accuracy'], 1)


if __name__ == '__main__':
    unittest.main()
