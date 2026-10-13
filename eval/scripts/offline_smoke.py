"""Run one real input per task and both open-task judges against localhost mocks."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from embodied_eval.common import load_corpus, read_jsonl, write_json
from embodied_eval.serving import run
from embodied_eval.scoring import judge, score


def smoke(corpus, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('use a new smoke output directory')
    output.mkdir(parents=True)
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            is_judge = len(body['messages']) == 3
            media = body['messages'][1]['content']
            requests.append(dict(stage='judge' if is_judge else 'solve',
                                 images=sum(b['type']=='image_url' for b in media)))
            if is_judge:
                data = json.loads(body['messages'][2]['content'])
                correct = data['task_rubric'].startswith('Evaluate the meaning')
                answer = json.dumps(dict(verdict='correct' if correct else 'incorrect', score=int(correct),
                    reason='Local mock for interface validation only.', missing_required_events=[],
                    incorrect_claims=[] if correct else ['Mock mismatch.'], relation_errors=[],
                    evidence_refs=[dict(image_index=0, observation='Mock evidence for protocol validation.')]))
            else:
                instruction = media[-1]['text']
                answer = 'A' if 'uppercase option letter' in instruction else 'Yes' if 'Yes or No' in instruction else 'Mock action.'
            response = dict(model='LOCAL_MOCK_NOT_A_BENCHMARK_MODEL',
                            choices=[dict(message=dict(content=answer), finish_reason='stop')], usage={})
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    config = output/'mock.json'
    write_json(config, dict(endpoint=f'http://127.0.0.1:{server.server_port}/v1/chat/completions',
                            model='LOCAL_MOCK_NOT_A_BENCHMARK_MODEL', model_revision='mock-only',
                            reasoning_mode='disabled', generation=dict(temperature=0, max_tokens=128),
                            workers=2, requests_per_second=100, max_attempts=1))
    try:
        manifest, items = load_corpus(corpus)
        progress = run(corpus, output/'solve', config, canary_per_task=1)
        if progress['statuses'].get('success') != len(manifest['tasks']):
            raise ValueError('a task failed the mock solve')
        # No new requests on a zero-work resume of the same contract.
        run(corpus, output/'solve', config, limit=0)
        judge_progress = judge(corpus, output/'solve', output/'judge', config)
        open_tasks = {i['task'] for i in items if i['answer_format']=='open'}
        if judge_progress['statuses'].get('success') != len(open_tasks):
            raise ValueError('an open task failed the mock judge')
        report = score(corpus, output/'solve', output/'scores', output/'judge')
        if report['overall']['total'] != manifest['total']:
            raise ValueError('fixed denominator changed')
        solve_calls = [r for r in requests if r['stage']=='solve']
        judge_calls = [r for r in requests if r['stage']=='judge']
        if len(solve_calls) != len(manifest['tasks']) or len(judge_calls) != len(open_tasks):
            raise ValueError('unexpected repeated requests')
        decisions = [r['judgment'] for r in read_jsonl(output/'scores/scores.jsonl') if 'judgment' in r]
        if len(decisions) != len(open_tasks) or {d['score'] for d in decisions} != {0, 1}:
            raise ValueError('both binary judge outcomes must be exercised')
        result = dict(status='passed_local_mock_only', tasks=len(manifest['tasks']),
                      solve_requests=len(solve_calls), judge_requests=len(judge_calls),
                      solve_images=sum(r['images'] for r in solve_calls),
                      fixed_denominator=manifest['total'], real_model_requests=0,
                      binary_judgments_verified=True, correct_and_incorrect_exercised=True)
        write_json(output/'validation.json', result)
        return result
    finally:
        server.shutdown()
        server.server_close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus', required=True)
    p.add_argument('--output', required=True)
    print(smoke(**vars(p.parse_args())))
