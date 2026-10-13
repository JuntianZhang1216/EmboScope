"""HTTP adapter and durable, bounded-concurrency runner. No private answers loaded."""
import base64
import hashlib
import json
import os
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from urllib.parse import urlparse

import requests

from .common import (code_hashes, digest, encoded, file_hash, load_corpus, lock,
                     read_json, write_json, write_jsonl, atomic_bytes)


def validate_config(config):
    allowed = {'endpoint', 'endpoints', 'model', 'model_revision', 'api_key_env', 'auth_header', 'auth_prefix',
               'generation', 'request_extra', 'workers', 'requests_per_second', 'timeout_sec',
               'connect_timeout_sec', 'max_attempts', 'max_payload_bytes', 'trust_env_proxy',
               'reasoning_mode', 'notes', 'answer_adapter'}
    if set(config) - allowed:
        raise ValueError(f'unknown config fields: {sorted(set(config) - allowed)}')
    endpoints = config.get('endpoints', [config['endpoint']])
    if not isinstance(endpoints, list) or not endpoints or any(not isinstance(x, str) for x in endpoints):
        raise ValueError('endpoints must be a nonempty list of URLs')
    if len(set(endpoints)) != len(endpoints) or endpoints[0] != config['endpoint']:
        raise ValueError('endpoints must be unique with endpoint as the first entry')
    for endpoint in endpoints:
        url = urlparse(endpoint)
        if url.scheme not in ('http', 'https') or not url.netloc or url.username or url.password or url.query or url.fragment:
            raise ValueError('endpoint must be an HTTP URL without inline credentials/query/fragment')
    if not config.get('model'):
        raise ValueError('model is required')
    if 'model_revision' in config and (not isinstance(config['model_revision'], str)
                                     or not config['model_revision'].strip()):
        raise ValueError('model_revision must be a nonempty string when provided')
    if config.get('reasoning_mode') not in ('disabled', 'enabled', 'provider_default'):
        raise ValueError('declare reasoning_mode')
    from .answer_adapters import ADAPTERS
    if config.get('answer_adapter') is not None and config['answer_adapter'] not in ADAPTERS:
        raise ValueError('unknown answer adapter')
    generation = config.get('generation', {})
    allowed_gen = {'temperature', 'top_p', 'seed', 'max_tokens', 'max_completion_tokens', 'reasoning_effort'}
    if set(generation) - allowed_gen or not ({'max_tokens', 'max_completion_tokens'} & set(generation)):
        raise ValueError('unsupported generation fields or missing output budget')
    if 'max_tokens' in generation and 'max_completion_tokens' in generation:
        raise ValueError('use only one output budget field')
    # Backend extensions cannot replace the frozen messages or silently constrain answers.
    if set(config.get('request_extra', {})) - {'chat_template_kwargs'}:
        raise ValueError('unsupported request_extra; add a reviewed adapter for other extensions')
    for name, default in [('workers', 4), ('max_attempts', 3), ('max_payload_bytes', 100_000_000)]:
        value = config.get(name, default)
        if type(value) is not int or value < 1:
            raise ValueError(f'invalid {name}')
    for name in ('requests_per_second', 'timeout_sec', 'connect_timeout_sec'):
        if name in config and config[name] <= 0:
            raise ValueError(f'invalid {name}')
    return config


def visual_content(item):
    content = []
    for block in item['content']:
        if block['type'] == 'text':
            content.append(dict(type='text', text=block['text']))
        elif block['type'] == 'image':
            data = (getattr(item, 'media_root', Path.cwd()) / block['path']).read_bytes()
            if hashlib.sha256(data).hexdigest() != block['sha256']:
                raise ValueError('frozen image changed')
            content.append(dict(type='image_url', image_url=dict(
                url='data:' + block['mime'] + ';base64,' + base64.b64encode(data).decode())))
        else:
            raise ValueError('unknown public input block')
    return content


def solve_messages(item):
    return [dict(role='system', content=item['system']),
            dict(role='user', content=visual_content(item) + [dict(type='text', text=item['output_instruction'])])]


class Client:
    def __init__(self, config):
        self.config = validate_config(config)
        self.abort = threading.Event()
        self.rate_lock = threading.Lock()
        self.next_start = 0.0
        self.endpoints = config.get('endpoints', [config['endpoint']])
        self.endpoint_lock = threading.Lock()
        self.endpoint_cursor = 0
        self.failure_lock = threading.Lock()
        self.consecutive_failures = 0
        self.headers = {'Content-Type': 'application/json'}
        if config.get('api_key_env'):
            secret = os.environ.get(config['api_key_env'])
            if not secret:
                raise ValueError('missing API key environment variable: ' + config['api_key_env'])
            self.headers[config.get('auth_header', 'Authorization')] = config.get('auth_prefix', 'Bearer ') + secret

    def call(self, messages, path, identity, response_validator=None):
        cfg = self.config
        payload = dict(model=cfg['model'], messages=messages, stream=False,
                       **cfg['generation'], **cfg.get('request_extra', {}))
        data = encoded(payload)
        result = dict(**identity, status='in_flight', attempts=[], request_sha256=hashlib.sha256(data).hexdigest(),
                      payload_bytes=len(data), requested_model=cfg['model'], started_at=time.time())
        if len(data) > cfg.get('max_payload_bytes', 100_000_000):
            result.update(status='input_error', error='payload exceeds configured budget; no truncation applied')
            write_json(path, result)
            return result
        write_json(path, result)
        with requests.Session() as session:
            session.trust_env = cfg.get('trust_env_proxy', False)
            for attempt in range(cfg.get('max_attempts', 3)):
                if self.abort.is_set():
                    result.update(status='not_sent', error='global configuration/service abort')
                    break
                with self.rate_lock:
                    slot = max(time.monotonic(), self.next_start)
                    self.next_start = slot + 1 / cfg.get('requests_per_second', 2)
                if self.abort.wait(max(0, slot - time.monotonic())):
                    result.update(status='not_sent', error='global configuration/service abort')
                    break
                began = time.monotonic()
                with self.endpoint_lock:
                    endpoint = self.endpoints[self.endpoint_cursor % len(self.endpoints)]
                    self.endpoint_cursor += 1
                result['requested_endpoint'] = endpoint
                try:
                    response = session.post(endpoint, data=data, headers=self.headers,
                                            timeout=(cfg.get('connect_timeout_sec', 15), cfg.get('timeout_sec', 300)),
                                            allow_redirects=False)
                except requests.RequestException as exc:
                    # A timeout may already have generated an answer. Never silently sample again.
                    result['attempts'].append(dict(endpoint=endpoint, error=type(exc).__name__, elapsed_sec=time.monotonic() - began))
                    result.update(status='transport_unknown', error=type(exc).__name__)
                    break
                result['attempts'].append(dict(endpoint=endpoint, http_status=response.status_code, elapsed_sec=time.monotonic() - began))
                if response.status_code != 200:
                    result.update(status='http_error', error=f'HTTP {response.status_code}',
                                  error_body=response.text[:4000])
                    if response.status_code in (400, 401, 403, 404, 413, 422):
                        self.abort.set()
                    write_json(path, result)
                    if response.status_code in (429, 500, 502, 503, 504) and attempt + 1 < cfg.get('max_attempts', 3):
                        try:
                            delay = float(response.headers.get('Retry-After', 2 ** attempt))
                        except ValueError:
                            delay = 2 ** attempt
                        if self.abort.wait(min(60, max(0, delay))):
                            break
                        continue
                    break
                try:
                    raw = response.json()
                    result['raw_response'] = raw
                    if not isinstance(raw, dict):
                        raise ValueError('expected response object')
                    choices = raw.get('choices')
                    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                        raise ValueError('expected nonempty choices list of objects')
                    choice = raw['choices'][0]
                    result.update(returned_model=raw.get('model'), usage=raw.get('usage', {}),
                                  finish_reason=choice.get('finish_reason'))
                    message = choice.get('message')
                    if not isinstance(message, dict):
                        raise ValueError('expected message object')
                    answer = message.get('content')
                    if answer is None and ('content' in message or message.get('refusal') or message.get('tool_calls') or choice.get('finish_reason') in ('length', 'content_filter')):
                        answer = ''
                    if not isinstance(answer, str):
                        raise ValueError('expected string final answer')
                    result.update(status='success', answer=answer, finish_reason=choice.get('finish_reason'),
                                  refusal=message.get('refusal'), returned_model=raw.get('model'), usage=raw.get('usage', {}),
                                  has_reasoning=bool(message.get('reasoning_content') or message.get('reasoning')))
                    if cfg['reasoning_mode'] == 'disabled' and (result['has_reasoning'] or '<think>' in answer):
                        result.update(status='protocol_error', error='backend emitted thinking despite disabled configuration')
                        self.abort.set()
                    if cfg.get('answer_adapter'):
                        from .answer_adapters import adapt_answer
                        result.update(raw_answer=answer, answer_adapter=cfg['answer_adapter'],
                                      answer=adapt_answer(answer, cfg['answer_adapter']))
                except (ValueError, KeyError, IndexError, TypeError) as exc:
                    result.update(status='protocol_error', error=type(exc).__name__, error_detail=str(exc))
                    if 'raw_response' not in result:
                        result['response_text'] = response.text[:4000]
                if result['status'] == 'success' and response_validator is not None:
                    try:
                        result['judgment'] = response_validator(result)
                    except (ValueError, TypeError) as exc:
                        result.update(status='judge_protocol_error', error=str(exc))
                break
        with self.failure_lock:
            self.consecutive_failures = 0 if result['status'] == 'success' else self.consecutive_failures + 1
            if self.consecutive_failures >= 3:
                self.abort.set()
        result['finished_at'] = time.time()
        write_json(path, result)
        return result


def execute(items, output, config, identity, message_builder, retry_pending=False, retry_unknown=False, limit=None, canary_per_task=None, response_validator=None):
    """Sequential waves retain bounded memory even for multi-image payloads."""
    output = Path(output)
    with lock(output):
        manifest_path = output / 'run.json'
        contract = digest(identity)
        if manifest_path.exists():
            if read_json(manifest_path)['contract'] != contract:
                raise ValueError('run configuration/input/code changed; choose a new run directory')
        else:
            for source in Path(__file__).parent.glob('*.py'):
                atomic_bytes(output / 'code_snapshot' / source.name, source.read_bytes())
            write_json(manifest_path, dict(contract=contract, identity=identity, created_at=time.time()))
        client = Client(config)
        todo, all_results = [], {}
        safe_retry = {'input_error', 'http_error', 'not_sent'}
        unknown_retry = {'in_flight', 'transport_unknown', 'protocol_error'}
        for item in items:
            path = output / 'responses' / (item['id'] + '.json')
            if path.exists():
                old = read_json(path)
                if old['contract'] != contract or old['input_sha256'] != item['input_sha256']:
                    raise ValueError('cached response identity mismatch')
                all_results[item['id']] = old
                eligible = retry_pending and old['status'] in safe_retry or retry_unknown and old['status'] in unknown_retry
                if not eligible:
                    continue
            todo.append(item)
        if canary_per_task is not None:
            chosen, counts = [], Counter()
            for item in todo:
                if counts[item['task']] < canary_per_task:
                    chosen.append(item)
                    counts[item['task']] += 1
            todo = chosen
        if limit is not None:
            todo = todo[:limit]

        def process(item):
            path = output / 'responses' / (item['id'] + '.json')
            if path.exists():
                write_json(output / 'retry_history' / item['id'] / f'{time.time_ns()}.json', read_json(path))
            ident = dict(id=item['id'], task=item['task'], input_sha256=item['input_sha256'], contract=contract)
            try:
                messages = message_builder(item)
            except (OSError, ValueError) as exc:
                result = dict(**ident, status='input_error', error=type(exc).__name__ + ': ' + str(exc), attempts=[])
                write_json(path, result)
                return result
            validator = (lambda result: response_validator(item, result)) if response_validator else None
            return client.call(messages, path, ident, validator)

        def progress():
            value = dict(total=len(items), saved=len(all_results), statuses=dict(Counter(r['status'] for r in all_results.values())),
                         remaining=len(items) - len(all_results), aborted=client.abort.is_set(), updated_at=time.time())
            write_json(output / 'progress.json', value)
            return value

        workers = config.get('workers', 4)
        queue = iter(todo)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {}
            def submit_one():
                if client.abort.is_set():
                    return
                item = next(queue, None)
                if item:
                    pending[pool.submit(process, item)] = item['id']
            for _ in range(workers):
                submit_one()
            while pending:
                done, _ = wait(pending, timeout=15, return_when=FIRST_COMPLETED)
                for future in done:
                    sid = pending.pop(future)
                    all_results[sid] = future.result()
                    submit_one()
                progress()
                if done:
                    print(f"saved={len(all_results)}/{len(items)} success={sum(r['status']=='success' for r in all_results.values())}", flush=True)
        write_jsonl(output / 'predictions.jsonl', [all_results[i['id']] for i in items if i['id'] in all_results])
        return progress()


def run(corpus, output, config_path, retry_pending=False, retry_unknown=False, limit=None, canary_per_task=None):
    manifest, items = load_corpus(corpus)
    config = validate_config(read_json(config_path))
    identity = dict(stage='solve', corpus_manifest_sha256=file_hash(Path(corpus) / 'manifest.json'),
                    corpus=Path(os.path.relpath(Path(corpus).resolve(), Path(output).resolve())).as_posix(),
                    config=config, code=code_hashes())
    return execute(items, output, config, identity, solve_messages, retry_pending, retry_unknown, limit, canary_per_task)


def probe(config_path, output=None):
    config = validate_config(read_json(config_path))
    client = Client(config)
    result = dict(endpoint=config['endpoint'], requested_model=config['model'], checked_at=time.time(), ready=False)
    replicas = []
    with requests.Session() as session:
        session.trust_env = config.get('trust_env_proxy', False)
        for endpoint in client.endpoints:
            row = dict(endpoint=endpoint, ready=False)
            try:
                base = endpoint.removesuffix('/chat/completions')
                response = session.get(base + '/models', headers=client.headers, timeout=(10, 15), allow_redirects=False)
                row['http_status'] = response.status_code
                if response.status_code == 200:
                    models = [entry['id'] for entry in response.json()['data']]
                    row.update(available_models=models, ready=config['model'] in models)
            except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
                row['error'] = type(exc).__name__
            replicas.append(row)
    if len(replicas) == 1:
        result.update(replicas[0])
    result.update(replicas=replicas, ready=all(row['ready'] for row in replicas))
    if output:
        write_json(output, result)
    return result
