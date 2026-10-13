"""Pinned HY MoT Transformers backend for the benchmark's OpenAI HTTP contract."""
import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import secrets
import threading
import time
import uuid

from PIL import Image


def convert_messages(messages):
    if not isinstance(messages, list) or not messages:
        raise ValueError('messages must be a nonempty list')
    converted, images = [], []
    for message in messages:
        if not isinstance(message, dict) or message.get('role') not in {'system', 'user', 'assistant'}:
            raise ValueError('unsupported message role')
        content = message.get('content')
        if isinstance(content, str):
            converted.append(dict(role=message['role'], content=content))
            continue
        if not isinstance(content, list):
            raise ValueError('message content must be text or a list of blocks')
        blocks = []
        for block in content:
            if not isinstance(block, dict):
                raise ValueError('content blocks must be objects')
            if block.get('type') == 'text':
                if not isinstance(block.get('text'), str):
                    raise ValueError('text content must be a string')
                blocks.append(dict(type='text', text=block['text']))
            elif block.get('type') == 'image_url':
                image_url = block.get('image_url')
                url = image_url.get('url') if isinstance(image_url, dict) else None
                if not isinstance(url, str) or ',' not in url:
                    raise ValueError('image_url must contain a base64 data URL')
                header, encoded = url.split(',', 1)
                if not header.startswith('data:image/') or not header.endswith(';base64'):
                    raise ValueError('benchmark images must use base64 data URLs')
                raw = base64.b64decode(encoded, validate=True)
                with Image.open(io.BytesIO(raw)) as image:
                    image.load()
                    image = image.copy()
                images.append(dict(sha256=hashlib.sha256(raw).hexdigest(), size=list(image.size)))
                blocks.append(dict(type='image', image=image))
            else:
                raise ValueError('unsupported content block')
        converted.append(dict(role=message['role'], content=blocks))
    if len(images) > 32:
        raise ValueError('more than 32 images; never silently drop input')
    return converted, images


def validate_request(body, model_id):
    allowed = {'model', 'messages', 'temperature', 'top_p', 'max_tokens', 'seed',
               'stream', 'chat_template_kwargs'}
    if set(body) - allowed:
        raise ValueError('unsupported request fields')
    if body.get('model') != model_id or body.get('stream', False):
        raise ValueError('wrong model or streaming unsupported')
    if body.get('temperature', 0) != 0 or body.get('top_p', 1) != 1:
        raise ValueError('only the frozen greedy configuration is supported')
    if body.get('seed', 42) != 42:
        raise ValueError('unexpected generation seed')
    if body.get('chat_template_kwargs') != {'enable_thinking': False}:
        raise ValueError('explicit non-thinking template required')
    if type(body.get('max_tokens')) is not int or body['max_tokens'] != 1024:
        raise ValueError('frozen output budget is 1024')


def prepare_inputs(processor, messages):
    # The pinned generic apply_chat_template(tokenize=True) assumes every
    # content is a list, but HY's native template requires a string system
    # message. Render that template unchanged, then use the native processor.
    prompt = processor.apply_chat_template(messages, tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    images = [block['image'] for message in messages if isinstance(message['content'], list)
              for block in message['content'] if block['type'] == 'image']
    return processor(text=[prompt], images=images or None, return_tensors='pt',
                     add_special_tokens=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--weights', required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--runtime-dir', type=Path, required=True)
    p.add_argument('--api-key-env')
    args = p.parse_args()
    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    if args.api_key_env and not api_key:
        p.error('missing API key environment variable: ' + args.api_key_env)
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    import uvicorn

    torch.set_num_threads(8)
    runtime = args.runtime_dir
    runtime.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(args.weights, trust_remote_code=False, local_files_only=True)
    processor.chat_template = (Path(args.weights) / 'chat_template.jinja').read_text(encoding='utf-8')
    model, loading = AutoModelForImageTextToText.from_pretrained(
        args.weights, torch_dtype=torch.bfloat16, device_map='cuda',
        trust_remote_code=False, local_files_only=True, output_loading_info=True)
    (runtime / 'model_loading.json').write_text(json.dumps(loading, indent=2, default=str)+'\n')
    if any(loading.get(k) for k in ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs']):
        raise ValueError('checkpoint loading mismatch; see model_loading.json')
    model.eval()
    app = create_app(model, processor, args.model, runtime, api_key=api_key)
    uvicorn.run(app, host=args.host, port=args.port, workers=1, log_level='info')


def create_app(model, processor, model_id, runtime, api_key=None):
    import torch
    import transformers
    from fastapi import FastAPI, Header, HTTPException

    runtime = Path(runtime)
    runtime.mkdir(parents=True, exist_ok=True)
    context = int(model.config.text_config.max_position_embeddings)
    eos = model.generation_config.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos)
    (runtime / 'backend.json').write_text(json.dumps(dict(
        transformers=transformers.__version__, torch=torch.__version__, context=context,
        enable_thinking=False, max_new_tokens=1024, eos_token_ids=sorted(eos_ids),
        image_processor=processor.image_processor.to_dict()), default=str, indent=2)+'\n')
    lock = threading.Lock()
    app = FastAPI()

    @app.get('/health')
    def health():
        return {'status': 'ok'}

    @app.get('/v1/models')
    def models():
        return dict(object='list', data=[dict(id=model_id, object='model', owned_by='local')])

    @app.post('/v1/chat/completions')
    def completion(body: dict, authorization: str = Header(default='')):
        if api_key and not secrets.compare_digest(authorization, 'Bearer ' + api_key):
            raise HTTPException(status_code=401, detail='invalid API key')
        request_id = uuid.uuid4().hex
        start = time.time()
        audit = dict(id=request_id, started_at=start)
        try:
            validate_request(body, model_id)
            messages, images = convert_messages(body.get('messages'))
            with lock, torch.inference_mode():
                torch.manual_seed(42)
                inputs = prepare_inputs(processor, messages)
                input_len = inputs['input_ids'].shape[1]
                grid = inputs.get('image_grid_thw')
                if images and (grid is None or grid.shape[0] != len(images)):
                    raise ValueError('processor did not preserve every image')
                image_tokens = int((inputs['input_ids'] == processor.image_token_id).sum())
                if grid is not None and image_tokens != int(grid.prod(dim=1).sum()) // processor.image_processor.merge_size ** 2:
                    raise ValueError('image token and visual patch counts differ')
                if input_len + body['max_tokens'] > context:
                    raise ValueError('native context budget exceeded; input retained without truncation')
                audit.update(images=images, input_tokens=input_len,
                             input_ids_sha256=hashlib.sha256(inputs['input_ids'].numpy().tobytes()).hexdigest(),
                             image_grid_thw=grid.tolist() if grid is not None else [],
                             image_tokens=image_tokens,
                             enable_thinking=False)
                output = model.generate(**inputs.to(model.device), do_sample=False,
                                        max_new_tokens=body['max_tokens'], use_cache=True)
                tokens = output[0, input_len:].tolist()
                raw = processor.decode(tokens, skip_special_tokens=False)
                answer = processor.decode(tokens, skip_special_tokens=True)
                finish = 'stop' if tokens and tokens[-1] in eos_ids else 'length'
                audit.update(completion_tokens=len(tokens), finish_reason=finish,
                             output_token_ids=tokens, raw_decode=raw)
            return dict(id='chatcmpl-'+request_id, object='chat.completion', created=int(start),
                        model=model_id, choices=[dict(index=0, finish_reason=finish,
                        message=dict(role='assistant', content=answer))],
                        usage=dict(prompt_tokens=input_len, completion_tokens=len(tokens),
                                   total_tokens=input_len+len(tokens)),
                        hy_native_output=raw, hy_input_images=len(images),
                        hy_input_tokens=input_len)
        except ValueError as exc:
            audit.update(error_type=type(exc).__name__, error=str(exc))
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            audit.update(error_type=type(exc).__name__, error=str(exc))
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=type(exc).__name__)
        finally:
            audit['finished_at'] = time.time()
            folder = runtime / 'request_audits'
            folder.mkdir(exist_ok=True)
            (folder / (request_id+'.json')).write_text(json.dumps(audit, ensure_ascii=False)+'\n', encoding='utf-8')

    return app


if __name__ == '__main__':
    main()
