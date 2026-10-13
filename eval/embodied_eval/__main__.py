import argparse
import json
from pathlib import Path

from .inputs import prepare
from .scoring import judge, score
from .serving import run, probe
from .release import validate_release
from .layout import dataset_path, project_root


def main():
    resources = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='Frozen-input Embodied Benchmark serving client')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('validate', help='Check every released question, image and checksum; no API calls',
                       description='Validate the fixed data/ directory against this checkout\'s metadata/.')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--output')
    p = sub.add_parser('prepare', help='Validate all selected media, compile public inputs and private gold',
                       description='Prepare this checkout\'s data/benchmark_data.jsonl and data/images/.')
    p.add_argument('--catalog', default=str(resources/'task_catalog.json'))
    p.add_argument('--prompts-path', default=str(resources/'prompts.json'))
    p.add_argument('--output', required=True)
    p.add_argument('--tasks', nargs='+')
    p.add_argument('--per-task', type=int)
    p.add_argument('--workers', type=int, default=8)
    p = sub.add_parser('probe', help='Read-only serving model discovery')
    p.add_argument('--config-path', required=True)
    p.add_argument('--output')
    for name in ('run', 'judge'):
        p = sub.add_parser(name)
        p.add_argument('--corpus', required=True)
        p.add_argument('--output', required=True)
        p.add_argument('--config-path', required=True)
        p.add_argument('--canary-per-task', type=int, help='Send this many new questions per task; resume same run for full evaluation')
        p.add_argument('--limit', type=int, help='Limit new requests, preserving full corpus denominator and resume identity')
        p.add_argument('--retry-pending', action='store_true', help='Retry local input errors / explicit HTTP failures')
        p.add_argument('--retry-unknown', action='store_true', help='Explicitly resend unknown in-flight/timeout outcomes; archived separately')
        if name == 'judge':
            p.add_argument('--run-dir', required=True)
            p.add_argument('--tasks', nargs='+', help='Judge only these open tasks; selection is frozen in the run contract')
            p.add_argument('--facts-path', help='Frozen per-item semantic contracts keyed by question id')
    p = sub.add_parser('score')
    p.add_argument('--corpus', required=True)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--judge-dir')
    args = vars(parser.parse_args())
    command = args.pop('command')
    if command in {'validate', 'prepare'}:
        try:
            root = project_root()
            args['dataset'] = dataset_path(root)
            if command == 'validate':
                args['metadata_dir'] = root / 'metadata'
        except ValueError as exc:
            parser.error(str(exc))
    for key in ('limit', 'per_task', 'workers', 'canary_per_task'):
        if key in args and args[key] is not None and args[key] < 1:
            parser.error(f'{key} must be positive')
    result = {'validate': validate_release, 'prepare': prepare, 'run': run, 'judge': judge, 'score': score, 'probe': probe}[command](**args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if command == 'probe' and not result['ready']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
