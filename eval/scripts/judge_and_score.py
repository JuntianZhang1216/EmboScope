"""Durable semantic judge followed by scoring, optionally waiting for a solve run."""
import argparse
import fcntl
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from embodied_eval.common import write_json
from embodied_eval.scoring import judge, predictions_for, score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--scores-output', required=True)
    parser.add_argument('--facts-path')
    parser.add_argument('--wait-for-run', action='store_true',
                        help='Wait for all solve responses to succeed and its writer lock to be released')
    parser.add_argument('--wait-timeout-sec', type=float, default=86400)
    parser.add_argument('--retry-pending', action='store_true')
    parser.add_argument('--retry-unknown', action='store_true')
    args = parser.parse_args()
    if args.wait_timeout_sec <= 0:
        parser.error('wait-timeout-sec must be positive')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.controller.lock').open('a') as controller_lock:
        fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = dict(pid=os.getpid(), started_at=time.time(), status='waiting_for_predictions',
                     corpus=args.corpus, run_dir=args.run_dir, config=args.config)
        write_json(out / 'controller.json', state)
        try:
            if args.wait_for_run:
                deadline = time.monotonic() + args.wait_timeout_sec
                while True:
                    # Avoid reading thousands of responses while the solve writer is active.
                    with (Path(args.run_dir) / '.lock').open('a') as solve_lock:
                        try:
                            fcntl.flock(solve_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            busy = False
                        except BlockingIOError:
                            busy = True
                        if not busy:
                            items, predictions = predictions_for(args.corpus, args.run_dir)
                            if all(predictions.get(i['id'], {}).get('status') == 'success' for i in items):
                                break
                    if time.monotonic() >= deadline:
                        raise TimeoutError('waiting for completed solve run timed out')
                    state['updated_at'] = time.time()
                    write_json(out / 'controller.json', state)
                    time.sleep(10)
            state.update(status='judging', judge_started_at=time.time())
            write_json(out / 'controller.json', state)
            progress = judge(args.corpus, args.run_dir, out, args.config, args.facts_path,
                             retry_pending=args.retry_pending, retry_unknown=args.retry_unknown)
            report = score(args.corpus, args.run_dir, args.scores_output, out)
            state.update(status='completed' if report['open_micro']['pending'] == 0
                         else 'finished_with_pending', progress=progress,
                         open_micro=report['open_micro'],
                         open_statuses=report['open_statuses'])
        except BaseException as exc:
            state.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
            raise
        finally:
            state['finished_at'] = time.time()
            write_json(out / 'controller.json', state)


if __name__ == '__main__':
    main()
