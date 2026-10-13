#!/usr/bin/env bash
# Run the EmboScope evaluation stages using this checkout's Python package.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
python_bin="${EMBOSCOPE_PYTHON:-python3}"
dataset="$repo_root/data/benchmark_data.jsonl"
output=""
model_config=""
judge_config=""
prompts_path="$repo_root/eval/embodied_eval/prompts.json"
stage=all
per_task=""
skip_judge=0
retry_args=()

usage() {
    cat <<'EOF'
Usage: bash evaluate.sh --output DIR [options]

  --model-config FILE   Model serving configuration (required for all/run)
  --judge-config FILE   Judge serving configuration (required for all/judge,
                        unless --skip-judge is used)
  --prompts-path FILE   Prompt bundle (use the profile-specific bundle for InternVL Thinking)
  --output DIR          Workflow directory: corpus/, model/, judge/, scores/, logs/
  --stage STAGE         all (default), prepare, run, judge, or score
  --per-task N          Prepare N questions per task; omit for the full dataset
  --skip-judge          Score closed questions only (all/score stages)
  --retry-pending       Retry known input/HTTP failures through the Python runner
  --retry-unknown       Explicitly retry requests with unknown delivery status
  --python EXECUTABLE   Python 3.10+ interpreter (or set EMBOSCOPE_PYTHON)
  -h, --help            Show this help

Install dependencies with: python -m pip install .
Download benchmark_data.jsonl and images/ into this checkout's data/ directory.
The dataset location is fixed; it does not depend on the working directory.
Paths passed as arguments are relative to your current working directory.
Repeat the same command to resume. Keep dataset, selection and configurations fixed.
Exit codes: 0 = requested stage complete, 3 = incomplete results; other nonzero values indicate errors.
EOF
}

fail() { printf 'Error: %s\n' "$*" >&2; exit 2; }
value_required() { [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || fail "$1 requires a value"; }
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-config|--judge-config|--prompts-path|--output|--stage|--per-task|--python)
            value_required "$@"
            case "$1" in
                --model-config) model_config="$2" ;;
                --judge-config) judge_config="$2" ;;
                --prompts-path) prompts_path="$2" ;;
                --output) output="$2" ;;
                --stage) stage="$2" ;;
                --per-task) per_task="$2" ;;
                --python) python_bin="$2" ;;
            esac
            shift 2 ;;
        --skip-judge) skip_judge=1; shift ;;
        --retry-pending|--retry-unknown) retry_args+=("$1"); shift ;;
        -h|--help) usage; exit 0 ;;
        *) fail "unknown argument: $1" ;;
    esac
done
[[ -n "$output" ]] || fail '--output is required'
case "$stage" in all|prepare|run|judge|score) ;; *) fail "unknown stage: $stage" ;; esac
[[ -z "$per_task" || "$per_task" =~ ^[1-9][0-9]*$ ]] || fail '--per-task must be positive'
if [[ "$skip_judge" == 1 && "$stage" != all && "$stage" != score ]]; then
    fail '--skip-judge applies to all/score stages'
fi
if [[ "$stage" == all || "$stage" == run ]]; then
    [[ -f "$model_config" ]] || fail '--model-config must name an existing file'
fi
if [[ "$stage" == judge || ( "$stage" == all && "$skip_judge" == 0 ) ]]; then
    [[ -f "$judge_config" ]] || fail '--judge-config must name an existing file, or use --skip-judge'
fi
[[ -f "$dataset" ]] || fail "missing data/benchmark_data.jsonl; download the question JSONL into $repo_root/data/"
[[ -d "$repo_root/data/images" ]] || fail "missing data/images/; extract the complete media archive into $repo_root/data/"
[[ -f "$prompts_path" ]] || fail "prompts not found: $prompts_path"
command -v "$python_bin" >/dev/null || fail "Python executable not found: $python_bin"
export PYTHONPATH="$repo_root/eval${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
corpus="$output/corpus"
model_run="$output/model"
judge_run="$output/judge"
scores="$output/scores"

# Validate both serving configurations before inference starts.
"$python_bin" - "$stage" "$skip_judge" "$model_config" "$judge_config" <<'PY'
import sys
from embodied_eval.common import read_json
from embodied_eval.serving import Client
stage, skip, model, judge = sys.argv[1:]
if stage in ('all', 'run'):
    Client(read_json(model))
if stage == 'judge' or stage == 'all' and skip == '0':
    Client(read_json(judge))
PY

mkdir -p "$output/logs"
cli() {
    local command="$1"
    printf '\n[%s] %s\n' "$(date -u +%FT%TZ)" "$command" | tee -a "$output/logs/$command.log"
    "$python_bin" -m embodied_eval "$@" 2>&1 | tee -a "$output/logs/$command.log"
}

if [[ ! -f "$corpus/manifest.json" ]]; then
    [[ "$stage" == all || "$stage" == prepare ]] || fail 'prepare the corpus first with --stage prepare'
    prepare_args=()
    if [[ -n "$per_task" ]]; then prepare_args+=(--per-task "$per_task"); fi
    cli prepare --prompts-path "$prompts_path" --output "$corpus" ${prepare_args[@]+"${prepare_args[@]}"}
fi

# Check the prepared selection before reusing it on a resumed workflow.
"$python_bin" - "$dataset" "$corpus" "$per_task" "$prompts_path" <<'PY'
from pathlib import Path
import sys
import embodied_eval
from embodied_eval.common import code_hashes, file_hash, load_corpus, read_json

dataset, corpus = map(lambda p: Path(p).resolve(), sys.argv[1:3])
per_task = int(sys.argv[3]) if sys.argv[3] else None
manifest, _ = load_corpus(corpus, private=True)
resources = Path(embodied_eval.__file__).parent
checks = {
    'dataset': manifest['dataset_sha256'] == file_hash(dataset),
    'media root': (corpus / manifest['media_root']).resolve() == dataset.parent,
    'selection': manifest['selection'] == dict(tasks=None, per_task=per_task),
    'catalog': manifest['catalog_sha256'] == file_hash(resources / 'task_catalog.json'),
    'prompts': read_json(corpus / 'prompts.json') == read_json(sys.argv[4]),
    'compiler': manifest['compiler'] == code_hashes(),
}
changed = [name for name, valid in checks.items() if not valid]
if changed:
    raise SystemExit('Prepared corpus mismatch (' + ', '.join(changed) + '); use a new --output directory.')
PY

score_run() {
    local judge_args=()
    if [[ "$skip_judge" == 0 && -f "$judge_run/run.json" ]]; then
        judge_args+=(--judge-dir "$judge_run")
    fi
    cli score --corpus "$corpus" --run-dir "$model_run" --output "$scores" ${judge_args[@]+"${judge_args[@]}"}
}

if [[ "$stage" == all || "$stage" == run ]]; then
    cli run --corpus "$corpus" --config-path "$model_config" --output "$model_run" ${retry_args[@]+"${retry_args[@]}"}
fi
if [[ "$stage" == all || "$stage" == run || "$stage" == judge ]]; then
    if "$python_bin" - "$corpus" "$model_run" <<'PY'
import sys
from embodied_eval.scoring import predictions_for
items, predictions = predictions_for(*sys.argv[1:])
pending = sum(predictions.get(item['id'], {}).get('status') != 'success' for item in items)
if pending:
    print(f'{pending} model responses are incomplete; judging has not started.', file=sys.stderr)
    raise SystemExit(3)
PY
    then
        :
    else
        status=$?
        if [[ "$status" == 3 ]]; then
            # A partial report excludes any judgments tied to a different prediction snapshot.
            cli score --corpus "$corpus" --run-dir "$model_run" --output "$scores"
            printf 'Inspect %s and resume with the appropriate retry flag.\n' "$model_run/progress.json" >&2
        fi
        exit "$status"
    fi
fi
if [[ "$stage" == judge || ( "$stage" == all && "$skip_judge" == 0 ) ]]; then
    cli judge --corpus "$corpus" --run-dir "$model_run" --config-path "$judge_config" --output "$judge_run" ${retry_args[@]+"${retry_args[@]}"}
fi
if [[ "$stage" == all || "$stage" == judge || "$stage" == score ]]; then
    score_run
    "$python_bin" - "$scores/report.json" "$skip_judge" <<'PY'
import sys
from embodied_eval.common import read_json
path, skip = sys.argv[1:]
report = read_json(path)
result = report['closed_micro'] if skip == '1' else report['overall']
print(('Closed-question report: ' if skip == '1' else 'Evaluation report: ') + path)
if result['pending']:
    print(f"{result['pending']} questions remain unscored; inspect the report and stage logs.", file=sys.stderr)
    raise SystemExit(3)
PY
fi
printf 'Completed stage: %s. Outputs: %s\n' "$stage" "$output"
