# Evaluation contract

## Unified Shell workflow

Place the downloaded JSONL at `data/benchmark_data.jsonl` and all media in `data/images/`. Install the evaluator, start the model and judge services, and obtain their JSON configurations as shown in the [README](../README.md#evaluate). `evaluate.sh` calls existing HTTP services; it does not deploy them. For the local deployment examples, run:

```bash
bash evaluate.sh --python .venv/bin/python \
  --model-config outputs/services/qwen35-4b/model.json \
  --judge-config outputs/services/judge/judge.json \
  --output outputs/evaluations/qwen35-4b
```

For existing API services, use the edited copies in `outputs/configs/` instead. The `*.example.json` files contain placeholder model IDs and are not ready to run unchanged. Both services must be available for the default `all` workflow; see [sequential evaluation](../deploy/README.md#sequential-evaluation--分阶段评测) to share GPUs between inference and judging.

The entry point calls this checkout's Python evaluator and works from other directories when invoked by its path. Configuration and output arguments refer to the caller's working directory; the dataset is fixed to the script's checkout. Missing downloads produce an actionable error before inference.

The installed Python CLI also uses the checkout's fixed `data/` directory; run it from the checkout or a subdirectory. Full validation uses the checksum manifests in the repository-root `metadata/`, so users only need to download the JSONL and media.

| Stage | Behavior |
| --- | --- |
| `all` (default) | Prepare or verify the corpus, run inference, judge open responses, and score |
| `prepare` | Prepare the corpus or verify that the existing selection matches |
| `run` | Run or resume inference on the prepared corpus |
| `judge` | Require complete inference, run or resume the judge, and write scores |
| `score` | Write scores from saved predictions and any saved judgments; no model requests |

When running stages separately, call `--stage prepare` before `--stage run`; only `all` and `prepare` create a missing corpus. The `judge` stage requires complete saved model predictions and also writes the scores.

The output directory contains `corpus/`, `model/`, `judge/`, `scores/` and per-stage logs in `logs/`. `--model-config` is required by `all`/`run`; `--judge-config` is required by `all`/`judge`. `all --skip-judge` requires only the model configuration. The script validates both configurations and authentication environment variables before starting an `all` run.

Resume with the same dataset, output directory, configurations and `--per-task` selection. A changed dataset, selection, prompt, catalog or compiler requires a new output directory. To test one question per task, add `--per-task 1` and use a separate output directory. Inference and judging retain the Python runner's configuration and response contracts.

`--prompts-path` selects a complete prompt bundle for preparation and resume validation. Use `configs/prompts/internvl35-14b-thinking.json` for the paper's InternVL Thinking profile, retaining this argument on all stages and resumes. It changes only the solver system prompt; the current judge protocol remains fixed.

Incomplete inference produces a partial score report and exits with code 3 before creating a judge run. This keeps the judge's prediction snapshot complete and stable. Incomplete scoring also exits with code 3. Retry flags are forwarded to the existing runner; they do not automatically resample successful or invalid semantic responses. All other command failures propagate a nonzero exit code.

`--skip-judge` explicitly requests closed-question scoring in `all` or `score`. Open responses remain unscored, and a successful closed-question run does not imply a complete overall benchmark score. Full runs report results at `scores/report.json` and `scores/report.md`.

## Public inputs

The compiler separates question inputs from reference answers. `task`, local paths, checksums, question IDs and GT are not sent as model hints. It preserves candidate labels, image/text interleaving, final-anchor candidate images and chronological ordering where the task specifies it. The model receives original image bytes as non-streaming, multi-image chat-completions requests.

`prepare` writes `items.jsonl`, `private/answers.jsonl`, `prompts.json` and `manifest.json`. Input hashes cover the serialized relative paths and metadata. Runtime path resolution is not serialized into items. A corpus refers to the dataset directory via a relative `media_root` in its manifest. A solve run refers to the corpus relative to the run directory.

## Closed scoring

- Multiple choice: strip outer whitespace and require exactly one valid uppercase option letter. Explanations, option text, lowercase letters and multiple letters receive 0 with a format failure.
- Yes/No: allow case differences, outer whitespace, an optional English `Answer:` prefix, and at most one terminal `.`, `!`, `。` or `！`. Other formats fail.
- All denominators are fixed to the prepared selection. Incomplete results have bounds and a null complete accuracy. Bounds are not confidence intervals.

Reports include per-task scores, closed micro and macro averages, L1/L2/L3 question-weighted scores, Yes/No recalls, completion and scoring coverage, format failures, and binary open scores and overall accuracy. The complete pool has 4,020 closed questions and 447 open questions.

## Open scoring

Use one fixed multimodal judge configuration across compared models. The versioned rubric is `semantic_judge_v3_binary_sequence_alignment` in `scoring.py`.

Action sequence description aligns the meaning of complete sequences before checking distinct action coverage, essential order, invented events and interaction correctness. Reasonable splitting or merging of consecutive actions is allowed. Adjacent action requires the immediately preceding/following action around the specified anchor.

The judge produces strict JSON with a `correct` or `incorrect` verdict and integer score 1 or 0. Every valid judgment is final under this protocol. Correct verdicts require visual evidence and empty diagnostic error arrays. Judge errors and missing responses are execution states, not answer choices; they remain unscored and appear in `judge_errors.jsonl`. Explicit empty candidate answers and refusals score 0.

Use the fixed reference and any frozen facts together with the visual observations to resolve semantic ambiguity. If the candidate still cannot be established as satisfying an essential requirement, return incorrect (0) and identify that requirement. Sparse frames alone do not prove a missing action or hallucination; reasonable action splitting/merging and support from observed state transitions remain allowed. Do not invent visual evidence.

Reports use `open_micro`, `overall` and `overall_macro_accuracy`. `status` is `complete` when every selected question has a score, otherwise `incomplete`. `official_all_accuracy` is populated once the full dataset has been scored; it is null only for incomplete or subset evaluations.

The corpus prompts and judge run must declare `semantic_judge_v3_binary_sequence_alignment`. Evaluation rejects missing or mismatched protocol declarations.

## Configuration and retries

Configure the model identifier, reasoning mode, generation parameters, output budget and any named response adapter. `model_revision` is optional for existing services; if supplied, it is recorded and included in resume checks. Choose `disabled`, `enabled` or `provider_default` according to the actual backend. Temperature zero is not a guarantee of bitwise deterministic generation. Different token budgets must be reported as different conditions.

The generic templates use `provider_default` and no backend-specific chat-template extension. For backends supporting it, add `"request_extra": {"chat_template_kwargs": {"enable_thinking": false}}` and declare `"reasoning_mode": "disabled"` to explicitly disable thinking. Configure a corresponding supported setting for enabled thinking. The 24 paper profiles in `configs/models/` already declare their model-specific generation and serialization settings. The [deployment launcher](../deploy/README.md) generates a compatible config with the endpoint and an automatic deployment identifier in `model_revision`, together with a deployment fingerprint. No manual weight version is needed.

Successful responses, including wrong answers, format errors and refusals, are not resampled. Bounded retries apply to explicit transport/service failures. `--retry-pending` retries known input/HTTP failures; `--retry-unknown` explicitly resends requests whose delivery status is unknown and archives their previous records. A changed input, configuration or evaluator requires a new run directory.

Resume by repeating the same `embodied-eval run` or `embodied-eval judge` command with the same corpus, configuration and output directory. Code snapshots and hashes record the implementation used for each run. Moving the data, corpus and run together while retaining their relative layout preserves the run identity.

## Verification

Run unit tests without any credentials:

```bash
python -m pip install '.[test]'
python -m unittest discover -s eval/tests -v
```

Full dataset validation is separate and requires the media archive:

```bash
embodied-eval validate --output artifacts/data_validation.json
```

Local mock tests verify the request/scoring pipeline; their outputs are not model benchmark results.

To exercise one actual dataset input per task and the judge for both open tasks locally:

```bash
python eval/scripts/offline_smoke.py --corpus artifacts/corpus --output artifacts/offline_smoke
```

Use a new output directory on each invocation. The mock judge exercises both binary outcomes and makes no claim about actual model answer correctness.
