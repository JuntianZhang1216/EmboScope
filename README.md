# EmboScope

[中文说明](README_zh.md)

This repository provides the **evaluation code and dataset accompanying the paper _EmboScope: A Multi-Source Comprehensive Benchmark for Embodied Perception, Understanding, and Planning_**. It includes input preparation, model inference, closed-answer scoring, and binary multimodal judging for open responses.

EmboScope contains **4,467 synthetically constructed visual question-answer examples** across **14 tasks** and three capability levels. This release includes every question, its reference answer, and the fixed visual observations used for evaluation.

| Level | Tasks | Questions |
| --- | ---: | ---: |
| L1: Perception | 4 | 1,241 |
| L2: Embodied semantics | 6 | 1,752 |
| L3: Planning and reasoning | 4 | 1,474 |

There are 3,520 multiple-choice questions, 500 Yes/No questions, and 447 open questions. Action Prediction and Selection (APS) evaluates action prediction and selection in L3.

## Benchmark overview

### Construction framework

[![EmboScope construction framework: interaction sources, shared schema, rule-based QA generation, and three capability levels](docs/figures/overview.png)](docs/figures/overview.pdf)

The paper's framework connects interaction observations to a shared schema of objects, actions, states, and temporal relations, then constructs questions under task-specific observation constraints. The worked example illustrates all three capability levels from one interaction; future observations are held out for action prediction. [View PDF](docs/figures/overview.pdf).

### Representative tasks

[![Examples of perception, embodied semantics, and planning and reasoning questions in EmboScope](docs/figures/task_examples.png)](docs/figures/task_examples.pdf)

One example from each level: relative distance, action between frames, and action prediction and selection. Green marks reference answers. Questions are abridged; panels (b) and (c) omit distractors. Panel (c) displays inputs 1, 3, 6, and 8, with frame 8 as the current observation. [View PDF](docs/figures/task_examples.pdf).

### Results reported in the paper

<p align="center">
  <a href="docs/figures/capability_profiles.pdf">
    <img src="docs/figures/capability_profiles.png" alt="Task-level capability profiles of five baselines across the three EmboScope levels" width="600">
  </a>
</p>

**Capability profiles.** Five representative baselines on the same 0–100% accuracy scale. Each axis represents a task; polygon areas are not aggregate scores and should not be compared across panels. [View PDF](docs/figures/capability_profiles.pdf).

<p align="center">
  <a href="docs/figures/task_accuracy.pdf">
    <img src="docs/figures/task_accuracy.png" alt="Accuracy distributions across evaluated models for all 14 EmboScope tasks" width="600">
  </a>
</p>

**Task accuracy distributions.** Dots show individual model accuracies, short bars indicate means, and violins summarize the distributions across evaluated models. [View PDF](docs/figures/task_accuracy.pdf).

<details>
<summary>Task abbreviations used in the figures</summary>

| Level | Code | Task |
| --- | --- | --- |
| L1 | RD | Relative distance |
| L1 | AM | Attribute matching |
| L1 | CVM | Cross-view matching |
| L1 | FC | Frame completion |
| L2 | AFM | Action–frame matching |
| L2 | ABF | Action between frames |
| L2 | SFA | Step–frame alignment |
| L2 | ASD | Action sequence description |
| L2 | AA | Adjacent action |
| L2 | SC | Step completion |
| L3 | APS | Action prediction and selection |
| L3 | PC | Progress comparison |
| L3 | PO | Progress ordering |
| L3 | PAD | Progress anomaly detection |

</details>

## Layout

```text
evaluate.sh                    # Unified prepare → run → judge → score workflow
data/                         # Fixed download destination; only .gitkeep is tracked
  benchmark_data.jsonl        # Download all 4,467 questions here
  images/                     # Extract all 42,637 image files here
metadata/
  release_manifest.json       # Version, counts and checksums
  images.jsonl                # Image dimensions, encoding, size and SHA-256
  video_frames.jsonl          # Fixed frame positions for 229 video questions
configs/                      # Model and judge configuration templates
  models/                     # 24 local-model evaluation profiles from the paper
  prompts/                    # InternVL Thinking prompt bundle
deploy/                       # Local servers, pinned dependencies and compatibility patches
eval/embodied_eval/            # Installable evaluation package
eval/scripts/                 # Packaging, local mock and judging utilities
eval/tests/                   # Local mock tests; no API credentials required
docs/                         # Dataset and evaluation contracts
  figures/                    # Paper figures for this README (PNG and PDF)
```

The question JSONL has **no `source` field**. Each `image_list[*].image` is a path such as `images/<sha256>.jpg`, relative to the directory containing the JSONL. All evaluation media is stored in the one `images/` directory. The 229 video questions are represented by their exact frozen observation frames, with no video decoding or resampling required. Duplicate image bytes share a file; observation order and repeated references are preserved.

## Install

Use Linux or macOS and Python 3.10+. The runner uses POSIX file locks.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
embodied-eval --help
```

For development, use `python -m pip install -e .`. The package includes its task catalog and prompts. Installing it does not copy the media into Python's site-packages.

## Data

**Dataset download links have not yet been published.** The commands below assume you have obtained the matching v0.2.1 dataset archive or its JSONL and media files.

The code checkout contains an empty `data/` placeholder. Download the question JSONL and all media into **`data/benchmark_data.jsonl` and `data/images/`**. Both evaluation entry points use this fixed location. The dataset payload is excluded from Git and the code archive; checksum metadata is already included in `metadata/` at the repository root.

From the repository root, extract the matching dataset archive into `data/`:

```bash
tar -xzf /path/to/embodied-benchmark-data-v0.2.1.tar.gz -C data
```

Do not add another `data/` directory inside it. If downloading the JSONL and media separately, place the JSONL directly in `data/` and extract the images into `data/images/`. README paper figures in `docs/figures/` are included in the code repository. See [packaging](docs/RELEASING.md).

Validate all records, decode all images and check all media hashes:

```bash
embodied-eval validate --output artifacts/data_validation.json
```

The Shell workflow below prepares its own corpus. For the [individual CLI commands](#individual-cli-commands), prepare a corpus explicitly:

```bash
embodied-eval prepare --output artifacts/corpus
```

For a subset used with individual CLI commands, prepare one question per task in a separate corpus:

```bash
embodied-eval prepare --per-task 1 --output artifacts/smoke_corpus
```

Run the installed `embodied-eval` CLI from the checkout or one of its subdirectories. The Shell entry point also works from any working directory when invoked by path; it always uses its own checkout's `data/`. Prepared corpora record relative media links, so moving the checkout together with `data/` and its evaluation outputs preserves their layout.

## Evaluate

`evaluate.sh` calls **running model services** over HTTP. It prepares inputs, requests predictions, judges open responses and writes scores; it does not download weights or start model servers. Full evaluation needs a model service and a multimodal judge service, each with its own JSON configuration. Both must support OpenAI-compatible chat completions with multiple images.

Run the following commands from the repository root, after completing [installation](#install) and [data preparation](#data). The lightweight evaluator uses `.venv`; local GPU servers use separate `.venvs/*` environments. An existing API service does not require a local GPU environment.

### Option A: Deploy a local model

[`deploy/`](deploy/README.md) provides **20 local models and 24 evaluation profiles**, including four Thinking variants. Local serving requires Linux, NVIDIA GPUs, Python 3.12, compatible drivers and CUDA build tools; `pip install .` only installs the evaluator. See the [deployment guide](deploy/README.md) for build prerequisites, model-specific environments and reference GPU counts.

This example serves Qwen3.5-4B on GPU 0 and port 8000. The environment and deployment output directories must be new:

```bash
# List supported profiles; this does not start a server.
bash deploy/serve.sh --list

# Install the model environment, then download the complete checkpoint.
bash deploy/install.sh --environment vllm --venv .venvs/vllm --python python3.12
.venvs/vllm/bin/hf download Qwen/Qwen3.5-4B --local-dir weights/qwen35-4b

# Terminal 1: keep this process running.
CUDA_VISIBLE_DEVICES=0 bash deploy/serve.sh \
  --profile qwen35-4b --model-path weights/qwen35-4b \
  --python .venvs/vllm/bin/python \
  --port 8000 --output outputs/services/qwen35-4b
```

The launcher creates **`outputs/services/qwen35-4b/model.json`**, containing the endpoint, served model name and evaluation settings. Pass this generated file to `--model-config`; `configs/models/*.json` are the launcher's input profiles. `--model-path` is a weight directory used only by the server, not an evaluation configuration.

In **Terminal 2**, from the same checkout, check readiness:

```bash
.venv/bin/embodied-eval probe \
  --config-path outputs/services/qwen35-4b/model.json
```

Continue after `ready` is `true`. The configuration file is written before weights finish loading, so its existence alone does not indicate readiness. `probe` checks model discovery; the one-question-per-task run below checks actual image inference.

For the local reference judge, follow the complete [judge deployment commands](deploy/README.md#local-judge--本地裁判). The `qwen35-a17b-fp8` profile on port 8001 with `--output outputs/services/judge` generates **`outputs/services/judge/judge.json`**. This is a large model with a reference tensor-parallel count of **8 GPUs**; weights and sufficient GPU memory are required. Use `judge.json` for judging, because its generation budget differs from `model.json`. You may instead configure an existing multimodal judge using Option B. Keep the judge settings fixed across models you compare.

If the model and judge share GPUs, use the [sequential workflow](deploy/README.md#sequential-evaluation--分阶段评测) to finish model inference before stopping that server and starting the judge. The default `all` workflow assumes both services remain available. To run only the model and score closed questions, replace `--judge-config ...` with `--skip-judge`; open questions remain unscored.

For another local model, select its profile, upstream checkpoint and environment from the [profile table](deploy/README.md#model-profiles--模型配置), then pass the newly generated `model.json`. InternVL Thinking also requires `--prompts-path configs/prompts/internvl35-14b-thinking.json` on every stage and resume.

### Option B: Connect existing model and judge services

Skip local deployment if you already have compatible services. Copy the templates to the ignored `outputs/` directory, then edit the copies:

```bash
mkdir -p outputs/configs
cp configs/model.example.json outputs/configs/model.json
cp configs/judge.example.json outputs/configs/judge.json
```

| Field in each JSON | What to fill in |
| --- | --- |
| `endpoint` | The complete chat-completions URL, typically ending in `/v1/chat/completions`; a base URL ending only in `/v1` is insufficient. |
| `model` | The exact model ID accepted by that service, usually listed by `/v1/models`; not the local weight path. |
| `reasoning_mode` | `disabled`, `enabled` or `provider_default`, matching the service. This declaration alone does not switch backend thinking on or off. |
| `generation` | Supported generation parameters and an output budget. Use only one of `max_tokens` or `max_completion_tokens`; omit unsupported optional fields such as `seed`. |
| `api_key_env` | Optional environment-variable name such as `MODEL_API_KEY` or `JUDGE_API_KEY`. Omit it if authentication is not required. |

The template model names are placeholders and must be replaced. Both services must accept the benchmark's image inputs; a text-only judge cannot perform this evaluation. For authenticated services, add `"api_key_env": "MODEL_API_KEY"` to the model JSON and `"api_key_env": "JUDGE_API_KEY"` to the judge JSON, then set the values in the evaluation terminal (Bash) without putting them in the files:

```bash
read -r -s -p "Model API key: " MODEL_API_KEY
export MODEL_API_KEY
read -r -s -p "Judge API key: " JUDGE_API_KEY
export JUDGE_API_KEY

.venv/bin/embodied-eval probe --config-path outputs/configs/model.json
.venv/bin/embodied-eval probe --config-path outputs/configs/judge.json
```

Skip the `read`/`export` commands for services without authentication. `probe` requires a `/models` discovery endpoint; if a provider does not expose one, verify its advertised model ID and use the small evaluation below to check chat-completions compatibility. For services on another machine, use an address reachable from the evaluator: `127.0.0.1` always refers to the machine running the evaluator. For this repository's launcher, see [host and authentication settings](deploy/README.md#other-hosts-and-resources--其他机器与资源).

### Pass the configurations to the evaluator

| Argument | Locally deployed example | Existing-service example |
| --- | --- | --- |
| `--model-config` | `outputs/services/qwen35-4b/model.json` | `outputs/configs/model.json` |
| `--judge-config` | `outputs/services/judge/judge.json` | `outputs/configs/judge.json` |
| `--python` | `.venv/bin/python` (evaluator environment) | `.venv/bin/python` |
| `--output` | A directory for evaluation results, separate from service records | Same requirement |

First run one question per task. The following uses the local configurations; for existing services, substitute the two paths from the right-hand column. A local model and an existing judge can also be combined.

```bash
bash evaluate.sh --python .venv/bin/python \
  --model-config outputs/services/qwen35-4b/model.json \
  --judge-config outputs/services/judge/judge.json \
  --per-task 1 --output outputs/evaluations/qwen35-4b-smoke
```

After the interface check succeeds, evaluate **all 4,467 questions** by omitting `--per-task` and choosing a **different output directory**:

```bash
bash evaluate.sh --python .venv/bin/python \
  --model-config outputs/services/qwen35-4b/model.json \
  --judge-config outputs/services/judge/judge.json \
  --output outputs/evaluations/qwen35-4b
```

The final report is written to `outputs/evaluations/qwen35-4b/scores/report.json` and `report.md`. Intermediate outputs are stored in `corpus/`, `model/`, `judge/`, `scores/` and `logs/` under that evaluation directory. No dataset-path argument is needed.

Repeat the same command to resume; retain `--per-task 1` when resuming a small run. Successful responses are reused. Changed model settings, deployment, prompts or question selection require a new evaluation output directory. Incomplete inference produces a partial report and stops before judging; incomplete inference or scoring exits with code 3. Inspect the logs before using `--retry-pending` for known failures or `--retry-unknown` for requests whose delivery is unknown.

`--stage prepare|run|judge|score` selects one stage. `--skip-judge` scores closed questions only and does not yield a complete overall accuracy. Configuration and output paths are relative to the calling directory; the dataset always comes from this checkout's `data/`. See `bash evaluate.sh --help` and the [workflow details](docs/EVALUATION.md#unified-shell-workflow).

### Individual CLI commands

The commands below use the edited existing-service configurations from Option B and the corpus prepared in the Data section. Substitute generated service configurations when using local deployments.

```bash
embodied-eval probe --config-path outputs/configs/model.json

embodied-eval run --corpus artifacts/corpus \
  --config-path outputs/configs/model.json --output artifacts/model_run \
  --canary-per-task 1

# Resume the same run after checking the first responses.
embodied-eval run --corpus artifacts/corpus \
  --config-path outputs/configs/model.json --output artifacts/model_run

embodied-eval score --corpus artifacts/corpus \
  --run-dir artifacts/model_run --output artifacts/model_scores
```

For open questions, use the fixed multimodal judge configured in `outputs/configs/judge.json`:

```bash
embodied-eval judge --corpus artifacts/corpus --run-dir artifacts/model_run \
  --config-path outputs/configs/judge.json --output artifacts/judge_run

embodied-eval score --corpus artifacts/corpus --run-dir artifacts/model_run \
  --judge-dir artifacts/judge_run --output artifacts/model_scores_with_judge
```

The inference process only reads prepared public inputs. Reference answers are stored separately in `private/answers.jsonl` for scoring. **Reference answers are publicly distributed in this release**; this separation prevents accidental prompt leakage and is not a hidden evaluation server.

Multiple-choice answers must be a single valid uppercase option letter. Yes/No uses the documented normalization. Open answers use the versioned semantic rubric; the released judge returns only `correct`/`incorrect` with integer scores `1`/`0`. Missing predictions and pending judgments stay in the denominator. See [evaluation details](docs/EVALUATION.md).

Each run records configuration and code hashes, request hashes, original responses, token usage, finish reasons and transmission attempts. Resume with the same configuration; changed inputs, code or settings require a new output directory. Repeating the same command resumes unfinished work without resampling successful responses.

## Tests

```bash
python -m pip install '.[test]'
python -m unittest discover -s eval/tests -v
```

Tests use generated images and local mock services. They cover scoring, GT isolation, media integrity, frozen run contracts, retry semantics, relocation, local deployment profiles, native-server HTTP behavior and model-specific prompt/answer handling. They do not require GPUs, the full media archive or model credentials.

For the completed full-data checks, see [release validation](docs/VALIDATION.md).

## Release status

This is release 0.2.1, with the v3 binary open-question judging protocol. The code is hosted at [JuntianZhang1216/EmboScope](https://github.com/JuntianZhang1216/EmboScope). Dataset download links and formal citation metadata will be added when available.

## License

The evaluation code is licensed under the [Apache License 2.0](LICENSE). Third-party components retain their own licenses; see [third-party notices](deploy/THIRD_PARTY.md). The question JSONL and evaluation media are distributed separately and require their own licensing terms.
