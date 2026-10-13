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

[![Task-level capability profiles of five baselines across the three EmboScope levels](docs/figures/capability_profiles.png)](docs/figures/capability_profiles.pdf)

**Capability profiles.** Five representative baselines on the same 0–100% accuracy scale. Each axis represents a task; polygon areas are not aggregate scores and should not be compared across panels. [View PDF](docs/figures/capability_profiles.pdf).

[![Accuracy distributions across evaluated models for all 14 EmboScope tasks](docs/figures/task_accuracy.png)](docs/figures/task_accuracy.pdf)

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

Prepare the complete evaluation corpus:

```bash
embodied-eval prepare --output artifacts/corpus
```

For a quick interface check, prepare one question per task in a separate corpus:

```bash
embodied-eval prepare --per-task 1 --output artifacts/smoke_corpus
```

Run the installed `embodied-eval` CLI from the checkout or one of its subdirectories. The Shell entry point also works from any working directory when invoked by path; it always uses its own checkout's `data/`. Prepared corpora record relative media links, so moving the checkout together with `data/` and its evaluation outputs preserves their layout.

## Evaluate

### Deploy the paper's local models

[`deploy/`](deploy/README.md) provides **20 local models and 24 evaluation profiles**, including the four Thinking variants. It includes vLLM launchers, the native HY MoT server, Phi/HY compatibility patches, and model-specific settings. The code archive contains the deployment code; weights and model environments are downloaded separately.

```bash
bash deploy/serve.sh --list
bash deploy/install.sh --environment vllm --venv .venvs/vllm --dry-run
```

Follow the [deployment guide](deploy/README.md) to install a separate GPU environment and start a model. Select `--profile` and point `--model-path` to the downloaded weights; deployment information is recorded automatically. Each deployment generates a `model.json` for `evaluate.sh`; the Qwen A17B profile also generates a local `judge.json`. InternVL Thinking requires `--prompts-path configs/prompts/internvl35-14b-thinking.json`.

### Unified Shell entry point

For locally deployed models, pass the generated `model.json` and `judge.json` directly to the evaluator. For existing services, configure the endpoints, model names and generation parameters in the JSON templates, then run:

```bash
bash evaluate.sh \
  --model-config configs/model.example.json \
  --judge-config configs/judge.example.json \
  --output artifacts/my_model
```

The script prepares the corpus, runs inference, judges open answers and writes the final report to `artifacts/my_model/scores/report.json` and `report.md`. Stage outputs are stored in `corpus/`, `model/`, `judge/`, `scores/` and `logs/` under that directory. It uses the Python package in this checkout.

Repeat the same command to resume. The script checks that the dataset, question selection, prompts and evaluator still match; successful responses are reused. Incomplete inference produces a partial report and stops before judging. Incomplete inference or scoring exits with code 3; inspect the logs and use `--retry-pending` or `--retry-unknown` only for the corresponding failures.

For a quick end-to-end check, add `--per-task 1` and use a separate output directory such as `artifacts/smoke`. Retain that selection flag when resuming. Use `--stage prepare|run|judge|score` to execute a single stage, or `--skip-judge` for closed-question scoring. With `--skip-judge`, open questions remain unscored and the report does not contain a complete overall accuracy. `--python` selects the interpreter. Configuration and output paths are resolved from the calling directory; the dataset always comes from this checkout's `data/`.

See `bash evaluate.sh --help` and the [workflow details](docs/EVALUATION.md#unified-shell-workflow).

### Individual CLI commands

Copy `configs/model.example.json` to a local configuration, set the full chat-completions endpoint and model identifier, then declare the model's reasoning mode. For authentication, add `"api_key_env": "MODEL_API_KEY"` and set that environment variable. Keys are never written into configuration files by the evaluator.

```bash
embodied-eval probe --config-path configs/model.example.json

embodied-eval run --corpus artifacts/corpus \
  --config-path configs/model.example.json --output artifacts/model_run \
  --canary-per-task 1

# Resume the same run after checking the first responses.
embodied-eval run --corpus artifacts/corpus \
  --config-path configs/model.example.json --output artifacts/model_run

embodied-eval score --corpus artifacts/corpus \
  --run-dir artifacts/model_run --output artifacts/model_scores
```

For open questions, configure a fixed multimodal judge in `configs/judge.example.json`:

```bash
embodied-eval judge --corpus artifacts/corpus --run-dir artifacts/model_run \
  --config-path configs/judge.example.json --output artifacts/judge_run

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
