# Local model deployment / 本地模型部署

This directory contains the local inference implementations and settings used for the EmboScope paper: **20 models and 24 evaluation profiles**, plus the local Qwen judge configuration. It serves OpenAI-compatible chat completions consumed by the current evaluator. Cloud API models use the generic endpoint configuration and are not deployed here.

本目录提供论文中的 **20 个本地模型、24 组评测配置**，以及本地 Qwen 裁判配置。所有服务通过现有评测框架支持的 chat-completions 接口接入；开放题沿用当前发布版的 v3 二元裁判。代码仓库不包含权重、环境、服务日志或评测媒体。

## Quick start / 使用方法

Run these commands from the repository root. Keep the evaluator environment (`.venv`, Python 3.10+) separate from GPU serving environments (`.venvs/*`, Python 3.12). The server dependencies are not installed by `pip install .`.

Before evaluating, download `benchmark_data.jsonl` and `images/` into the checkout's fixed `data/` directory. The evaluation command reads them automatically; no data path argument is needed. 评测数据固定放在 `data/benchmark_data.jsonl` 和 `data/images/`，代码仓库仅保留空目录占位。

以下命令在仓库根目录执行。评测端与模型服务端使用独立环境；可以部署在不同机器上。模型服务需要 Linux、NVIDIA GPU、匹配的驱动与 CUDA 编译工具链。vLLM 安装器从固定提交构建，使用 CUDA 12.9 的 PyTorch；HY MoT 使用 PyTorch 2.6 / CUDA 12.4，并构建 FlashAttention。构建需要 Git、`patch`、C/C++ 编译器、`nvcc` 和联网下载依赖。

```bash
# Inspect profiles and installation commands without creating an environment.
bash deploy/serve.sh --list
bash deploy/install.sh --environment vllm --venv .venvs/vllm --dry-run

# Create a NEW isolated serving environment. This may compile CUDA extensions.
bash deploy/install.sh --environment vllm --venv .venvs/vllm --python python3.12

# Download the complete checkpoint, including processor code.
.venvs/vllm/bin/hf download Qwen/Qwen3.5-4B --local-dir weights/qwen35-4b

# This command stays in the foreground. Use a new --output for each deployment.
CUDA_VISIBLE_DEVICES=0 bash deploy/serve.sh \
  --profile qwen35-4b --model-path weights/qwen35-4b \
  --python .venvs/vllm/bin/python \
  --port 8000 --output outputs/services/qwen35-4b
```

In another terminal, enter the repository root and check the generated configuration against the running service. Install the evaluator in `.venv` first, as shown in the [main README](../README.md#install):

```bash
.venv/bin/embodied-eval probe \
  --config-path outputs/services/qwen35-4b/model.json
```

Wait for `ready: true` before evaluating. The launcher writes `model.json` before loading the model, so file creation alone is not a readiness check. This probe uses model discovery; a small evaluation is still needed to check image inference.

Pass **`outputs/services/qwen35-4b/model.json`** to `evaluate.sh --model-config`. Full evaluation additionally needs `--judge-config`: deploy the [local judge](#local-judge--本地裁判) or configure an [existing multimodal service](../README.md#option-b-connect-existing-model-and-judge-services). If the model and judge share GPUs, use the [sequential workflow](#sequential-evaluation--分阶段评测). With no judge, `--skip-judge` yields only closed-question scores.

| Deployment argument | What the user supplies |
| --- | --- |
| `--profile` | A profile from the table below or `bash deploy/serve.sh --list`. |
| `--model-path` | The complete downloaded checkpoint directory, including model and processor files. |
| `--python` | The selected serving environment's interpreter, such as `.venvs/vllm/bin/python`. The evaluator instead uses `.venv/bin/python`. |
| `--port` | A free service port; examples use 8000 for the model and 8001 for the judge. |
| `--output` | A new directory for generated configs and deployment records, not the evaluation result directory. |
| `--endpoint` | When needed, the full chat-completions URL reachable from the evaluation machine. |
| `--api-key-env` | Optional name of the environment variable holding the API key, not the key itself. |

服务启动时生成 `model.json`、`deployment.json` 和 `evaluation-command.txt`。`deployment.json` 自动保存权重目录、checkpoint 配置哈希、启动命令、部署代码哈希、依赖版本和补丁校验结果；对应指纹写入评测配置，因此变更部署后需要新的评测目录。它不对全部权重张量做哈希验证，也不保证不同 GPU 上生成完全一致。

Use `--dry-run` with a profile, model path and output to inspect the command without an installed backend or downloaded checkpoint. Normal startup checks the dependency versions and required patch hashes before loading weights. Both the installation target and deployment output must be new directories. The foreground process can be stopped with Ctrl-C; the scripts do not find or terminate other services.

## Model profiles / 模型配置

The TP column is the reference tensor-parallel GPU count, not a minimum memory guarantee. Context and processor settings are in [`models.json`](models.json). Baseline profiles use temperature 0, top-p 1 and seed 42; the output budget is 1,024 tokens except MiMo (8,192). The common server limit is 32 images per request, max-num-seqs 2, GPU memory utilization 0.85, and eager execution. GPU memory also depends on image sizes and concurrency.

| Profile | Upstream checkpoint | Environment | TP | Context |
| --- | --- | --- | ---: | ---: |
| `qwen35-4b` | Qwen/Qwen3.5-4B | vllm | 1 | 131072 |
| `qwen35-27b` | Qwen/Qwen3.5-27B | vllm | 2 | 131072 |
| `qwen35-a17b-fp8` | Qwen/Qwen3.5-397B-A17B-FP8 | vllm | 8 | 131072 |
| `internvl35-8b` | OpenGVLab/InternVL3_5-8B | vllm | 1 | 40960 |
| `internvl35-14b` | OpenGVLab/InternVL3_5-14B | vllm | 1 | 40960 |
| `internvl35-30b-a3b` | OpenGVLab/InternVL3_5-30B-A3B | vllm | 2 | 40960 |
| `internvl35-38b` | OpenGVLab/InternVL3_5-38B | vllm | 2 | 40960 |
| `gemma3-4b-it` | google/gemma-3-4b-it | vllm | 1 | 32768 |
| `gemma3-27b-it` | google/gemma-3-27b-it | vllm | 2 | 32768 |
| `minicpm-v45` | openbmb/MiniCPM-V-4_5 | vllm | 1 | 32768 |
| `minicpm-o45` | openbmb/MiniCPM-o-4_5 | vllm | 1 | 40960 |
| `ovis2-8b` | AIDC-AI/Ovis2-8B | vllm | 1 | 32768 |
| `phi4-multimodal` | microsoft/Phi-4-multimodal-instruct | phi | 1 | 65536 |
| `ministral3-14b-instruct` | mistralai/Ministral-3-14B-Instruct-2512 | vllm | 1 | 131072 |
| `glm46v` | zai-org/GLM-4.6V | glm | 4 | 131072 |
| `cosmos-reason1-7b` | nvidia/Cosmos-Reason1-7B | vllm | 1 | 128000 |
| `robobrain20-7b` | BAAI/RoboBrain2.0-7B | vllm | 1 | 128000 |
| `mimo-embodied-7b` | XiaomiMiMo/MiMo-Embodied-7B | vllm | 1 | 128000 |
| `hy-embodied-05-mot-2b` | tencent/HY-Embodied-0.5 | hy-mot | 1 | native: 262144 |
| `hy-embodied-vlm-10` | tencent/Hy-Embodied-VLM-1.0 | hy-vlm | 4 | 32768 |

There are four additional profiles:

| Thinking profile | Output tokens | Prompt bundle |
| --- | ---: | --- |
| `minicpm-v45-thinking` | 8192 | Default |
| `qwen35-27b-thinking` | 8192 | Default |
| `glm46v-thinking` | 8192 | Default |
| `internvl35-14b-thinking` | 7168 | `configs/prompts/internvl35-14b-thinking.json` |

For InternVL Thinking, **pass the supplied prompt bundle to every evaluation/resume stage**:

```bash
bash evaluate.sh --python .venv/bin/python \
  --model-config outputs/services/internvl35-14b-thinking/model.json \
  --prompts-path configs/prompts/internvl35-14b-thinking.json \
  --judge-config outputs/services/judge/judge.json \
  --output outputs/evaluations/internvl35-14b-thinking
```

Its launcher enables the Qwen3 reasoning parser. The current online answer adapter uses the text after the final `</think>` delimiter; unfinished reasoning has no final answer. The original server response remains available in the run records. The prompt bundle changes only the solver system prompt and keeps the released binary judge rubric. Other serialized answer adapters cover GLM box tokens, RoboBrain/HY answer tags and MiMo reasoning tags; they do not extract option letters from explanations.

## Local judge / 本地裁判

The local reference judge uses **Qwen3.5-397B-A17B-FP8**, with a reference tensor-parallel count of **8 GPUs**. This is a large checkpoint; the number of GPUs alone does not guarantee sufficient memory. Reuse the vLLM environment installed above, or install it on the judge machine first. Download the judge weights separately:

```bash
.venvs/vllm/bin/hf download Qwen/Qwen3.5-397B-A17B-FP8 \
  --local-dir weights/qwen35-a17b-fp8
```

Start the following server only when GPUs 0–7 are available. **GPU 0 overlaps with the model example above**: on the same GPU pool, first complete the inference stage described in [sequential evaluation](#sequential-evaluation--分阶段评测), then stop the model server. To keep both services running, allocate separate GPUs or machines and adjust the device selection and endpoint accordingly.

```bash
# Foreground process; keep this terminal open during judging.
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash deploy/serve.sh \
  --profile qwen35-a17b-fp8 --model-path weights/qwen35-a17b-fp8 \
  --python .venvs/vllm/bin/python \
  --port 8001 --output outputs/services/judge
```

From the evaluation terminal, wait for readiness:

```bash
.venv/bin/embodied-eval probe \
  --config-path outputs/services/judge/judge.json
```

The launcher writes both `model.json` (1,024 output tokens for solving) and **`judge.json` (2,048 tokens for judging)**. Pass `outputs/services/judge/judge.json` to `--judge-config`. The judge uses non-thinking mode, temperature 0, top-p 1 and seed 42. Use this judge configuration consistently across compared models. The static reference template is [`configs/judge.local.example.json`](../configs/judge.local.example.json); the generated file contains the actual deployment endpoint and identifier.

两个服务在独立资源上保持可用时，可直接按主 README 的命令运行完整评测。共用 GPU 时不要同时照抄两个启动命令，应使用下面的分阶段流程；暂时没有裁判时可使用 `--skip-judge`，但不产生完整总分。裁判采用当前发布版的二元协议，只输出正确/错误及 1/0。

## Sequential evaluation / 分阶段评测

The solver and judge do not have to occupy GPU memory at the same time. This workflow uses the same evaluation output throughout; model predictions are saved before the model server is stopped. It produces the same evaluation stages as `all`.

模型与裁判可以分时使用同一组 GPU。下面三条评测命令必须使用同一输出目录；先准备语料并完成被评模型推理，再切换到裁判。已有成功响应不会重新采样。

**1. With the model service ready, prepare inputs and complete model inference:**

```bash
bash evaluate.sh --python .venv/bin/python \
  --stage prepare --output outputs/evaluations/qwen35-4b

bash evaluate.sh --python .venv/bin/python \
  --stage run --model-config outputs/services/qwen35-4b/model.json \
  --output outputs/evaluations/qwen35-4b
```

The `run` stage requires the prepared corpus and does not need a judge configuration. Continue only after it exits successfully; exit code 3 means predictions are incomplete. Resolve the reported failures and resume this stage before switching services.

**2. Stop the model server with Ctrl-C in its own terminal, then start the judge** using the commands in the [local judge section](#local-judge--本地裁判). In the evaluation terminal, wait for the judge's `probe` result to report `ready: true`.

**3. Judge the saved answers and write the final scores:**

```bash
bash evaluate.sh --python .venv/bin/python \
  --stage judge --judge-config outputs/services/judge/judge.json \
  --output outputs/evaluations/qwen35-4b
```

The judge stage also runs scoring; no additional `score` command or running model service is needed. Read `outputs/evaluations/qwen35-4b/scores/report.json` and `report.md`.

To check one question per task first, add `--per-task 1` to **all three evaluation commands** and use a separate directory such as `outputs/evaluations/qwen35-4b-smoke`. For InternVL Thinking, also retain its `--prompts-path` on every stage. Service restarts require a new deployment output; use the new generated configuration for a new inference/judging run rather than overwriting an existing deployment's records.

## Dependencies and model compatibility / 依赖与兼容实现

- Common vLLM: [commit `c21d0039ecc85b06617034ff2166a1fb79309d53`](https://github.com/vllm-project/vllm/commit/c21d0039ecc85b06617034ff2166a1fb79309d53), version `0.16.1rc1.dev167+gc21d0039e`, PyTorch `2.10.0+cu129`, Transformers `4.57.6`, Triton `3.6.0`. This is a source build, not an assumed PyPI release.
- GLM-4.6V: dedicated environment with Transformers `5.2.0` and huggingface_hub `1.4.1`. The installer changes the dedicated vLLM build's Transformers dependency declaration to this tested version; model execution code remains at the pinned commit.
- Phi-4: SciPy `1.17.1`, the checkpoint's `vision-lora`, and a small patch that zero-pads ragged image crops **and** crop masks together. Input images remain intact.
- HY-VLM-1.0: the official [HY plugin source](https://github.com/Tencent-Hunyuan/HY-Embodied/tree/16289e99111ee362a23f93d5b3a463ddc844decc/Hy-Embodied-VLM-1.0/inference/vllm), a hash-checked compatibility patch, and the OpenAI-content chat template. The reasoning parser preserves `enable_thinking` for template rendering. Every image placeholder and the system instruction are retained.
- HY MoT: [`hy_mot_server.py`](hy_mot_server.py), PyTorch `2.6.0`, [Transformers commit `9293856c419762ebf98fbe2bd9440f9ce7069f1a`](https://github.com/huggingface/transformers/commit/9293856c419762ebf98fbe2bd9440f9ce7069f1a), FlashAttention `2.8.3`. This server preserves the native processor and template, validates image/token counts, rejects context overflow, and records input/output audits. Its frozen API supports non-streaming, non-thinking generation with a 1,024-token output budget.

The installer pins inference-critical dependencies and records the complete resolved environment in `emboscope-freeze.txt`; the requirements are **not a complete transitive dependency lock**. Compatibility patches apply only when the upstream file hashes match, and the launcher verifies their resulting hashes. Source archives, model weights and third-party implementations are downloaded from their upstream projects into the user's chosen environment. See [third-party notices](THIRD_PARTY.md).

目前公开的整理版通过 CPU/模拟服务检查；本次整理未重建 CUDA 环境、未启动真实模型。使用前应在目标硬件完成服务启动及 `--per-task 1` 联调，再运行完整评测。

## Deployment records / 部署记录

The launcher records the local weight directory, checkpoint configuration hash and runtime environment, and generates a deployment identifier in `model.json` and `judge.json`. No manual weight version is required. The identifier distinguishes deployment records; it is not a hash of all weight tensors or proof of an upstream checkpoint commit. After changing weights, start a new deployment with a new `--output` and use its generated configuration for a new evaluation.

用户只需指定模型配置和权重目录，部署信息与标识由程序自动记录。更换权重后，使用新的 `--output` 重新部署，并使用生成的配置开始新评测。

## Other hosts and resources / 其他机器与资源

`--host` is the server bind address; `--endpoint` is the address the evaluator calls. For a model on another machine, add the following flags to its launch command, replacing `SERVER_HOST` with the host or IP reachable from the evaluation machine:

```bash
--host 0.0.0.0 --endpoint http://SERVER_HOST:8000/v1/chat/completions
```

For the judge example, use port 8001 in the endpoint. Without `--endpoint`, a wildcard bind still generates a loopback client URL, which would point at the wrong machine when copied to a remote evaluator. Copy the generated `model.json` or `judge.json` to the evaluation machine and pass that copy to `evaluate.sh`; weights stay on the serving machine and benchmark data stays in the evaluator's `data/` directory.

To enable authentication, set a secret environment variable and pass its **name** with `--api-key-env MODEL_API_KEY` (or `JUDGE_API_KEY` for the judge). Set the same variable and value on the evaluator machine. The generated JSON contains only the variable name; secrets are not placed in launch arguments or deployment records. See the main README for [Bash key-entry commands](../README.md#option-b-connect-existing-model-and-judge-services).

`CUDA_VISIBLE_DEVICES` selects devices. vLLM profiles support explicit `--tensor-parallel`, `--max-model-len`, `--gpu-memory-utilization` and `--max-num-seqs` overrides, recorded in the deployment manifest. HY MoT uses one GPU and the checkpoint's native context. Processor limits remain the model-specific reference values. No launcher flag silently drops frames or truncates a question.

For multiple replicas, start separate services with distinct ports/outputs and configure `endpoints` using the evaluator's existing multi-endpoint support. Keep model, weights, processor, generation settings and prompt bundle identical across replicas.
