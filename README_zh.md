# EmboScope

[English](README.md)

本仓库为论文 **《EmboScope: A Multi-Source Comprehensive Benchmark for Embodied Perception, Understanding, and Planning》对应的评测代码与数据集**，包括评测输入准备、模型推理、封闭题评分，以及开放题裁判。

本版本包含全部 **4,467 道合成视觉问答题、14 类任务**及评测代码。三层题量为 1,241 / 1,752 / 1,474，任务数为 4 / 6 / 4。动作预测与选择（APS）归入 L3。

## Benchmark 概览

### 数据构建框架

[![EmboScope 构建框架：交互数据、共享表示、规则式问答生成与三层能力](docs/figures/overview.png)](docs/figures/overview.pdf)

论文中的构建框架将交互观察整理为包含对象、动作、状态和时间关系的共享表示，再按各任务的观察范围生成问题。下方示例展示同一段交互如何构造三层能力问题；动作预测时将未来观察保留为参考证据。[查看 PDF 原图](docs/figures/overview.pdf)。

### 代表性任务示例

[![EmboScope 感知、具身语义、规划与推理三个层级的题目示例](docs/figures/task_examples.png)](docs/figures/task_examples.pdf)

三个示例分别展示相对近远判断、帧间动作识别和动作预测与选择。绿色标出参考答案。图中题干经过缩写，(b)、(c) 省略了干扰选项；(c) 展示第 1、3、6、8 个输入，其中第 8 帧是当前观察。[查看 PDF 原图](docs/figures/task_examples.pdf)。

### 论文报告的评测结果

<p align="center">
  <a href="docs/figures/capability_profiles.pdf">
    <img src="docs/figures/capability_profiles.png" alt="五个代表性基线在 EmboScope 三层任务上的能力雷达图" width="600">
  </a>
</p>

**模型能力分布。** 五个代表性基线统一使用 0–100% 准确率刻度，每条轴对应一个任务；多边形面积不表示综合得分，也不用于跨面板比较。[查看 PDF 原图](docs/figures/capability_profiles.pdf)。

<p align="center">
  <a href="docs/figures/task_accuracy.pdf">
    <img src="docs/figures/task_accuracy.png" alt="被评模型在 EmboScope 全部 14 类任务上的准确率分布" width="600">
  </a>
</p>

**各任务准确率分布。** 散点表示各模型的准确率，短横线表示均值，小提琴图表示被评模型的得分分布。[查看 PDF 原图](docs/figures/task_accuracy.pdf)。

<details>
<summary>图中任务缩写</summary>

| 层级 | 缩写 | 任务 |
| --- | --- | --- |
| L1 | RD | 相对近远判断 |
| L1 | AM | 属性与状态匹配 |
| L1 | CVM | 跨视角状态匹配 |
| L1 | FC | 连续交互缺帧补全 |
| L2 | AFM | 动作与帧匹配 |
| L2 | ABF | 帧间动作识别 |
| L2 | SFA | 步骤与图组对齐 |
| L2 | ASD | 动作序列描述 |
| L2 | AA | 相邻动作识别 |
| L2 | SC | 指定步骤完成判定 |
| L3 | APS | 动作预测与选择 |
| L3 | PC | 目标进度比较 |
| L3 | PO | 目标进度排序 |
| L3 | PAD | 目标进度异常检测 |

</details>

## 数据结构

- `evaluate.sh`：串联数据准备、模型推理、开放题裁判与评分的统一入口。
- `data/`：固定的数据下载位置，代码仓库中仅保留空目录占位。
- `data/benchmark_data.jsonl`：下载后放入的全部题目，固定 ID 为 `q000001` 至 `q004467`，没有 `source` 字段。
- `data/images/`：下载后放入的全部视觉输入，按内容哈希命名，共 42,637 张独立图片，约 4.65 GB。
- `metadata/`：图片校验清单、数据版本与统计、229 道视频题的固定帧位置。
- `eval/`：可安装的评测包、辅助脚本和测试。
- `configs/`：通用模型与裁判配置、24 组本地模型评测配置，以及 InternVL Thinking 专用提示词。
- `deploy/`：本地模型启动入口、依赖版本、HY 原生推理服务及 Phi/HY 兼容补丁。
- `docs/figures/`：README 使用的论文展示图，包含 PNG 预览和 PDF 原图，随代码仓库发布。

图片引用形如 `images/<sha256>.jpg`，相对于 JSONL 所在目录解析。229 道视频题使用固定图片序列，按题目指定顺序与图文绑定提供视觉输入。相同图片字节只存一份，题目中的重复引用仍然保留。

**数据下载链接目前尚未公布。** 以下命令假设你已取得 v0.2.1 的完整数据包，或对应的 JSONL 和媒体文件。

clone 仓库后，下载对应版本的 JSONL 和媒体，将它们放到 **`data/benchmark_data.jsonl` 和 `data/images/`**。评测入口固定读取此位置，无需填写数据路径。题目和媒体由 Git 忽略，也不打入代码包；校验清单已随代码放在根目录 `metadata/` 中，无需用户另行准备。

在仓库根目录，将完整数据包解压到 `data/`：

```bash
tar -xzf /下载位置/embodied-benchmark-data-v0.2.1.tar.gz -C data
```

解压后应直接得到 `data/benchmark_data.jsonl` 和 `data/images/`，不要再嵌套一层 `data/`。单独下载 JSONL 和图片时也保持此结构。README 展示图仍随代码发布。详见[打包说明](docs/RELEASING.md)。

## 快速开始

`evaluate.sh` 通过 HTTP 调用**已经启动的模型服务**，依次准备输入、请求模型回答、裁判开放题并生成分数。它不会自动下载权重或启动模型服务器。完整评测需要被评模型和多模态裁判各一份 JSON 配置，服务均须支持带多张图片的 OpenAI 兼容 chat-completions 接口。

### 1. 安装评测环境

以下命令均在仓库根目录执行。评测端支持 Linux/macOS、Python 3.10 及以上；先按上面的数据结构放好 JSONL 和图片。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
embodied-eval validate --output artifacts/data_validation.json
```

`.venv` 用于运行评测。模型服务单独使用 `.venvs/*`，不能把 `pip install .` 当成模型部署环境的安装步骤。已有 API 服务的用户可直接选择下面的方式 B。

### 2A. 在本地部署模型

仓库提供论文中的 **20 个本地模型、24 组评测配置**，包含 4 组 Thinking 实验。本地服务需要 Linux、NVIDIA GPU、Python 3.12、匹配的驱动和 CUDA 编译工具链。完整安装条件、各模型的环境及参考 GPU 数量见[部署说明](deploy/README.md)。

下面以 Qwen3.5-4B 为例，在 GPU 0、端口 8000 上启动服务。安装环境目录和部署输出目录都必须是尚不存在的新目录：

```bash
# 列出支持的配置；这一步不会启动服务。
bash deploy/serve.sh --list

# 实际安装推理环境并下载完整权重。
bash deploy/install.sh --environment vllm --venv .venvs/vllm --python python3.12
.venvs/vllm/bin/hf download Qwen/Qwen3.5-4B --local-dir weights/qwen35-4b

# 终端 1：启动后保持该进程运行。
CUDA_VISIBLE_DEVICES=0 bash deploy/serve.sh \
  --profile qwen35-4b --model-path weights/qwen35-4b \
  --python .venvs/vllm/bin/python \
  --port 8000 --output outputs/services/qwen35-4b
```

程序会生成 **`outputs/services/qwen35-4b/model.json`**，已经包含服务地址、模型名称与评测参数。评测时将此文件传给 `--model-config`；`configs/models/*.json` 是部署程序读取的预设。`--model-path` 指向权重目录，只在部署时使用，不能把权重路径传给评测配置参数。

另开**终端 2**，进入同一仓库并检查服务：

```bash
.venv/bin/embodied-eval probe \
  --config-path outputs/services/qwen35-4b/model.json
```

等待 `ready` 为 `true` 再继续。配置文件在模型完成加载之前就会写出，文件存在不代表服务已就绪；`probe` 检查模型发现接口，后面的每类 1 题评测才会实际测试图片推理。

本地参考裁判的完整命令见[裁判部署](deploy/README.md#local-judge--本地裁判)。使用 `qwen35-a17b-fp8`、端口 8001 和 `--output outputs/services/judge` 后，会生成 **`outputs/services/judge/judge.json`**。该大型模型的参考配置使用 **8 张 GPU**，需要另行下载权重并具备足够显存。裁判应使用 `judge.json`，它的生成长度与同目录下的 `model.json` 不同；也可以按方式 B 接入已有多模态裁判。比较不同模型时应固定裁判设置。

模型和裁判共用 GPU 时，按[分阶段评测](deploy/README.md#sequential-evaluation--分阶段评测)先完成模型推理，停止被评服务后再启动裁判。默认的完整流程要求两个服务均保持可用。暂时没有裁判时，可将 `--judge-config ...` 替换成 `--skip-judge`，这时只会得到封闭题分数。

换用其他本地模型时，从[模型表](deploy/README.md#model-profiles--模型配置)选择对应的 profile、权重仓库和安装环境，再使用新部署生成的配置。InternVL Thinking 在每个评测阶段及续跑时还须添加 `--prompts-path configs/prompts/internvl35-14b-thinking.json`。

### 2B. 接入已有模型与裁判 API

已有兼容服务时无需本地部署。先将模板复制到已被 Git 忽略的 `outputs/` 目录，再编辑副本：

```bash
mkdir -p outputs/configs
cp configs/model.example.json outputs/configs/model.json
cp configs/judge.example.json outputs/configs/judge.json
```

两份配置分别填写：

| JSON 字段 | 填写内容 |
| --- | --- |
| `endpoint` | 完整的 chat-completions 地址，通常以 `/v1/chat/completions` 结尾，不能只填到 `/v1`。 |
| `model` | 服务实际接受的模型 ID，通常可以从 `/v1/models` 查询；不是本地权重目录。 |
| `reasoning_mode` | 按服务实际情况填写 `disabled`、`enabled` 或 `provider_default`；此声明本身不会切换后端思考模式。 |
| `generation` | 接口支持的生成参数及输出长度；`max_tokens` 与 `max_completion_tokens` 只能保留一个，不支持的可选字段（例如 `seed`）应删除。 |
| `api_key_env` | 可选，填写保存密钥的环境变量名，例如 `MODEL_API_KEY` 或 `JUDGE_API_KEY`；无需鉴权时不填写。 |

模板中的模型名称是占位符，不能原样使用。被评服务和裁判都需要接受评测图片，纯文本裁判不适用。需要鉴权时，在模型 JSON 中添加 `"api_key_env": "MODEL_API_KEY"`，在裁判 JSON 中添加 `"api_key_env": "JUDGE_API_KEY"`，然后在运行评测的 Bash 终端输入密钥：

```bash
read -r -s -p "Model API key: " MODEL_API_KEY
export MODEL_API_KEY
read -r -s -p "Judge API key: " JUDGE_API_KEY
export JUDGE_API_KEY

.venv/bin/embodied-eval probe --config-path outputs/configs/model.json
.venv/bin/embodied-eval probe --config-path outputs/configs/judge.json
```

无需鉴权的服务跳过 `read`/`export` 即可，密钥不要写入 JSON。`probe` 依赖 `/models` 发现接口；供应商未提供该接口时，先确认服务公布的模型 ID，再用每类 1 题评测检查 chat-completions 是否兼容。跨机器部署时应填写评测机能访问的地址：`127.0.0.1` 始终指向运行评测的那台机器。本仓库的监听地址、客户端地址和鉴权配置见[跨机器部署说明](deploy/README.md#other-hosts-and-resources--其他机器与资源)。

### 3. 将配置传给评测脚本

| 参数 | 本地部署示例 | 已有 API 示例 |
| --- | --- | --- |
| `--model-config` | `outputs/services/qwen35-4b/model.json` | `outputs/configs/model.json` |
| `--judge-config` | `outputs/services/judge/judge.json` | `outputs/configs/judge.json` |
| `--python` | `.venv/bin/python`，即评测环境 | `.venv/bin/python` |
| `--output` | 评测结果目录，与服务部署记录目录分开 | 要求相同 |

先运行每类 1 题。以下命令使用本地部署生成的配置；使用已有 API 时，将两个配置路径替换为表格右列。也可以组合本地被评模型和已有裁判服务。

```bash
bash evaluate.sh --python .venv/bin/python \
  --model-config outputs/services/qwen35-4b/model.json \
  --judge-config outputs/services/judge/judge.json \
  --per-task 1 --output outputs/evaluations/qwen35-4b-smoke
```

检查通过后，去掉 `--per-task`，并使用**另一个输出目录**运行全部 **4,467 题**：

```bash
bash evaluate.sh --python .venv/bin/python \
  --model-config outputs/services/qwen35-4b/model.json \
  --judge-config outputs/services/judge/judge.json \
  --output outputs/evaluations/qwen35-4b
```

结果写入 `outputs/evaluations/qwen35-4b/scores/report.json` 和 `report.md`，中间文件位于该评测目录的 `corpus/`、`model/`、`judge/`、`scores/`、`logs/` 下。数据位置固定，无需传递数据路径。

重复执行相同命令即可续跑，已完成的响应会直接复用。续跑小样本时保留 `--per-task 1`；更换配置、部署、提示词或题目选择时使用新的评测输出目录。推理未完成时先保存阶段报告并停止，待推理完成后再启动裁判。推理或评分未完成时退出码为 3，应先检查日志，再按错误类型使用 `--retry-pending` 或 `--retry-unknown`。

常用选项：

| 选项 | 用途 |
| --- | --- |
| `--per-task 1` | 每类取 1 题检查完整流程；使用独立输出目录，续跑时保留此参数 |
| `--stage prepare/run/judge/score` | 选择一个阶段单独执行，参数取其中一个值；分阶段运行时先 `prepare` |
| `--skip-judge` | 只计算封闭题分数；开放题保持未评分，总准确率不视为完整结果 |
| `--prompts-path 路径` | 指定提示词包；InternVL Thinking 使用随仓库提供的专用文件 |
| `--python 路径` | 指定评测环境的 Python 解释器 |

完整参数见 `bash evaluate.sh --help`。`embodied-eval` 命令也可单独使用，见[英文说明](README.md#individual-cli-commands)与[评分规则](docs/EVALUATION.md)。

`text_list[-1]` 为参考答案；编译器只向被评模型发送之前的题干、选项和公开媒体。所有参考答案随本版公开，`private/answers.jsonl` 仅表示评测内部的输入与评分隔离。

已准备语料通过相对路径定位数据；搬迁时保留仓库、`data/` 和评测输出的相对布局。安装后的 `embodied-eval` 命令在仓库内执行；`evaluate.sh` 可以从其他工作目录按路径调用，始终读取脚本所在仓库的 `data/`。配置和输出路径按运行命令时的工作目录解析。

## 测试

```bash
python -m pip install '.[test]'
python -m unittest discover -s eval/tests -v
```

测试使用生成的小图片和本地模拟服务，不需要 GPU、真实模型接口或完整媒体包。

当前发布版本为 0.2.1，开放题采用 v3 二元判定协议，仅返回正确/错误及 1/0。代码托管于 [JuntianZhang1216/EmboScope](https://github.com/JuntianZhang1216/EmboScope)，数据下载链接及正式引用信息后续补充。

## 许可证

评测代码采用 [Apache License 2.0](LICENSE)。第三方组件保留各自的许可证，详见[第三方声明](deploy/THIRD_PARTY.md)。题目 JSONL 和评测媒体单独分发，其许可条款另行说明。
