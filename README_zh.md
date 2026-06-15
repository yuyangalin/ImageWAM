# ImageWAM

**ImageWAM: Do World Action Models Really Need Video Generation, or Just Image Editing?** 的官方代码仓库。

[English](./README.md) | [中文](./README_zh.md)

项目主页、论文链接、模型权重和数据集链接会在正式公开前补充。

ImageWAM 是一组基于图像编辑模型基座的 world action model。本仓库包含论文实验中在 LIBERO、LIBERO-plus 和 RoboTwin 上使用的训练与评测代码。

建议从 **FLUX.2 ImageWAM** 开始使用；这一版本提供了4B和9B两个变体，基于FLUX.2 [klein] 4B/9B base，提供系列模型中最强大的性能。仓库同时提供 **OmniGen2 ImageWAM** 和 **Ovis-U1 ImageWAM** 的训练与评测入口，这两个模型基于 OmniGen2 与 Ovis-U1 构建，同样提供了较为良好的性能，其中 Ovis-U1 变体是系列中最小的模型（用于图像编辑的DiT仅1.1B），但在诸多方面与更大的变体相媲美。

下文所有命令默认都在仓库根目录执行。

## 目录

- [仓库结构](#仓库结构)
- [基础安装](#基础安装)
- [模型准备](#模型准备)
- [数据准备](#数据准备)
- [Benchmark 环境](#benchmark-环境)
- [训练](#训练)
- [评测](#评测)
- [Release 权重](#release-权重)
- [致谢](#致谢)
- [引用](#引用)

## 仓库结构

```text
ImageWAM/
├── configs/                  # 模型、数据、任务和 benchmark 配置
├── docs/                     # 更详细的安装、数据、模型和依赖说明
├── experiments/              # LIBERO / RoboTwin 评测 manager
├── scripts/
│   ├── flux2/                # FLUX.2 ImageWAM 训练与评测入口
│   ├── omnigen2/             # OmniGen2 ImageWAM 训练与评测入口
│   ├── ovis_u1/              # Ovis-U1 ImageWAM 训练与评测入口
│   ├── data/                 # 数据处理工具
│   └── setup/                # benchmark 环境安装辅助脚本
├── src/imagewam/             # ImageWAM 核心代码
└── third_party/              # 随仓库保留的 benchmark / model adapter 代码
```

此外，还需要在本地准备数据集（默认放置在`./data`）、预训练模型权重；生成 ActionDiT 初始化权重（默认放置在./checkpoints/）。

## 基础安装

ImageWAM 使用 `uv` 管理 Python 依赖。我们测试过的推荐环境是 CUDA 11.8、Python 3.11 和 PyTorch 2.7.1。

```bash
uv sync --python 3.11 --extra shared
source .venv/bin/activate
```

复制本地配置模板：

```bash
cp .env.example .env.local
```

`scripts/` 下的 shell 入口会自动读取 `.env.local`。你可以把本地路径写进 `.env.local`，也可以直接在 shell 里 `export`。

常用变量：

```bash
export DATA_ROOT=/path/to/datasets # 对于不同的数据集有各自的data_root
export MODEL_ROOT=/path/to/model/checkpoints # 用于存放checkpoint 
export OUTPUT_ROOT=./runs
```

基础安装只包含公共依赖。不同模型变体需要的源码路径、模型权重和依赖见下一节。

## 模型准备

这一部分所做的准备主要是准备所需的外部模型代码库，以及必要的模型下载。

### FLUX.2 ImageWAM

下面先介绍建议优先使用的 FLUX.2 ImageWAM 配置。

如果使用 FLUX.2 变体，请将 `transformers` 切换到 FLUX 对应版本：

```bash
uv pip install "transformers==4.56.1"
```

先 clone FLUX.2 源码，并切到固定 commit：

```bash
git clone https://github.com/black-forest-labs/flux2 third_party/flux2
git -C third_party/flux2 checkout 50fe5162777813d869182b139e83b10743caef15

export FLUX2_SRC="$(pwd)/third_party/flux2"
```

下载 FLUX.2 权重。部分 FLUX.2 Hugging Face 仓库可能需要先申请访问权限。

```bash
# 默认下载 FLUX.2 klein-base-4B、autoencoder，并同时下载 9B 变体。
bash scripts/flux2/prepare_flux2_files.sh
```

如果使用默认的 4B 变体，可以设置：

```bash
export FLUX2_MODEL_PATH="${MODEL_ROOT:-$(pwd)/checkpoints}/flux2/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors"
export FLUX2_AE_MODEL_PATH="${MODEL_ROOT:-$(pwd)/checkpoints}/flux2/FLUX.2-dev/ae.safetensors"
export FLUX2_QWEN3_MODEL_SPEC=Qwen/Qwen3-4B
```

如果要使用 9B 变体，设置 `FLUX2_VARIANT=9b`，并把 `FLUX2_MODEL_PATH` 指到对应的 9B 权重。

### OmniGen2 ImageWAM

OmniGen2 ImageWAM 基于 `VectorSpaceLab/OmniGen2@18e6f9d5271b517fcb32e999f10df943ae9b8f20`，额外包含 ImageWAM 所需的一处 patch。
OmniGen2 变体对应的 `transformers` 版本是 `4.51.3`；如果你之前为 FLUX.2 切换过版本，运行 OmniGen2 前请切回该版本。

```bash
git clone https://github.com/yuyangalin/OmniGen2 third_party/OmniGen2

export OMNIGEN2_SRC="$(pwd)/third_party/OmniGen2"
export OMNIGEN2_MODEL_PATH=/path/to/OmniGen2/model 
export QWEN_MODEL_PATH=/path/to/Qwen2.5-VL-3B-Instruct
```

### Ovis-U1 ImageWAM

本仓库在 `third_party/ovis_u1_hf` 中保留了 Ovis-U1 代码。

Ovis-U1 脚本默认使用 Hugging Face 模型 ID `AIDC-AI/Ovis-U1-3B`：

```bash
export OVIS_U1_MODEL_PATH=AIDC-AI/Ovis-U1-3B
```

你也可以把 `OVIS_U1_MODEL_PATH` 设置成本地权重目录。

## 数据准备

针对LIBERO和RoboTwin, 我们使用了FastWAM提供的预处理数据集。

### LIBERO

```bash
mkdir -p data/libero_mujoco3.3.2
huggingface-cli download yuanty/LIBERO-fastwam \
  --repo-type dataset \
  --local-dir data/libero_mujoco3.3.2
```

下载压缩包后解压：

```bash
cd data/libero_mujoco3.3.2
for f in *.tar.gz; do
  tar -xzf "$f"
done
cd ../..
```

期望目录结构：

```text
data/libero_mujoco3.3.2/
├── libero_10_no_noops_lerobot/
├── libero_goal_no_noops_lerobot/
├── libero_object_no_noops_lerobot/
└── libero_spatial_no_noops_lerobot/
```

运行 LIBERO 训练或评测时设置：

```bash
export DATA_ROOT="$(pwd)/data/libero_mujoco3.3.2"
```

### RoboTwin

```bash
mkdir -p data/robotwin2.0
huggingface-cli download yuanty/robotwin2.0-fastwam \
  --repo-type dataset \
  --local-dir data/robotwin2.0
```

下载所有分卷压缩包后解压：

```bash
cd data/robotwin2.0
cat robotwin2.0.tar.gz.part-* | tar -xzf -
cd ../..
```

期望目录结构：

```text
data/robotwin2.0/
└── robotwin2.0/
    ├── data/
    ├── meta/
    └── videos/
```

运行 RoboTwin 训练或评测时设置：

```bash
export DATA_ROOT="$(pwd)/data/robotwin2.0"
export ROBOTWIN_ROOT="${DATA_ROOT}/robotwin2.0"
```

为了过滤 RoboTwin中的no-ops帧，我们使用了预先计算的过滤json，这可以通过如下json得到。

```bash
bash scripts/data/precompute_noops_lerobot.sh
```

默认会生成 `${ROBOTWIN_ROOT}/nonidle_ranges.json`。

## Benchmark 环境

### LIBERO / LIBERO-plus

只有需要做评测时才需要安装 benchmark 环境。

```bash
# LIBERO
bash scripts/setup/_install_libero_env.sh

# LIBERO-plus
bash scripts/setup/_install_libero_plus_env.sh
```

评测脚本通过下面的变量为 worker 进程激活环境（这是因为worker在独立进程中启动）：

```bash
export LIBERO_WORKER_ENV_SOURCE=/path/to/imagewam/.venv/bin/activate
```

### RoboTwin

RoboTwin 评测代码保留在 `third_party/RoboTwin` 中，但 assets 和本地模拟器依赖仍需要用户在本机准备。

```bash
bash scripts/setup/install_robotwin_env.sh
ln -sfn "$(pwd)/experiments/robotwin/imagewam_policy" "$(pwd)/third_party/RoboTwin/policy/imagewam_policy"
```

assets 准备细节请参考 `third_party/RoboTwin/README.vendor.md` 和 RoboTwin upstream 文档。

## 训练

训练 wrapper 会在 `ACTION_INIT` 不存在时自动生成 ActionDiT 初始化权重。如果需要强制重新生成，可以设置 `REBUILD_ACTION_INIT=true`。

### FLUX.2

LIBERO：

```bash
export DATA_ROOT="$(pwd)/data/libero_mujoco3.3.2"

GPU_PER_NODE=8 \
TASK_TYPE=libero \
FLUX2_VARIANT=4b \
PRECOMPUTE_QWEN3_CACHE=true \
bash scripts/flux2/run_train_flux2_klein_imagewam.sh
```

RoboTwin：

```bash
export DATA_ROOT="$(pwd)/data/robotwin2.0"
export ROBOTWIN_ROOT="${DATA_ROOT}/robotwin2.0"

GPU_PER_NODE=8 \
TASK_TYPE=robotwin \
FLUX2_VARIANT=4b \
PRECOMPUTE_QWEN3_CACHE=true \
bash scripts/flux2/run_train_flux2_klein_imagewam.sh
```

常用 FLUX.2 覆盖项：

```bash
export FLUX2_VARIANT=4b          # 4b 或 9b
export ZERO_STAGE=1              # 1/zero1 或 2/zero2
export QWEN_CACHE_DIR=/path/to/qwen3/cache # 可选，不设置时会自动生成
export ACTION_INIT=/path/to/action_dit_flux2_init.pt # 可选，不设置时自动生成
```

### OmniGen2

LIBERO：

```bash
export DATA_ROOT="$(pwd)/data/libero_mujoco3.3.2"

GPU_PER_NODE=8 \
TASK_TYPE=libero \
PRECOMPUTE_QWEN_CACHE=true \
bash scripts/omnigen2/run_train_imagewam.sh
```

RoboTwin：

```bash
export DATA_ROOT="$(pwd)/data/robotwin2.0"
export ROBOTWIN_ROOT="${DATA_ROOT}/robotwin2.0"

GPU_PER_NODE=8 \
TASK_TYPE=robotwin \
PRECOMPUTE_QWEN_CACHE=true \
bash scripts/omnigen2/run_train_imagewam.sh
```

### Ovis-U1

LIBERO：

```bash
export DATA_ROOT="$(pwd)/data/libero_mujoco3.3.2"

GPU_PER_NODE=8 \
TASK_TYPE=libero \
bash scripts/ovis_u1/run_train_ovis_u1_imagewam.sh
```

RoboTwin：

```bash
export DATA_ROOT="$(pwd)/data/robotwin2.0"
export ROBOTWIN_ROOT="${DATA_ROOT}/robotwin2.0"

GPU_PER_NODE=8 \
TASK_TYPE=robotwin \
bash scripts/ovis_u1/run_train_ovis_u1_imagewam.sh
```

## 评测

评测脚本支持直接指定 checkpoint：

```bash
export CKPT_PATH=/path/to/checkpoint.pt
export DATASET_STATS_PATH=/path/to/dataset_stats.json
```

也支持通过训练 run 目录和 step 自动推导：

```bash
export EXP_PATH=/path/to/runs/{task}/{run_id}
export EVAL_TRAIN_STEP=10000
```

设置 `EXP_PATH` 后，wrapper 会默认使用：

```text
CKPT_PATH=${EXP_PATH}/checkpoints/weights/step_${EVAL_TRAIN_STEP}.pt
DATASET_STATS_PATH=${EXP_PATH}/dataset_stats.json
```

评测使用的 GPU 数通过 `NUM_GPUS` 设置。

### FLUX.2

LIBERO：

```bash
NUM_GPUS=8 \
FLUX2_VARIANT=4b \
bash scripts/flux2/run_eval_flux2_libero.sh
```

LIBERO-plus：

```bash
NUM_GPUS=8 \
FLUX2_VARIANT=9b \
bash scripts/flux2/run_eval_flux2_libero_plus.sh
```

RoboTwin：

```bash
NUM_GPUS=8 \
FLUX2_VARIANT=4b \
bash scripts/flux2/run_eval_flux2_robotwin.sh
```

RoboTwin 评测默认开启 `EVALUATION.skip_get_obs_within_replan=true` 以加速评测。如果需要保存完整渲染视频，可以设置 `SKIP_GET_OBS_WITHIN_REPLAN=false`。

### OmniGen2

LIBERO：

```bash
NUM_GPUS=8 bash scripts/omnigen2/run_eval_omnigen2_libero.sh
```

LIBERO-plus：

```bash
NUM_GPUS=8 bash scripts/omnigen2/run_eval_omnigen2_libero_plus.sh
```

RoboTwin：

```bash
NUM_GPUS=8 bash scripts/omnigen2/run_eval_omnigen2_robotwin.sh
```

### Ovis-U1

LIBERO-plus：

```bash
NUM_GPUS=8 bash scripts/ovis_u1/run_eval_ovis_libero_plus.sh
```

## Release 权重

目前已上传以下 FLUX.2 ImageWAM checkpoint 到 Hugging Face：

- `yuyangalin/ImageWAM-FLUX.2-4B-LIBERO`
- `yuyangalin/ImageWAM-FLUX.2-4B-RoboTwin`
- `yuyangalin/ImageWAM-FLUX.2-9B-LIBERO`

其他变体的checkpoint即将放出。

```bash
mkdir -p checkpoints/imagewam_release/libero/flux2_klein_4b
huggingface-cli download yuyangalin/ImageWAM-FLUX.2-4B-LIBERO \
  --repo-type model \
  --local-dir checkpoints/imagewam_release/libero/flux2_klein_4b

mkdir -p checkpoints/imagewam_release/robotwin/flux2_klein_4b
huggingface-cli download yuyangalin/ImageWAM-FLUX.2-4B-RoboTwin \
  --repo-type model \
  --local-dir checkpoints/imagewam_release/robotwin/flux2_klein_4b

mkdir -p checkpoints/imagewam_release/libero/flux2_klein_9b
huggingface-cli download yuyangalin/ImageWAM-FLUX.2-9B-LIBERO \
  --repo-type model \
  --local-dir checkpoints/imagewam_release/libero/flux2_klein_9b
```

每个模型目录应包含 `checkpoint.pt`、`dataset_stats.json`，以及原始训练配置，通常命名为 `train_config.yaml`。

使用 release 权重评测 FLUX.2 LIBERO 的示例：

```bash
export CKPT_PATH="$(pwd)/checkpoints/imagewam_release/libero/flux2_klein_4b/checkpoint.pt"
export DATASET_STATS_PATH="$(pwd)/checkpoints/imagewam_release/libero/flux2_klein_4b/dataset_stats.json"

NUM_GPUS=8 FLUX2_VARIANT=4b bash scripts/flux2/run_eval_flux2_libero.sh
```

## 致谢

ImageWAM 基于以下多个代码库构建：
- 本仓库内置或使用的图像编辑基座：OmniGen2、FLUX.2、Ovis-U1
- 本仓库代码框架基于FastWAM，感谢他们的出色工作！
- 代码库中使用了来自 RoboTwin, LIBERO/LIBERO-Plus 的评测代码。

## 引用

如果本仓库对你的研究有帮助，欢迎引用我们的论文：

```bibtex
@misc{imagewam2026,
  title  = {ImageWAM: Do World Action Models Really Need Video Generation, or Just Image Editing?},
  author = {TODO},
  year   = {2026},
  note   = {TODO: add arXiv or conference information}
}
```

