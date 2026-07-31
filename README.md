<div align="center">
  <h1>ImageWAM</h1>
  <h3>Do World Action Models Really Need Video Generation,<br>or Just Image Editing?</h3>
  <p>
    Yuyang Zhang<sup>1,2,3*</sup>,
    Wenyao Zhang<sup>1,2,3*†</sup>,
    Zekun Qi<sup>4</sup>,
    He Zhang<sup>3</sup>,
    Haitao Lin<sup>3</sup>,
    Jingbo Zhang<sup>3</sup>,
    Yao Mu<sup>1</sup>,
    Xiaokang Yang<sup>1</sup>,
    Wenjun Zeng<sup>2</sup>,
    Xin Jin<sup>2,5✉</sup>
  </p>
  <p>
    <sup>1</sup>Shanghai Jiao Tong University &nbsp;&nbsp;
    <sup>2</sup>Eastern Institute of Technology &nbsp;&nbsp;
    <sup>3</sup>Tencent Robotics X<br>
    <sup>4</sup>Tsinghua University &nbsp;&nbsp;
    <sup>5</sup>Zhongguancun Academy
  </p>
  <p><sup>*</sup>Equal contribution &nbsp;·&nbsp; <sup>†</sup>Project lead &nbsp;·&nbsp; <sup>✉</sup>Corresponding author</p>
  <p><b>Official PyTorch implementation</b></p>
  <p>
    <a href="https://zhangwenyao1.github.io/ImageWAM/"><img src="https://img.shields.io/badge/Project-Page-blue.svg" alt="Project Page"></a>
    <a href="https://arxiv.org/abs/2606.19531"><img src="https://img.shields.io/badge/arXiv-2606.19531-b31b1b.svg?logo=arxiv" alt="arXiv"></a>
    <a href="https://huggingface.co/collections/yuyangalin/imagewam"><img src="https://img.shields.io/badge/Hugging_Face-Models-yellow.svg?logo=huggingface" alt="Hugging Face Models"></a>
    <a href="./LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License"></a>
  </p>
  <p>
    <a href="https://paperswithcode.co/benchmark/robotwin-2-0-easy-50-tasks?task=robotics&amp;eval=15812"><img src="https://paperswithcode.co/api/v1/papers/2606.19531/leaderboard-badge.svg?eval=15812&amp;live=1" alt="Papers with Code: SOTA on RoboTwin 2.0 Easy (50 Tasks)"></a>
    <a href="https://paperswithcode.co/benchmark/robotwin-2-0-hard-50-tasks?task=robotics&amp;eval=15813"><img src="https://paperswithcode.co/api/v1/papers/2606.19531/leaderboard-badge.svg?eval=15813&amp;live=1" alt="Papers with Code: SOTA on RoboTwin 2.0 Hard (50 Tasks)"></a>
  </p>
  <p><a href="./README.md">English</a> | <a href="./README_zh.md">中文</a></p>
</div>

ImageWAM is a family of world action models built on image-editing foundation models. This repository contains the training and evaluation code used in the paper experiments on LIBERO, LIBERO-plus, and RoboTwin.

We recommend starting with **FLUX.2 ImageWAM**. It provides 4B and 9B variants based on FLUX.2 [klein] 4B/9B base models, and gives the strongest performance in the series. This repository also provides training and evaluation entrypoints for **OmniGen2 ImageWAM** and **Ovis-U1 ImageWAM**. These variants are built on OmniGen2 and Ovis-U1 and also perform well. The Ovis-U1 variant is the smallest model in the series, with only a 1.1B DiT for image editing, while remaining competitive with larger variants in many settings.

## News

<!-- Add new entries at the top:
- **[YYYY-MM-DD]** Brief update.
-->

- **[2026-07-30]** We've uploaded new model checkpoint pretrained from [InternData-A1](https://huggingface.co/datasets/InternRobotics/InternData-A1), a synthetic large-scale robot dataset! Now you could fine-tune your own model from [yuyangalin/ImageWAM-FLUX.2-4B-InternData-A1-EE](https://huggingface.co/yuyangalin/ImageWAM-FLUX.2-4B-InternData-A1-EE). We've tested the pretrain-finetuning results on RoboTwin Clean2Random setting, reaching 59.0% SR, which is +40.9% vs. ImageWAM w/o pretrain and +11.5% vs. π-0.5. Check this C2R checkpoint at [yuyangalin/ImageWAM-FLUX.2-4B-RoboTwin-C2R-FT](https://huggingface.co/yuyangalin/ImageWAM-FLUX.2-4B-RoboTwin-C2R-FT)

## TODO

<!--
- [ ] Planned release or project milestone.
-->
- [x] Release RoboTwin C2R checkpoint fine-tuned from pretrained ckpt.
- [ ] Release a sample code on training and evaluating on real-world robots.
- [ ] Release results & checkpoints on more benchmarks.

All commands below are assumed to run from the repository root.

## Table Of Contents

- [News](#news)
- [TODO](#todo)
- [Repository Structure](#repository-structure)
- [Basic Installation](#basic-installation)
- [Model Preparation](#model-preparation)
- [Data Preparation](#data-preparation)
- [Benchmark Environments](#benchmark-environments)
- [Training](#training)
- [Evaluation](#evaluation)
- [Release Checkpoints](#release-checkpoints)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

## Repository Structure

```text
ImageWAM/
├── configs/                  # Model, data, task, and benchmark configs
├── docs/                     # More detailed setup, data, model, and dependency notes
├── experiments/              # LIBERO / RoboTwin evaluation managers
├── scripts/
│   ├── flux2/                # FLUX.2 ImageWAM training and evaluation entrypoints
│   ├── omnigen2/             # OmniGen2 ImageWAM training and evaluation entrypoints
│   ├── ovis_u1/              # Ovis-U1 ImageWAM training and evaluation entrypoints
│   ├── data/                 # Data processing utilities
│   └── setup/                # Benchmark environment setup helpers
├── src/imagewam/             # Core ImageWAM code
└── third_party/              # Vendored benchmark / model adapter code
```

You also need to prepare datasets locally, usually under `./data`, pretrained model weights, and generated ActionDiT initialization weights, usually under `./checkpoints`.

## Basic Installation

ImageWAM uses `uv` to manage Python dependencies. Our recommended tested environment is CUDA 11.8, Python 3.11, and PyTorch 2.7.1.

```bash
uv sync --python 3.11 --extra shared
source .venv/bin/activate
```

Copy the local configuration template:

```bash
cp .env.example .env.local
```

Shell entrypoints under `scripts/` automatically read `.env.local`. You can write local paths there, or export them directly in your shell.

Common variables:

```bash
export DATA_ROOT=/path/to/datasets # Each dataset uses its own data root.
export MODEL_ROOT=/path/to/model/checkpoints # Used to store checkpoints.
export OUTPUT_ROOT=./runs
```

The basic installation only includes shared dependencies. Source paths, model weights, and extra dependencies for each model variant are described below.

## Model Preparation

This section prepares the required external model repositories and model downloads.

### FLUX.2 ImageWAM

We first describe the recommended FLUX.2 ImageWAM setup.

If you use a FLUX.2 variant, switch `transformers` to the FLUX-compatible version:

```bash
uv pip install "transformers==4.56.1"
```

Clone the FLUX.2 source code and check out the pinned commit:

```bash
git clone https://github.com/black-forest-labs/flux2 third_party/flux2
git -C third_party/flux2 checkout 50fe5162777813d869182b139e83b10743caef15

export FLUX2_SRC="$(pwd)/third_party/flux2"
```

Download FLUX.2 weights. Some FLUX.2 Hugging Face repositories may require access approval first.

```bash
# By default, this downloads FLUX.2 klein-base-4B, the autoencoder,
# and the 9B variant.
bash scripts/flux2/prepare_flux2_files.sh
```

For the default 4B variant, set:

```bash
export FLUX2_MODEL_PATH="${MODEL_ROOT:-$(pwd)/checkpoints}/flux2/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors"
export FLUX2_AE_MODEL_PATH="${MODEL_ROOT:-$(pwd)/checkpoints}/flux2/FLUX.2-dev/ae.safetensors"
export FLUX2_QWEN3_MODEL_SPEC=Qwen/Qwen3-4B
```

To use the 9B variant, set `FLUX2_VARIANT=9b` and point `FLUX2_MODEL_PATH` to the corresponding 9B weights.

### OmniGen2 ImageWAM

OmniGen2 ImageWAM is based on `VectorSpaceLab/OmniGen2@18e6f9d5271b517fcb32e999f10df943ae9b8f20`, with an additional patch required by ImageWAM.
The OmniGen2 variant uses `transformers==4.51.3`. If you previously switched versions for FLUX.2, switch back before running OmniGen2.

```bash
git clone https://github.com/yuyangalin/OmniGen2 third_party/OmniGen2

export OMNIGEN2_SRC="$(pwd)/third_party/OmniGen2"
export OMNIGEN2_MODEL_PATH=/path/to/OmniGen2/model
export QWEN_MODEL_PATH=/path/to/Qwen2.5-VL-3B-Instruct
```

### Ovis-U1 ImageWAM

This repository keeps Ovis-U1 code under `third_party/ovis_u1_hf`.

Ovis-U1 scripts use the Hugging Face model ID `AIDC-AI/Ovis-U1-3B` by default:

```bash
export OVIS_U1_MODEL_PATH=AIDC-AI/Ovis-U1-3B
```

You can also set `OVIS_U1_MODEL_PATH` to a local weights directory.

## Data Preparation

For LIBERO and RoboTwin, we use the preprocessed datasets provided by FastWAM.

### LIBERO

```bash
mkdir -p data/libero_mujoco3.3.2
huggingface-cli download yuanty/LIBERO-fastwam \
  --repo-type dataset \
  --local-dir data/libero_mujoco3.3.2
```

After downloading the archives, extract them:

```bash
cd data/libero_mujoco3.3.2
for f in *.tar.gz; do
  tar -xzf "$f"
done
cd ../..
```

Expected directory structure:

```text
data/libero_mujoco3.3.2/
├── libero_10_no_noops_lerobot/
├── libero_goal_no_noops_lerobot/
├── libero_object_no_noops_lerobot/
└── libero_spatial_no_noops_lerobot/
```

Set this when running LIBERO training or evaluation:

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

After downloading all split archives, concatenate and extract them:

```bash
cd data/robotwin2.0
cat robotwin2.0.tar.gz.part-* | tar -xzf -
cd ../..
```

Expected directory structure:

```text
data/robotwin2.0/
└── robotwin2.0/
    ├── data/
    ├── meta/
    └── videos/
```

Set these when running RoboTwin training or evaluation:

```bash
export DATA_ROOT="$(pwd)/data/robotwin2.0"
export ROBOTWIN_ROOT="${DATA_ROOT}/robotwin2.0"
```

To filter no-op frames in RoboTwin, we use a precomputed JSON file. It can be generated with:

```bash
bash scripts/data/precompute_noops_lerobot.sh
```

By default, this generates `${ROBOTWIN_ROOT}/nonidle_ranges.json`.

## Benchmark Environments

### LIBERO / LIBERO-plus

Benchmark environments are only required for evaluation.

```bash
# LIBERO
bash scripts/setup/_install_libero_env.sh

# LIBERO-plus
bash scripts/setup/_install_libero_plus_env.sh
```

Evaluation scripts use the following variable to activate the worker environment, because workers are launched in separate processes:

```bash
export LIBERO_WORKER_ENV_SOURCE=/path/to/imagewam/.venv/bin/activate
```

### RoboTwin

RoboTwin evaluation code is kept under `third_party/RoboTwin`, but assets and local simulator dependencies still need to be prepared on your machine.

```bash
bash scripts/setup/install_robotwin_env.sh
ln -sfn "$(pwd)/experiments/robotwin/imagewam_policy" "$(pwd)/third_party/RoboTwin/policy/imagewam_policy"
```

For asset preparation details, see `third_party/RoboTwin/README.vendor.md` and the upstream RoboTwin documentation.

## Training

Training wrappers automatically generate ActionDiT initialization weights if `ACTION_INIT` does not exist. To force regeneration, set `REBUILD_ACTION_INIT=true`.

### FLUX.2

LIBERO:

```bash
export DATA_ROOT="$(pwd)/data/libero_mujoco3.3.2"

GPU_PER_NODE=8 \
TASK_TYPE=libero \
FLUX2_VARIANT=4b \
PRECOMPUTE_QWEN3_CACHE=true \
bash scripts/flux2/run_train_flux2_klein_imagewam.sh
```

RoboTwin:

```bash
export DATA_ROOT="$(pwd)/data/robotwin2.0"
export ROBOTWIN_ROOT="${DATA_ROOT}/robotwin2.0"

GPU_PER_NODE=8 \
TASK_TYPE=robotwin \
FLUX2_VARIANT=4b \
PRECOMPUTE_QWEN3_CACHE=true \
bash scripts/flux2/run_train_flux2_klein_imagewam.sh
```

RoboTwin fine-tuning from the released InternData-A1 16D pretrain weight:

```bash
export PRETRAIN_CHECKPOINT=/path/to/imagewam-interndata-a1-ee/checkpoint.pt
export ROBOTWIN_ROOT="$(pwd)/data/robotwin2.0/robotwin2.0"

GPU_PER_NODE=8 \
USE_CLEAN_ROBOTWIN=true \
PRECOMPUTE_QWEN3_CACHE=true \
bash scripts/flux2/run_finetune_flux2_klein_robotwin_ee16.sh
```

Here, `ee16` names the 16D pretrain checkpoint layout. RoboTwin's original
14D joint/action representation is padded with two masked dimensions; it is
not converted into end-effector poses.

Common FLUX.2 overrides:

```bash
export FLUX2_VARIANT=4b          # 4b or 9b
export ZERO_STAGE=1              # 1/zero1 or 2/zero2
export QWEN_CACHE_DIR=/path/to/qwen3/cache # Optional; generated automatically if unset.
export ACTION_INIT=/path/to/action_dit_flux2_init.pt # Optional; generated automatically if unset.
```

### OmniGen2

LIBERO:

```bash
export DATA_ROOT="$(pwd)/data/libero_mujoco3.3.2"

GPU_PER_NODE=8 \
TASK_TYPE=libero \
PRECOMPUTE_QWEN_CACHE=true \
bash scripts/omnigen2/run_train_imagewam.sh
```

RoboTwin:

```bash
export DATA_ROOT="$(pwd)/data/robotwin2.0"
export ROBOTWIN_ROOT="${DATA_ROOT}/robotwin2.0"

GPU_PER_NODE=8 \
TASK_TYPE=robotwin \
PRECOMPUTE_QWEN_CACHE=true \
bash scripts/omnigen2/run_train_imagewam.sh
```

### Ovis-U1

LIBERO:

```bash
export DATA_ROOT="$(pwd)/data/libero_mujoco3.3.2"

GPU_PER_NODE=8 \
TASK_TYPE=libero \
bash scripts/ovis_u1/run_train_ovis_u1_imagewam.sh
```

RoboTwin:

```bash
export DATA_ROOT="$(pwd)/data/robotwin2.0"
export ROBOTWIN_ROOT="${DATA_ROOT}/robotwin2.0"

GPU_PER_NODE=8 \
TASK_TYPE=robotwin \
bash scripts/ovis_u1/run_train_ovis_u1_imagewam.sh
```

## Evaluation

Evaluation scripts support directly specifying a checkpoint:

```bash
export CKPT_PATH=/path/to/model.pt
export DATASET_STATS_PATH=/path/to/dataset_stats.json
```

They also support deriving paths from a training run directory and step:

```bash
export EXP_PATH=/path/to/runs/{task}/{run_id}
export EVAL_TRAIN_STEP=10000
```

After setting `EXP_PATH`, the wrappers use:

```text
CKPT_PATH=${EXP_PATH}/checkpoints/weights/step_${EVAL_TRAIN_STEP}.pt
DATASET_STATS_PATH=${EXP_PATH}/dataset_stats.json
```

Set the number of GPUs for evaluation with `NUM_GPUS`.

### FLUX.2

LIBERO:

```bash
NUM_GPUS=8 \
FLUX2_VARIANT=4b \
bash scripts/flux2/run_eval_flux2_libero.sh
```

LIBERO-plus:

```bash
NUM_GPUS=8 \
FLUX2_VARIANT=9b \
bash scripts/flux2/run_eval_flux2_libero_plus.sh
```

RoboTwin:

```bash
NUM_GPUS=8 \
FLUX2_VARIANT=4b \
bash scripts/flux2/run_eval_flux2_robotwin.sh
```

RoboTwin checkpoints fine-tuned from the InternData-A1 16D pretrain weight:

```bash
export CKPT_PATH=/path/to/robotwin-ee16/checkpoint.pt
export DATASET_STATS_PATH=/path/to/robotwin-ee16/dataset_stats.json

NUM_GPUS=8 bash scripts/flux2/run_eval_flux2_robotwin_ee16.sh
```

RoboTwin evaluation enables `EVALUATION.skip_get_obs_within_replan=true` by default to speed up evaluation. If you need to save fully rendered videos, set `SKIP_GET_OBS_WITHIN_REPLAN=false`.

### OmniGen2

LIBERO:

```bash
NUM_GPUS=8 bash scripts/omnigen2/run_eval_omnigen2_libero.sh
```

LIBERO-plus:

```bash
NUM_GPUS=8 bash scripts/omnigen2/run_eval_omnigen2_libero_plus.sh
```

RoboTwin:

```bash
NUM_GPUS=8 bash scripts/omnigen2/run_eval_omnigen2_robotwin.sh
```

### Ovis-U1

LIBERO-plus:

```bash
NUM_GPUS=8 bash scripts/ovis_u1/run_eval_ovis_libero_plus.sh
```

## Release Checkpoints

The following FLUX.2 ImageWAM checkpoints are available on Hugging Face:

- `yuyangalin/ImageWAM-FLUX.2-4B-LIBERO`
- `yuyangalin/ImageWAM-FLUX.2-4B-RoboTwin`
- `yuyangalin/ImageWAM-FLUX.2-9B-LIBERO`

Checkpoints of other variants will be released later. Stay focused!

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

Each model directory is expected to contain `model.pt`, `dataset_stats.json`, and the original training config, usually `train_config.yaml`.

Example: evaluate the released FLUX.2 LIBERO checkpoint:

```bash
export CKPT_PATH="$(pwd)/checkpoints/imagewam_release/libero/flux2_klein_4b/model.pt"
export DATASET_STATS_PATH="$(pwd)/checkpoints/imagewam_release/libero/flux2_klein_4b/dataset_stats.json"

NUM_GPUS=8 FLUX2_VARIANT=4b bash scripts/flux2/run_eval_flux2_libero.sh
```

## Acknowledgements

ImageWAM is built on several codebases:

- Image-editing backbones built in or used by this repository: OmniGen2, FLUX.2, and Ovis-U1.
- This repository's code framework is based on FastWAM. We thank the authors for their excellent work.
- This codebase uses evaluation code from RoboTwin and LIBERO/LIBERO-plus.

## Citation

If you find this repository helpful for your research, please cite our paper:

```bibtex
@misc{zhang2026imagewam,
      title={ImageWAM: Do World Action Models Really Need Video Generation, or Just Image Editing?}, 
      author={Yuyang Zhang and Wenyao Zhang and Zekun Qi and He Zhang and Haitao Lin and Jingbo Zhang and Yao Mu and Xiaokang Yang and Wenjun Zeng and Xin Jin},
      year={2026},
      eprint={2606.19531},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2606.19531}, 
}
```

