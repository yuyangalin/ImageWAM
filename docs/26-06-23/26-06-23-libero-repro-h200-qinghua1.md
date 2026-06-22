# h200-qinghua-1 LIBERO 复现记录

## 目的

在当前 ImageWAM 仓库完成 LIBERO 复现闭环：复用 `h200-qinghua-1` 上已有 LIBERO 数据、Wan2.2 权重、FastWAM release checkpoint 和 LIBERO / LIBERO-plus benchmark 环境，打通训练、标准 LIBERO 评测和 LIBERO-plus 评测，并记录与原 repository 预期效果的差距。

## 分支与提交

- 本地分支：`main`
- 起始提交：`00a4e7a Revise paper citation in README.md`
- 本记录对应改动：随本次 conventional commit 一并提交，最终 hash 以 `git log --oneline -1` 为准
- 已有未提交用户改动：`.gitignore` 中新增 `docs/` 和 `AGENTS.md` 忽略规则，本次未覆盖、未加入提交

## 运行位置

- 训练/评测机器：`h200-qinghua-1`
- 主机名：`lacy--214-30-239-40`
- 项目路径：`/home/maxliu/projects/ImageWAM`
- 跳板机：`h200-qinghua-jump`
- 跳板机映射盘：`/data-214-30-239-40`

## 环境与资产

- 复用 conda 环境：`/data/home/frank/.conda/envs/fastwam`
- Python：`3.12.11`
- 关键包：`torch 2.7.1+cu128`、`torchvision 0.22.1+cu128`、`accelerate 1.12.0`、`deepspeed 0.18.5`、`datasets 3.6.0`、`hydra-core 1.3.2`、`mujoco 3.3.2`、`pyarrow 23.0.0`、`av 16.0.1`
- LIBERO FastWAM 预处理数据：`/data/home/frank/projects/FastWAM/data/libero_mujoco3.3.2`
  - `libero_spatial_no_noops_lerobot`
  - `libero_object_no_noops_lerobot`
  - `libero_goal_no_noops_lerobot`
  - `libero_10_no_noops_lerobot`
- LIBERO 文本 embedding cache：`/data/home/frank/projects/FastWAM/data/text_embeds_cache/libero`
- Wan2.2 权重：`/data/home/frank/projects/FastWAM/checkpoints/Wan-AI/Wan2.2-TI2V-5B`
- Wan2.1 tokenizer：`/data/home/frank/projects/FastWAM/checkpoints/Wan-AI/Wan2.1-T2V-1.3B`
- ActionDiT 初始化权重：`/data/home/frank/projects/FastWAM/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt`
- FastWAM release checkpoint：`/data/home/frank/projects/FastWAM/checkpoints/fastwam_release/libero_uncond_2cam224.pt`
- FastWAM release stats：`/data/home/frank/projects/FastWAM/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json`
- 标准 LIBERO benchmark：`/data/home/frank/projects/LIBERO`
- LIBERO-plus benchmark：`/data/home/frank/projects/LIBERO-plus`

## 代码与脚本修改

- 新增 `scripts/h200_qinghua1_libero_wan.sh`
  - 支持 `train-smoke`、`train`、`eval-libero`、`eval-libero-plus`。
  - 固定 h200-qinghua-1 的真实数据、权重、text cache、LIBERO/LIBERO-plus 路径，并允许通过环境变量覆盖。
  - 默认使用 Wan2.2 / `libero_uncond_2cam224_1e-4` 路线，设置 `model.redirect_common_files=false`，因为服务器上 real Wan 文件在 `Wan-AI/Wan2.2-TI2V-5B`，converted safetensors 目录不完整。
  - 使用 `+data.train.lerobot_v3_video_backend=pyav` 避开 torchcodec 缺失 `libnppicc.so.12` 的问题。
  - 评测时自动写入 `~/.libero/config.yaml`，在标准 LIBERO 与 LIBERO-plus 之间切换 `benchmark_root`。
  - 写入 `.remote/sitecustomize.py`，用 `logging.FileHandler` 子类将 robosuite 的 `/tmp/robosuite.log` 重定向到 `${TMPDIR}/robosuite.log`。
  - 默认设置 `MUJOCO_GL=egl` 和 `PYOPENGL_PLATFORM=egl`；OSMesa 在该环境中导入失败。
- 修改 `src/imagewam/runtime.py`
  - 写入 `output_dir/config.yaml` 前创建输出目录，避免新 run 目录不存在。
- 修改 `src/imagewam/datasets/lerobot/lerobot/lerobot_dataset.py`
  - `dataset.episodes is None` 时从 `dataset.meta.episodes` 或本地 episode id fallback，修复统计阶段 `NoneType` 下标错误。
- 修改 `src/imagewam/datasets/lerobot/base_lerobot_dataset.py`
  - v2 LeRobot 数据集也透传 `lerobot_v3_video_backend`，使 h200 脚本能够强制 v2 路径使用 `pyav`。
- 修改 `src/imagewam/trainer.py`
  - 当 `save_every` 与 `max_steps` 同步命中时避免最终 checkpoint 重复保存。
- 修改 `experiments/libero/run_libero_parallel_test.sh`
  - 内部调度 tmux 使用 `env -u LD_LIBRARY_PATH /usr/bin/tmux`，避开 conda `libtinfo.so.6` 导致的 `tiparm_s` 符号错误。

## 本地轻量检查

```bash
bash -n scripts/h200_qinghua1_libero_wan.sh
bash -n experiments/libero/run_libero_parallel_test.sh
python -m py_compile src/imagewam/runtime.py
python -m py_compile src/imagewam/datasets/lerobot/base_lerobot_dataset.py
python -m py_compile src/imagewam/datasets/lerobot/lerobot/lerobot_dataset.py
python -m py_compile src/imagewam/trainer.py
```

结果：均通过。本地未运行 GPU 训练或大规模推理。

## 远端训练 smoke

最终成功命令：

```bash
cd /home/maxliu/projects/ImageWAM
RUN_ID=h200q1_wan_libero_smoke_20260623_021519 \
GPU_PER_NODE=1 \
bash scripts/h200_qinghua1_libero_wan.sh train-smoke \
  > run_logs/train_smoke_20260623_021519.log 2>&1
```

结果：

- 日志：`run_logs/train_smoke_20260623_021519.log`
- run 目录：`runs/libero_uncond_2cam224_1e-4/h200q1_wan_libero_smoke_20260623_021519`
- checkpoint：`runs/libero_uncond_2cam224_1e-4/h200q1_wan_libero_smoke_20260623_021519/checkpoints/weights/step_000001.pt`
- stats：`runs/libero_uncond_2cam224_1e-4/h200q1_wan_libero_smoke_20260623_021519/dataset_stats.json`
- 指标：`loss=0.6649`、`loss_action=0.6504`、`loss_video=0.0145`
- 速度：约 `0.46 step/s`
- 结论：数据读取、stats 生成、Wan2.2/ActionDiT 初始化、forward/backward、checkpoint 写入均跑通。

## 远端代表性短训练

为避免只以 1 step smoke 作为训练证明，补跑一组真实 LIBERO 数据上的 20 step 短训练，仍使用 1 GPU、小 batch，验证训练主循环在多个 logging / checkpoint 周期内稳定。

命令：

```bash
cd /home/maxliu/projects/ImageWAM
RUN_ID=h200q1_wan_libero_train20_20260623_075931 \
GPU_PER_NODE=1 \
MAX_STEPS=20 \
SAVE_EVERY=20 \
LOG_EVERY=5 \
EVAL_EVERY=1000000 \
BATCH_SIZE=1 \
NUM_WORKERS=2 \
bash scripts/h200_qinghua1_libero_wan.sh train-smoke \
  > run_logs/train20_20260623_075931.log 2>&1
```

结果：

- tmux session：`imagewam_libero_train20_20260623_075931`
- 日志：`run_logs/train20_20260623_075931.log`
- run 目录：`runs/libero_uncond_2cam224_1e-4/h200q1_wan_libero_train20_20260623_075931`
- checkpoint：`runs/libero_uncond_2cam224_1e-4/h200q1_wan_libero_train20_20260623_075931/checkpoints/weights/step_000020.pt`
- state：`runs/libero_uncond_2cam224_1e-4/h200q1_wan_libero_train20_20260623_075931/checkpoints/state/step_000020`
- checkpoint 大小：约 `12.0G`
- 关键日志：
  - step 5：`loss=5.0528`、`loss_action=3.9721`、`loss_video=1.0807`
  - step 10：`loss=2.0225`、`loss_action=1.0209`、`loss_video=1.0016`
  - step 15：`loss=3.1690`、`loss_action=2.5443`、`loss_video=0.6247`
  - step 20：`loss=1.5375`、`loss_action=0.6072`、`loss_video=0.9303`
- step 20 速度：约 `1.79 step/s`
- 结论：训练主循环、日志、DeepSpeed ZeRO-1 state 保存和最终 weights 保存均完成；训练结束后 8 张 H200 GPU 均已释放。

历史排查摘要：

- `014417`：首次同步误排除仓库内 `configs/data`，Hydra 找不到 `data/libero_2cam`。
- `014802`：Wan VAE 路径检测错误，改为 `model.redirect_common_files=false`。
- `015102`：`dataset.episodes` 为 `None`，补 fallback。
- `015521`：torchcodec 缺 `libnppicc.so.12`，切到 `pyav`。
- `020221`：Hydra struct 需要用 `+data.train.lerobot_v3_video_backend=pyav`。
- `020336`：v2 数据集未透传 backend，补 `BaseLerobotDataset`。
- `020746`：1 step 可跑通，但最终保存重复；随后修复 trainer。

## 标准 LIBERO release checkpoint 评测

命令：

```bash
cd /home/maxliu/projects/ImageWAM
NUM_GPUS=1 MAX_TASKS_PER_GPU=1 NUM_TRIALS=1 TASK_SAMPLE_RATIO=0.05 \
bash scripts/h200_qinghua1_libero_wan.sh eval-libero \
  /data/home/frank/projects/FastWAM/checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  /data/home/frank/projects/FastWAM/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  EVALUATION.output_dir=./evaluate_results/libero_smoke/release_20260623_022741 \
  > run_logs/eval_libero_release_20260623_022741.log 2>&1
```

结果：

- 日志：`run_logs/eval_libero_release_20260623_022741.log`
- 输出目录：`evaluate_results/libero_smoke/libero_uncond_2cam224/release_20260623_022741`
- summary：`evaluate_results/libero_smoke/libero_uncond_2cam224/release_20260623_022741/summary.json`
- 任务：4 个 suite 各采样 1 个任务，`libero_10_6`、`libero_goal_2`、`libero_spatial_8`、`libero_object_3`
- 成功率：4/4，overall `100.00%`
- 视频示例：`evaluate_results/libero_smoke/libero_uncond_2cam224/release_20260623_022741/libero_10/videos/2026_06_23-02_27_45--episode=task6_trial0--success=True--task=put_the_white_mug_on_the_plate_and_put_the_chocola.mp4`

标准 LIBERO 评测历史排查：

- `022209`：`sitecustomize` 将 `logging.FileHandler` 替换为函数，破坏 `logging.handlers`，改为子类。
- `022319`：内部 tmux 被 conda `libtinfo.so.6` 污染，补 `env -u LD_LIBRARY_PATH /usr/bin/tmux`。
- `022500`：OSMesa 导入失败，改用 EGL。
- EGL 清理阶段会打印 `EGL_NOT_INITIALIZED` 析构警告和 `libGLU.so.0` warning，但 worker 退出码为 0，summary 正常生成。

## LIBERO-plus release checkpoint 评测

误开的采样：

- `TASK_SAMPLE_RATIO=0.05` 会在 LIBERO-plus 中抽到约 503 个任务，因为 plus 每个 suite 有数千任务。
- 已停止会话：`imagewam_libero_plus_eval_20260623_024054` 和内部 `libero_test_v3_release_20260623_024054_2757590`。
- 结论：plus 小样本 smoke 使用 `TASK_SAMPLE_RATIO=0.0001`，每个 suite 至少 1 个任务即可验证链路。

最终命令：

```bash
cd /home/maxliu/projects/ImageWAM
NUM_GPUS=1 MAX_TASKS_PER_GPU=1 NUM_TRIALS=1 TASK_SAMPLE_RATIO=0.0001 \
bash scripts/h200_qinghua1_libero_wan.sh eval-libero-plus \
  /data/home/frank/projects/FastWAM/checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  /data/home/frank/projects/FastWAM/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  EVALUATION.output_dir=./evaluate_results/libero_plus_smoke/release_20260623_023917 \
  > run_logs/eval_libero_plus_release_20260623_023917.log 2>&1
```

结果：

- 日志：`run_logs/eval_libero_plus_release_20260623_023917.log`
- 输出目录：`evaluate_results/libero_plus_smoke/libero_uncond_2cam224/release_20260623_023917`
- summary：`evaluate_results/libero_plus_smoke/libero_uncond_2cam224/release_20260623_023917/summary.json`
- CSV：`evaluate_results/libero_plus_smoke/libero_uncond_2cam224/release_20260623_023917/summary.csv`
- task CSV：`evaluate_results/libero_plus_smoke/libero_uncond_2cam224/release_20260623_023917/task_success_rates.csv`
- 总任务：4 个 suite 各采样 1 个任务
- overall：`25.00%`
- per suite：
  - `libero_spatial_2298`：`100.00%`，任务 `pick up the black bowl on the cookie box and place it on the plate light 25`
  - `libero_object_887`：`0.00%`，任务 `pick up the butter and place it in the basket view 0 0 170 0 0 initstate 0`
  - `libero_goal_524`：`0.00%`，任务 `push the plate to the front of the stove view 0 0 100 0 0 initstate 446`
  - `libero_10_1598`：`0.00%`，任务 `turn on the stove and put the moka pot on it view 0 0 100 0 0 initstate 0 noise 36`
- 总耗时：约 `08m19s`

结论：LIBERO-plus benchmark、路径切换、模型加载、rollout 和视频保存均跑通；但使用标准 LIBERO release checkpoint 直接迁移到 plus 分布时，小样本成功率只有 25%，不能视为 plus 任务效果已接近原 repository 的充分复现。

## 远端可复跑命令

进入项目和环境：

```bash
ssh h200-qinghua-1
cd /home/maxliu/projects/ImageWAM
source "$(/data/home/frank/.conda/bin/conda info --base)/etc/profile.d/conda.sh"
conda activate fastwam || conda activate /data/home/frank/.conda/envs/fastwam
git fetch origin
git checkout main
git pull origin main
```

训练 smoke：

```bash
RUN_ID=h200q1_wan_libero_smoke_$(date +%Y%m%d_%H%M%S) \
GPU_PER_NODE=1 \
bash scripts/h200_qinghua1_libero_wan.sh train-smoke
```

预期输出：

- 日志中出现 `Loaded Wan2.2-TI2V-5B components`
- 进入 `Train ...` progress
- 1 step 后出现 `[done] max_steps reached step=1`
- run 目录下生成 `dataset_stats.json` 和 `checkpoints/weights/step_000001.pt`

标准 LIBERO 小样本评测：

```bash
NUM_GPUS=1 MAX_TASKS_PER_GPU=1 NUM_TRIALS=1 TASK_SAMPLE_RATIO=0.05 \
bash scripts/h200_qinghua1_libero_wan.sh eval-libero \
  /data/home/frank/projects/FastWAM/checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  /data/home/frank/projects/FastWAM/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  EVALUATION.output_dir=./evaluate_results/libero_smoke/release_$(date +%Y%m%d_%H%M%S)
```

预期输出：

- `All tasks completed successfully!`
- `summary.json`、`summary.csv`、`task_success_rates.csv`
- 本次实测 4/4，overall `100.00%`

LIBERO-plus 小样本评测：

```bash
NUM_GPUS=1 MAX_TASKS_PER_GPU=1 NUM_TRIALS=1 TASK_SAMPLE_RATIO=0.0001 \
bash scripts/h200_qinghua1_libero_wan.sh eval-libero-plus \
  /data/home/frank/projects/FastWAM/checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  /data/home/frank/projects/FastWAM/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  EVALUATION.output_dir=./evaluate_results/libero_plus_smoke/release_$(date +%Y%m%d_%H%M%S)
```

预期输出：

- `All tasks completed successfully!`
- 每个 suite 抽 1 个任务，总计 4 个任务
- 本次实测 overall `25.00%`，说明 plus 链路可用但 release checkpoint 对 plus 分布不稳

## 差距与下一步

- 当前闭环复用了服务器已有 Wan2.2 / FastWAM release checkpoint；未在 h200-qinghua-1 找到 README 推荐的 FLUX.2、Omni、Ovis 相关源码和权重，因此未复现 FLUX.2 ImageWAM 路线。
- 标准 LIBERO 小样本达到 100%，能作为接近原 FastWAM release 行为的 sanity check。
- LIBERO-plus 小样本只有 25%，需要使用 plus 数据重新训练或补齐原 repository 对 plus 的专用 checkpoint 后再评估。
- 若要推进到更强结论，建议远端启动多卡代表性训练，例如 `GPU_PER_NODE=8 bash scripts/h200_qinghua1_libero_wan.sh train max_steps=... save_every=...`，训练后用标准 LIBERO 和 plus 两套命令评测同一个 checkpoint。
