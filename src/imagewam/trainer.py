import logging
import json
import inspect
import os
import re
import shutil
from math import ceil
from pathlib import Path
import time

import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import DictConfig
from PIL import Image
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from .utils.fs import ensure_dir
from .utils.logging_config import get_logger, setup_logging
from .utils.pytorch_utils import set_global_seed
from .utils.samplers import ResumableEpochSampler
from .utils.video_io import save_mp4
from .utils.video_metrics import pil_frames_to_video_tensor, video_psnr, video_ssim

logger = get_logger(__name__)


class Wan22Trainer:
    def __init__(self, model, train_dataset, val_dataset=None, *, cfg: DictConfig):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = cfg
        self.output_dir = str(cfg.output_dir)
        self.learning_rate = float(cfg.learning_rate)
        self.weight_decay = float(cfg.weight_decay)
        self.batch_size = int(cfg.batch_size)
        self.num_workers = int(cfg.num_workers)
        # Optional dataloader knobs (default: PyTorch defaults)
        self.prefetch_factor = (
            int(cfg.prefetch_factor) if getattr(cfg, "prefetch_factor", None) is not None else None
        )
        self.persistent_workers = bool(getattr(cfg, "persistent_workers", False))
        self.num_epochs = int(cfg.num_epochs)
        max_steps = cfg.max_steps
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.log_every = int(cfg.log_every)
        self.save_every = int(cfg.save_every)
        self.keep_latest_state_only = bool(getattr(cfg, "keep_latest_state_only", False))
        self.eval_every = int(cfg.eval_every)
        self.eval_num_inference_steps = int(cfg.eval_num_inference_steps)
        self.eval_num_samples = max(int(getattr(cfg, "eval_num_samples", 1)), 1)
        self.rank_timer_every = int(getattr(cfg, "rank_timer_every", 0) or 0)
        self.rank_timer_sync_cuda = bool(getattr(cfg, "rank_timer_sync_cuda", True))
        self.gradient_accumulation_steps = int(cfg.gradient_accumulation_steps)
        self.max_grad_norm = float(cfg.max_grad_norm)
        self.seed = int(cfg.seed)
        
        self.resume = cfg.resume
        self._weights_resume_loaded_before_prepare = False
        self.mixed_precision = str(cfg.mixed_precision).strip().lower()
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError(
                f"Unsupported mixed_precision: {cfg.mixed_precision}. "
                "Expected one of: ['no', 'fp16', 'bf16']."
            )
        self.wandb_enabled = bool(cfg.wandb.enabled)

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            step_scheduler_with_optimizer=False,
        )
        
        logger.info(
            "Accelerate training: distributed_type=%s zero_stage=%s world_size=%d process_index=%d cfg_mixed_precision=%s accelerator_mixed_precision=%s grad_accum=%d grad_clip=%.4f",
            self.accelerator.distributed_type,
            self.accelerator.state.deepspeed_plugin.deepspeed_config.get("zero_optimization", {}).get("stage", "unknown"),
            self.accelerator.num_processes,
            self.accelerator.process_index,
            self.mixed_precision,
            self.accelerator.mixed_precision,
            self.gradient_accumulation_steps,
            self.max_grad_norm,
        )
        logger.info("using accelerator.device=%s", self.accelerator.device)
        if self.accelerator.is_main_process:
            logger.info(
                "[mem-init] main pid=%d IMAGEWAM_MEM_TRIM_EVERY=%s "
                "IMAGEWAM_M_TRIM_THRESHOLD=%s IMAGEWAM_M_MMAP_THRESHOLD=%s "
                "IMAGEWAM_M_TOP_PAD=%s MALLOC_ARENA_MAX=%s",
                os.getpid(),
                os.environ.get("IMAGEWAM_MEM_TRIM_EVERY", "<unset>"),
                os.environ.get("IMAGEWAM_M_TRIM_THRESHOLD", "<unset>"),
                os.environ.get("IMAGEWAM_M_MMAP_THRESHOLD", "<unset>"),
                os.environ.get("IMAGEWAM_M_TOP_PAD", "<unset>"),
                os.environ.get("MALLOC_ARENA_MAX", "<unset>"),
            )
        worker_init_fn = set_global_seed(self.seed, get_worker_init_fn=True)
        self._assert_dataset_length_consistent(self.train_dataset, "train_dataset")
        if self.val_dataset is not None:
            self._assert_dataset_length_consistent(self.val_dataset, "val_dataset")

        # Freeze non-trainable modules before optimizer/deepspeed initialization.
        # This keeps DiT (+ optional proprio encoder) as trainable when ZeRO builds optimizer state.
        self._apply_dit_only_train_mode(self.model)
        self._load_weights_checkpoint_before_prepare()
        trainable_params = [param for param in self.model.dit.parameters() if param.requires_grad]
        proprio_encoder = getattr(self.model, "proprio_encoder", None)
        if proprio_encoder is not None:
            trainable_params.extend(param for param in proprio_encoder.parameters() if param.requires_grad)
        if not trainable_params:
            raise ValueError("No trainable parameters found after applying trainable policy.")
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )
        
        self.train_loader = self._build_loader(self.train_dataset, worker_init_fn=worker_init_fn)
        total_train_steps = self._estimate_total_train_steps()
        self.max_steps = total_train_steps
        warmup_steps_cfg = getattr(cfg, "warmup_steps", None)
        warmup_steps = int(total_train_steps * 0.05) if warmup_steps_cfg is None else int(warmup_steps_cfg)
        logger.info(
            "Scheduler setup: type=%s total_train_steps=%d warmup_steps=%d",
            cfg.lr_scheduler_type,
            total_train_steps,
            warmup_steps,
        )
        self.scheduler = self._build_scheduler(
            scheduler_type=cfg.lr_scheduler_type,
            total_train_steps=total_train_steps,
            warmup_steps=warmup_steps,
        )
        self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0

        self.checkpoint_root = os.path.join(self.output_dir, "checkpoints")
        self.weights_dir = os.path.join(self.checkpoint_root, "weights")
        self.state_dir = os.path.join(self.checkpoint_root, "state")
        self.eval_dir = os.path.join(self.output_dir, "eval")

        ensure_dir(self.output_dir)
        ensure_dir(self.checkpoint_root)
        ensure_dir(self.weights_dir)
        ensure_dir(self.state_dir)
        ensure_dir(self.eval_dir)

        self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.scheduler
        )
        self.optimizer.zero_grad(set_to_none=True)
        self.wandb_run = None
        self._init_wandb()
        self._resume_or_load_checkpoint()

        val_size = len(self.val_dataset) if self.val_dataset is not None else len(self.train_dataset)
        logger.info("Train/val dataset size: %d/%d", len(self.train_dataset), val_size)

    def _init_wandb(self):
        if not self.wandb_enabled or not self.accelerator.is_main_process:
            return
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb logging is enabled in config (`wandb.enabled=true`) but wandb is not installed."
            ) from e

        self.wandb_run = wandb.init(
            entity=self.cfg.wandb.workspace,
            project=self.cfg.wandb.project,
            name=self.cfg.wandb.name,
            group=None if self.cfg.wandb.group in (None, "null", "") else str(self.cfg.wandb.group),
            mode=self.cfg.wandb.mode,
            dir=self.output_dir,
        )
        logger.info(
            "Initialized wandb run: workspace=%s project=%s name=%s",
            self.cfg.wandb.workspace,
            self.cfg.wandb.project,
            self.cfg.wandb.name,
        )

    def _wandb_log(self, payload: dict):
        if self.wandb_run is None:
            return
        self.wandb_run.log(payload, step=self.global_step)

    def _finish_wandb(self):
        if self.wandb_run is None:
            return
        self.wandb_run.finish()
        self.wandb_run = None

    def _build_loader(self, dataset, worker_init_fn=None):
        self.train_sampler = ResumableEpochSampler(
            dataset=dataset,
            seed=self.seed,
            batch_size=self.batch_size,
            num_processes=self.accelerator.num_processes,
        )
        loader_kwargs = dict(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=worker_init_fn,
        )
        if self.num_workers > 0:
            if self.prefetch_factor is not None:
                loader_kwargs["prefetch_factor"] = self.prefetch_factor
            # persistent_workers avoids re-forking 24 workers every epoch boundary
            # (which is both slow and visually looks like a memory leak as RSS
            # ramps from ~0 again on each restart).
            loader_kwargs["persistent_workers"] = self.persistent_workers
        if self.accelerator.is_main_process:
            logger.info(
                "[loader] num_workers=%d batch_size=%d pin_memory=%s "
                "prefetch_factor=%s persistent_workers=%s",
                self.num_workers,
                self.batch_size,
                loader_kwargs["pin_memory"],
                loader_kwargs.get("prefetch_factor", "<default=2>"),
                loader_kwargs.get("persistent_workers", False),
            )
        return DataLoader(**loader_kwargs)

    def _assert_dataset_length_consistent(self, dataset, dataset_name: str):
        if not hasattr(dataset, "__len__"):
            raise TypeError(f"`{dataset_name}` must implement __len__ for rank consistency checks.")

        local_length = len(dataset)
        gathered_lengths = self.accelerator.gather(
            torch.tensor([local_length], device=self.accelerator.device, dtype=torch.int64)
        ).reshape(-1)
        if torch.all(gathered_lengths == gathered_lengths[0]):
            return

        if self.accelerator.is_main_process:
            print(f"[dataset-check] {dataset_name} length mismatch across ranks after initialization:")
            for rank, rank_length in enumerate(gathered_lengths.cpu().tolist()):
                print(f"rank {rank}: {rank_length}")
        self.accelerator.wait_for_everyone()
        raise RuntimeError(
            f"{dataset_name} length mismatch across ranks: {gathered_lengths.cpu().tolist()}"
        )

    def _estimate_total_train_steps(self) -> int:
        if self.max_steps is not None:
            return max(int(self.max_steps), 1)

        if not hasattr(self.train_dataset, "__len__"):
            raise TypeError("`train_dataset` must implement __len__ when `max_steps` is None.")

        num_processes = max(int(self.accelerator.num_processes), 1)
        global_batch_size = max(self.batch_size * num_processes, 1)
        micro_steps_per_epoch = max(ceil(len(self.train_dataset) / global_batch_size), 1)
        opt_steps_per_epoch = max(
            ceil(micro_steps_per_epoch / self.gradient_accumulation_steps),
            1,
        )
        return max(opt_steps_per_epoch * self.num_epochs, 1)

    def _build_scheduler(self, scheduler_type, total_train_steps: int, warmup_steps: int = 0):
        scheduler_type = str(scheduler_type).strip().lower()
        total_train_steps = max(int(total_train_steps), 1)
        warmup_steps = min(max(int(warmup_steps), 0), total_train_steps - 1)

        remaining_steps = max(total_train_steps - warmup_steps, 1)
        if scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=remaining_steps,
                eta_min=self.learning_rate * 0.01,
            )
        elif scheduler_type == "constant":
            main_scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=remaining_steps)
        else:
            raise ValueError(
                f"Unsupported lr_scheduler_type: {scheduler_type}. "
                "Expected one of: ['cosine', 'constant']."
            )

        if warmup_steps <= 0:
            return main_scheduler

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0 / warmup_steps,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )
    
    def _estimate_eta(self):
        elapsed = max(time.perf_counter() - self.run_start_time, 1e-6)
        done_steps = max(self.global_step - self.run_start_step, 1)
        steps_per_sec = done_steps / elapsed
        remaining_steps = max(self.max_steps - self.global_step, 0)
        eta_seconds = int(remaining_steps / max(steps_per_sec, 1e-9))
        eta_h, eta_rem = divmod(eta_seconds, 3600)
        eta_m, eta_s = divmod(eta_rem, 60)
        return f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}", steps_per_sec

    def _rank_timer_sync(self, enabled: bool):
        if not enabled or not self.rank_timer_sync_cuda:
            return
        if self.accelerator.device.type == "cuda":
            torch.cuda.synchronize(self.accelerator.device)

    def _gather_rank_timer_stats(self, timings: dict[str, float], device: torch.device):
        names = ("data", "forward", "backward", "optimizer", "metrics", "total")
        local = torch.tensor(
            [float(timings.get(name, 0.0)) for name in names],
            device=device,
            dtype=torch.float32,
        )
        gathered = self.accelerator.gather(local).reshape(self.accelerator.num_processes, len(names))
        return names, gathered.detach().cpu()

    @staticmethod
    def _format_rank_timer_stats(names, stats: torch.Tensor) -> str:
        parts = []
        for idx, name in enumerate(names):
            values = stats[:, idx]
            max_rank = int(torch.argmax(values).item())
            parts.append(
                f"{name}=mean:{values.mean().item():.3f}s "
                f"min:{values.min().item():.3f}s "
                f"max:r{max_rank}:{values[max_rank].item():.3f}s"
            )
        return " | ".join(parts)

    def _maybe_log_omnigen2_forward_profile(self, device: torch.device):
        model = self.accelerator.unwrap_model(self.model)
        profile_tensor = getattr(model, "_last_omnigen2_forward_profile_tensor", None)
        if profile_tensor is None:
            return

        profile_tensor = profile_tensor.detach().to(device=device, dtype=torch.float32)
        gathered = self.accelerator.gather(profile_tensor)
        num_ranks = max(int(self.accelerator.num_processes), 1)
        if gathered.numel() % num_ranks != 0:
            if self.accelerator.is_main_process:
                logger.warning(
                    "Cannot format OmniGen2 forward profile: gathered shape=%s num_ranks=%d",
                    tuple(gathered.shape),
                    num_ranks,
                )
            return
        gathered = gathered.reshape(num_ranks, -1).detach().cpu()
        if not self.accelerator.is_main_process:
            return

        phase_names = list(getattr(model, "_last_omnigen2_forward_profile", {}).get("phase_names", []))
        if not phase_names:
            phase_names = ["build_inputs", "noise_sched", "video_pre", "action_pre", "mask", "mot", "post_loss"]
        num_phases = len(phase_names)
        expected = 2 * num_phases + 10
        if gathered.shape[1] < expected:
            logger.warning("OmniGen2 forward profile is too short: shape=%s expected>=%d", tuple(gathered.shape), expected)
            return

        meta = gathered[:, 2 * num_phases :]
        should_log = bool(torch.any(meta[:, 1] > 0.5).item())
        if not should_log:
            return

        durations = gathered[:, :num_phases]
        arrivals = gathered[:, num_phases : 2 * num_phases]
        parts = []
        for idx, name in enumerate(phase_names):
            values = durations[:, idx]
            arrival_values = arrivals[:, idx]
            max_rank = int(torch.argmax(values).item())
            arrival_skew = float((arrival_values.max() - arrival_values.min()).item())
            parts.append(
                f"{name}=mean:{values.mean().item():.3f}s "
                f"min:{values.min().item():.3f}s max:r{max_rank}:{values[max_rank].item():.3f}s "
                f"arr_skew:{arrival_skew:.3f}s"
            )

        total = durations.sum(dim=1)
        total_max_rank = int(torch.argmax(total).item())
        peak = meta[:, 9]
        peak_rank = int(torch.argmax(peak).item())
        logger.info(
            "[omnigen2-forward-profile] step=%d total=mean:%.3fs min:%.3fs max:r%d:%.3fs "
            "video_seq=%.0f-%.0f action_seq=%.0f-%.0f cap=%.0f-%.0f seq=%.0f-%.0f "
            "peak=max:r%d:%.2fGiB | %s",
            self.global_step,
            total.mean().item(),
            total.min().item(),
            total_max_rank,
            total[total_max_rank].item(),
            meta[:, 3].min().item(),
            meta[:, 3].max().item(),
            meta[:, 4].min().item(),
            meta[:, 4].max().item(),
            meta[:, 5].min().item(),
            meta[:, 6].max().item(),
            meta[:, 7].min().item(),
            meta[:, 8].max().item(),
            peak_rank,
            peak[peak_rank].item(),
            " | ".join(parts),
        )

    @staticmethod
    def _looks_like_weights_checkpoint(path: Path) -> bool:
        return path.suffix.lower() in {".pt", ".pth", ".bin"} or path.is_file()

    def _load_weights_checkpoint_before_prepare(self):
        resume = self.resume
        if not resume:
            return

        resume_path = Path(str(resume))
        if resume_path.is_dir():
            return
        if not resume_path.exists():
            if self._looks_like_weights_checkpoint(resume_path):
                raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
            return
        if not self._looks_like_weights_checkpoint(resume_path):
            return

        logger.info(
            "Loading weights checkpoint before accelerator.prepare so optimizer/master weights start from checkpoint: %s",
            resume,
        )
        self.model.load_checkpoint(str(resume_path), optimizer=None)
        self._weights_resume_loaded_before_prepare = True
        logger.warning("Loaded weights only; optimizer/scheduler/global step will start fresh.")

    def _resume_or_load_checkpoint(self):
        resume = self.resume
        if not resume:
            return
        resume_path = Path(str(resume))
        if self._weights_resume_loaded_before_prepare:
            logger.info("Weights checkpoint was loaded before accelerator.prepare; skipping post-prepare weights load.")
            return
        if resume_path.is_dir():
            logger.info("Resuming full training state from directory: %s", resume)
            self.load_training_state(str(resume_path))
            return
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        logger.info("Loading weight checkpoint only: %s", resume)
        self.accelerator.unwrap_model(self.model).load_checkpoint(str(resume_path), optimizer=None)
        logger.warning(
            "Loaded weights after accelerator.prepare; optimizer/master weights may not be synchronized. "
            "Prefer loading weights-only checkpoints before prepare."
        )

    def _set_dit_only_train_mode(self):
        # Match DiffSynth's freeze_except("dit"): only DiT stays trainable/in-train-mode.
        logger.info("Setting DiT to train mode and freezing other model components.")
        model = self.accelerator.unwrap_model(self.model)
        self._apply_dit_only_train_mode(model)

    @staticmethod
    def _apply_dit_only_train_mode(model):
        model.eval()
        model.requires_grad_(False)
        model.dit.train()
        model.dit.requires_grad_(True)
        apply_policy = getattr(model, "apply_trainable_policy", None)
        if callable(apply_policy):
            apply_policy()
        proprio_encoder = getattr(model, "proprio_encoder", None)
        if proprio_encoder is not None:
            proprio_encoder.train()
            proprio_encoder.requires_grad_(True)

    @staticmethod
    def _to_batched_eval_sample(sample):
        video = sample["video"]
        prompt = sample["prompt"]
        action = sample.get("action", None)
        proprio = sample.get("proprio", None)
        context = sample.get("context", None)
        context_mask = sample.get("context_mask", None)
        text_hidden_states = sample.get("text_hidden_states", None)
        text_attention_mask = sample.get("text_attention_mask", None)
        dataset_name = sample.get("dataset_name", None)
        embodiment = sample.get("embodiment", None)
        action_is_pad = sample.get("action_is_pad", None)
        action_dim_is_pad = sample.get("action_dim_is_pad", None)
        image_is_pad = sample.get("image_is_pad", None)
        proprio_is_pad = sample.get("proprio_is_pad", None)
        proprio_dim_is_pad = sample.get("proprio_dim_is_pad", None)

        if not isinstance(video, torch.Tensor):
            raise TypeError(
                f"Expected tensor video for evaluation, got {type(video)}. "
                "Evaluation now expects `video` with shape [3,T,H,W] or [B,3,T,H,W]."
            )
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if video.ndim != 5:
            raise ValueError(f"Expected video shape [3,T,H,W] or [B,3,T,H,W], got {tuple(video.shape)}")
        num_video_frames = video.shape[2]
        if num_video_frames <= 1:
            raise ValueError(f"`sample['video']` must have at least 2 frames for action evaluation, got {num_video_frames}")

        if isinstance(prompt, str):
            prompt = [prompt]
        elif isinstance(prompt, tuple):
            prompt = list(prompt)
        elif not isinstance(prompt, list):
            raise TypeError(f"Expected prompt type str/list[str], got {type(prompt)}")
        if len(prompt) != video.shape[0]:
            raise ValueError(f"Prompt batch mismatch: len(prompt)={len(prompt)} vs video batch={video.shape[0]}")
        
        action_horizon = None
        action = None
        if "action" in sample:
            action = sample["action"]
            if not isinstance(action, torch.Tensor):
                raise TypeError(
                    f"`sample['action']` must be a torch.Tensor, got {type(action)}"
                )
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3:
                raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
            if action.shape[1] % (num_video_frames - 1) != 0:
                raise ValueError(f"`sample['action']` temporal dimension must be divisible by video frames-1={num_video_frames - 1}, got {action.shape[1]}")
            action_horizon = int(action.shape[1])

        proprio = None
        if "proprio" in sample:
            proprio = sample["proprio"]
            if not isinstance(proprio, torch.Tensor):
                raise TypeError(f"`sample['proprio']` must be a torch.Tensor, got {type(proprio)}")
            if proprio.ndim == 2:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")

        def _batch_mask(name: str, value, expected_ndim: int):
            if value is None:
                return None
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"`sample['{name}']` must be a torch.Tensor, got {type(value)}")
            if value.ndim == expected_ndim - 1:
                value = value.unsqueeze(0)
            if value.ndim != expected_ndim:
                raise ValueError(
                    f"`sample['{name}']` must have {expected_ndim} dims after batching, got {tuple(value.shape)}"
                )
            return value

        action_is_pad = _batch_mask("action_is_pad", action_is_pad, 2)
        action_dim_is_pad = _batch_mask("action_dim_is_pad", action_dim_is_pad, 2)
        image_is_pad = _batch_mask("image_is_pad", image_is_pad, 2)
        proprio_is_pad = _batch_mask("proprio_is_pad", proprio_is_pad, 2)
        proprio_dim_is_pad = _batch_mask("proprio_dim_is_pad", proprio_dim_is_pad, 2)

        if context is not None or context_mask is not None:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must both exist in eval sample.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
        if text_hidden_states is not None or text_attention_mask is not None:
            if text_hidden_states is None or text_attention_mask is None:
                raise ValueError("`text_hidden_states` and `text_attention_mask` must both exist in eval sample.")
            if text_hidden_states.ndim == 2:
                text_hidden_states = text_hidden_states.unsqueeze(0)
            if text_attention_mask.ndim == 1:
                text_attention_mask = text_attention_mask.unsqueeze(0)
            if text_hidden_states.ndim != 3 or text_attention_mask.ndim != 2:
                raise ValueError(
                    "`text_hidden_states/text_attention_mask` must be [B,L,D]/[B,L], "
                    f"got {tuple(text_hidden_states.shape)} and {tuple(text_attention_mask.shape)}"
                )

        return {
            "video": video,
            "prompt": prompt,
            "instruction": prompt,
            "action": action,
            "proprio": proprio,
            "action_is_pad": action_is_pad,
            "action_dim_is_pad": action_dim_is_pad,
            "image_is_pad": image_is_pad,
            "proprio_is_pad": proprio_is_pad,
            "proprio_dim_is_pad": proprio_dim_is_pad,
            "context": context,
            "context_mask": context_mask,
            "text_hidden_states": text_hidden_states,
            "text_attention_mask": text_attention_mask,
            "action_horizon": action_horizon,
            "dataset_name": dataset_name,
            "embodiment": [embodiment] if isinstance(embodiment, str) else embodiment,
        }

    @staticmethod
    def _single_eval_dataset_name(sample):
        dataset_name = sample.get("dataset_name", None)
        if dataset_name is None:
            return None
        if isinstance(dataset_name, str):
            return dataset_name
        if isinstance(dataset_name, (list, tuple)):
            if len(dataset_name) == 0:
                return None
            return str(dataset_name[0])
        return str(dataset_name)

    def _get_eval_processor(self, sample):
        if hasattr(self.val_dataset, "lerobot_dataset"):
            return self.val_dataset.lerobot_dataset.processor
        processor_by_dataset = getattr(self.val_dataset, "processor_by_dataset", None)
        if processor_by_dataset:
            dataset_name = self._single_eval_dataset_name(sample)
            if dataset_name in processor_by_dataset:
                return processor_by_dataset[dataset_name]
        processor = getattr(self.val_dataset, "processor", None)
        if processor is not None:
            return processor
        raise AttributeError(
            "Could not find a processor on val_dataset. Expected either "
            "`val_dataset.lerobot_dataset.processor` or `val_dataset.processor`."
        )

    @torch.no_grad()
    def evaluate(self):
        if self.val_dataset is None:
            return None

        model = self.accelerator.unwrap_model(self.model)
        was_dit_training = model.dit.training
        model.eval()

        model_stack = str(getattr(model, "stack", ""))
        is_omnigen2_stack = model_stack == "omnigen2"
        is_ovis_u1_stack = model_stack == "ovis_u1"
        is_flux2_stack = model_stack == "flux2"
        is_dim_stack = model_stack == "dim"
        is_image_prediction_stack = is_omnigen2_stack or is_ovis_u1_stack or is_flux2_stack or is_dim_stack
        rng = torch.Generator(device="cpu").manual_seed(self.global_step + self.accelerator.process_index)
        eval_indices = torch.randint(
            0,
            len(self.val_dataset),
            (self.eval_num_samples,),
            generator=rng,
        ).tolist()

        local_metric_rows = []
        video_path = None
        val_video_augmentation = getattr(self.val_dataset, "video_augmentation", None)
        val_has_video_augmentation = hasattr(self.val_dataset, "video_augmentation")
        if val_has_video_augmentation:
            self.val_dataset.video_augmentation = None
        try:
            for sample_slot, eval_index in enumerate(eval_indices):
                sample = self._to_batched_eval_sample(self.val_dataset[eval_index])

                # 1. training loss
                with self.accelerator.autocast():
                    val_loss, _ = model.training_loss(sample)
                    val_loss = val_loss.float().item()

                prompt = sample["prompt"][0]
                video0 = sample["video"][0] # Tensor [3, T, H, W] in (-1, 1)
                action = sample["action"][0] if "action" in sample and sample["action"] is not None else None
                proprio = sample["proprio"][0, 0] if "proprio" in sample and sample["proprio"] is not None else None # from [1, T, d] to [d]
                input_image = video0[:, 0].unsqueeze(0)
                _, num_frames, _, _ = video0.shape

                # 2. inference and video saving
                infer_kwargs = {
                    "input_image": input_image,
                    "num_frames": num_frames,
                    "action": action,
                    "action_horizon": sample["action_horizon"],
                    "proprio": proprio,
                    "text_cfg_scale": 1.0,
                    "action_cfg_scale": 1.0,
                    "num_inference_steps": self.eval_num_inference_steps,
                    "seed": 42,
                    "tiled": False,
                }
                if (is_omnigen2_stack or is_flux2_stack) and sample.get("text_hidden_states") is not None:
                    infer_kwargs["prompt"] = None
                    infer_kwargs["context"] = sample["text_hidden_states"][0]
                    infer_kwargs["context_mask"] = sample["text_attention_mask"][0]
                elif sample["context"] is not None:
                    infer_kwargs["prompt"] = None
                    infer_kwargs["context"] = sample["context"][0]
                    infer_kwargs["context_mask"] = sample["context_mask"][0]
                else:
                    infer_kwargs["prompt"] = prompt

                pred = model.infer(
                    **infer_kwargs,
                )

                pred_video = pred["video"]
                pred_action = pred.get("action", None)

                # 3. inference metrics against GT video
                pred_video_tensor = pil_frames_to_video_tensor(pred_video)
                gt_video_tensor = ((video0.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()
                if is_image_prediction_stack:
                    gt_video_tensor = gt_video_tensor[:, -1:, :, :]

                assert pred_video_tensor.shape == gt_video_tensor.shape, (
                    "Eval infer prediction/GT shape mismatch: "
                    f"pred={tuple(pred_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
                )

                psnr_rollout_vs_gt = video_psnr(pred=pred_video_tensor, target=gt_video_tensor)
                ssim_rollout_vs_gt = video_ssim(pred=pred_video_tensor, target=gt_video_tensor)

                action_l1 = None
                action_l2 = None
                if action is not None and pred_action is not None:
                    if sample["proprio"] is None:
                        raise ValueError("Eval sample must contain `proprio` for action denormalization.")
                    proprio = sample["proprio"].detach().to(device="cpu", dtype=torch.float32)

                    processor = self._get_eval_processor(sample)

                    denorm_actions = {}
                    action_meta = processor.shape_meta["action"]
                    state_meta = processor.shape_meta["state"]
                    for action_name, raw_action in (("pred", pred_action), ("gt", action)):
                        if not isinstance(raw_action, torch.Tensor):
                            raise TypeError(f"{action_name} action must be a torch.Tensor, got {type(raw_action)}")
                        if raw_action.ndim == 2:
                            action_btd = raw_action.unsqueeze(0)
                        elif raw_action.ndim == 3 and raw_action.shape[0] == 1:
                            action_btd = raw_action
                        else:
                            raise ValueError(
                                f"{action_name} action must have shape [T, D] or [1, T, D], got {tuple(raw_action.shape)}"
                            )
                        action_btd = action_btd.detach().to(device="cpu", dtype=torch.float32)

                        batch = {
                            "action": action_btd,
                            "state": proprio,
                            "embodiment": sample.get("embodiment"),
                        }
                        batch = processor.action_state_merger.backward(batch)
                        batch = processor.normalizer.backward(batch)
                        merged_batch = {
                            "action": {meta["key"]: batch["action"][meta["key"]].squeeze(0) for meta in action_meta},
                            "state": {meta["key"]: batch["state"][meta["key"]].squeeze(0) for meta in state_meta},
                        }
                        merged_batch = processor.action_state_merger.forward(merged_batch)
                        denorm_action = merged_batch["action"].unsqueeze(0)
                        if denorm_action.ndim != 3 or denorm_action.shape[0] != 1:
                            raise ValueError(
                                f"Denormalized {action_name} action must have shape [1, T, D], got {tuple(denorm_action.shape)}"
                            )
                        denorm_actions[action_name] = denorm_action

                    pred_action_denorm = denorm_actions["pred"]
                    gt_action_denorm = denorm_actions["gt"]

                    if pred_action_denorm.shape != gt_action_denorm.shape:
                        raise ValueError(
                            "Predicted action/GT action shape mismatch after denormalization: "
                            f"pred={tuple(pred_action_denorm.shape)} vs gt={tuple(gt_action_denorm.shape)}"
                        )
                    action_diff = pred_action_denorm - gt_action_denorm
                    action_valid = torch.ones_like(action_diff, dtype=torch.bool)
                    action_is_pad = sample.get("action_is_pad")
                    if isinstance(action_is_pad, torch.Tensor):
                        if action_is_pad.ndim != 2 or action_is_pad.shape != action_diff.shape[:2]:
                            raise ValueError(
                                "`action_is_pad` shape mismatch for eval action metrics: "
                                f"got {tuple(action_is_pad.shape)} vs expected {tuple(action_diff.shape[:2])}"
                            )
                        action_valid &= ~action_is_pad.to(device=action_valid.device, dtype=torch.bool).unsqueeze(-1)
                    action_dim_is_pad = sample.get("action_dim_is_pad")
                    if isinstance(action_dim_is_pad, torch.Tensor):
                        if action_dim_is_pad.ndim != 2 or action_dim_is_pad.shape[0] != action_diff.shape[0] or action_dim_is_pad.shape[1] != action_diff.shape[2]:
                            raise ValueError(
                                "`action_dim_is_pad` shape mismatch for eval action metrics: "
                                f"got {tuple(action_dim_is_pad.shape)} vs expected ({action_diff.shape[0]}, {action_diff.shape[2]})"
                            )
                        action_valid &= ~action_dim_is_pad.to(device=action_valid.device, dtype=torch.bool).unsqueeze(1)
                    valid_count = action_valid.sum().clamp(min=1)
                    action_l1 = action_diff.abs().masked_select(action_valid).sum().div(valid_count).item()
                    action_l2 = action_diff.pow(2).masked_select(action_valid).sum().div(valid_count).item()

                # 4. VAE reconstruction metrics against GT target.
                if is_omnigen2_stack:
                    gt_final = video0[:, -1].unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
                    vae_latents = model._encode_omnigen2_image_latents(gt_final)
                    vae_image = model._decode_omnigen2_image_latents(vae_latents)[0]
                    vae_np = ((vae_image.detach().float().cpu().clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8).permute(1, 2, 0).numpy()
                    vae_video_tensor = pil_frames_to_video_tensor([Image.fromarray(vae_np)])
                elif is_ovis_u1_stack:
                    gt_final = video0[:, -1].unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
                    vae_tokens, _ = model._encode_ovis_u1_image_tokens(gt_final, time_value=0.0)
                    vae_image = model._decode_ovis_u1_image_tokens(
                        vae_tokens,
                        height=int(gt_final.shape[-2]),
                        width=int(gt_final.shape[-1]),
                    )[0]
                    vae_np = ((vae_image.detach().float().cpu().clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8).permute(1, 2, 0).numpy()
                    vae_video_tensor = pil_frames_to_video_tensor([Image.fromarray(vae_np)])
                elif is_flux2_stack:
                    gt_final = video0[:, -1].unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
                    vae_tokens, _ = model._encode_flux2_image_tokens(gt_final, time_value=0.0)
                    vae_image = model._decode_flux2_image_tokens(
                        vae_tokens,
                        height=int(gt_final.shape[-2]),
                        width=int(gt_final.shape[-1]),
                    )[0]
                    vae_np = ((vae_image.detach().float().cpu().clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8).permute(1, 2, 0).numpy()
                    vae_video_tensor = pil_frames_to_video_tensor([Image.fromarray(vae_np)])
                elif is_dim_stack:
                    gt_final = video0[:, -1].unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
                    vae_latents = model._encode_dim_image_latents(gt_final)
                    vae_image = model._decode_dim_image_latents(vae_latents)[0]
                    vae_np = ((vae_image.detach().float().cpu().clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8).permute(1, 2, 0).numpy()
                    vae_video_tensor = pil_frames_to_video_tensor([Image.fromarray(vae_np)])
                else:
                    gt_video_batch = video0.unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
                    vae_latents = model._encode_video_latents(gt_video_batch, tiled=False)
                    vae_recon_video = model._decode_latents(vae_latents, tiled=False)
                    vae_video_tensor = pil_frames_to_video_tensor(vae_recon_video)

                assert vae_video_tensor.shape == gt_video_tensor.shape, (
                    "Eval VAE reconstruction/GT shape mismatch: "
                    f"vae={tuple(vae_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
                )

                psnr_decode_vs_gt = video_psnr(pred=vae_video_tensor, target=gt_video_tensor)
                ssim_decode_vs_gt = video_ssim(pred=vae_video_tensor, target=gt_video_tensor)

                psnr_rollout_vs_decode = video_psnr(pred=pred_video_tensor, target=vae_video_tensor)
                ssim_rollout_vs_decode = video_ssim(pred=pred_video_tensor, target=vae_video_tensor)

                stitched_video_tensor = torch.cat(
                    [pred_video_tensor, vae_video_tensor, gt_video_tensor],
                    dim=2,
                ).contiguous()
                stitched_frames = []
                for t in range(stitched_video_tensor.shape[1]):
                    frame = (stitched_video_tensor[:, t].permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
                    stitched_frames.append(Image.fromarray(frame))

                video_name = f"step_{self.global_step:06d}_rank_{self.accelerator.process_index:03d}"
                if self.eval_num_samples > 1:
                    video_name += f"_sample_{sample_slot:03d}"
                current_video_path = os.path.join(self.eval_dir, f"{video_name}.mp4")
                save_mp4(stitched_frames, current_video_path, fps=8)
                if video_path is None:
                    video_path = current_video_path

                action_valid = action_l2 is not None and action_l1 is not None
                local_metric_rows.append(
                    [
                        float(val_loss),
                        float(psnr_rollout_vs_gt),
                        float(ssim_rollout_vs_gt),
                        float(psnr_rollout_vs_decode),
                        float(ssim_rollout_vs_decode),
                        float(psnr_decode_vs_gt),
                        float(ssim_decode_vs_gt),
                        float(action_l2) if action_valid else 0.0,
                        float(action_l1) if action_valid else 0.0,
                        1.0 if action_valid else 0.0,
                    ]
                )
        finally:
            if val_has_video_augmentation:
                self.val_dataset.video_augmentation = val_video_augmentation
            if was_dit_training:
                self._set_dit_only_train_mode()

        local_metrics = torch.tensor(
            local_metric_rows,
            device=self.accelerator.device,
            dtype=torch.float32,
        )
        gathered_metrics = self.accelerator.gather_for_metrics(local_metrics)
        mean_metrics = gathered_metrics[:, :7].mean(dim=0)
        action_valid_count = gathered_metrics[:, 9].sum()
        action_l2_mean = None
        action_l1_mean = None
        if action_valid_count.item() > 0:
            action_l2_mean = (gathered_metrics[:, 7].sum() / action_valid_count).item()
            action_l1_mean = (gathered_metrics[:, 8].sum() / action_valid_count).item()

        result = {
            "val_loss": float(mean_metrics[0].item()),
            "psnr_rg": float(mean_metrics[1].item()),
            "ssim_rg": float(mean_metrics[2].item()),
            "psnr_rd": float(mean_metrics[3].item()),
            "ssim_rd": float(mean_metrics[4].item()),
            "psnr_dg": float(mean_metrics[5].item()),
            "ssim_dg": float(mean_metrics[6].item()),
            "video_path": video_path,
            "num_samples": int(gathered_metrics.shape[0]),
        }
        if action_l2_mean is not None:
            result["action_l2"] = float(action_l2_mean)
        if action_l1_mean is not None:
            result["action_l1"] = float(action_l1_mean)
        return result

    def _save_weights_checkpoint(self, step_tag: str):
        model = self.accelerator.unwrap_model(self.model)
        ckpt_path = os.path.join(self.weights_dir, f"{step_tag}.pt")
        model.save_checkpoint(ckpt_path, optimizer=None, step=self.global_step)
        return ckpt_path

    def _save_trainer_state(self, state_path: str):
        state_file = os.path.join(state_path, "trainer_state.json")
        payload = {
            "global_step": int(self.global_step),
            "epoch": int(self.epoch),
            "batch_in_epoch": int(self.batch_in_epoch),
        }
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    def _prune_old_state_checkpoints(self, keep_step_tag: str):
        if not self.keep_latest_state_only:
            return
        state_root = Path(self.state_dir)
        if not state_root.exists():
            return
        for path in state_root.iterdir():
            if not path.is_dir() or path.name == keep_step_tag:
                continue
            if not re.match(r"step[_-]\d+$", path.name):
                continue
            shutil.rmtree(path)
            logger.info("Removed old training state checkpoint: %s", path)

    def save_checkpoint(self):
        step_tag = f"step_{self.global_step:06d}"

        self.accelerator.wait_for_everyone()
        ckpt_path = None
        if self.accelerator.is_main_process:
            ckpt_path = self._save_weights_checkpoint(step_tag=step_tag)
        self.accelerator.wait_for_everyone()

        state_path = os.path.join(self.state_dir, step_tag)
        ensure_dir(state_path)
        self.accelerator.save_state(output_dir=state_path)
        if self.accelerator.is_main_process:
            self._save_trainer_state(state_path)
            self._prune_old_state_checkpoints(keep_step_tag=step_tag)
        self.accelerator.wait_for_everyone()

        return {"weights_path": ckpt_path, "state_path": state_path}

    def load_training_state(self, state_dir: str):
        self.accelerator.load_state(input_dir=state_dir)
        state_file = Path(state_dir) / "trainer_state.json"
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.global_step = int(payload["global_step"])

            if "epoch" in payload and "batch_in_epoch" in payload:
                self.epoch = int(payload["epoch"])
                self.batch_in_epoch = int(payload["batch_in_epoch"])
                self.train_sampler.set_epoch_offset(self.epoch)
                self.train_sampler.set_resume_batch_offset(self.batch_in_epoch)
                logger.info(
                    "Restored dataloader progress: epoch=%d batch_in_epoch=%d sample_offset=%d",
                    self.epoch,
                    self.batch_in_epoch,
                    self.batch_in_epoch * self.batch_size * self.accelerator.num_processes,
                )
            else:
                self.epoch = 0
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                logger.warning(
                    "State file does not contain `epoch`/`batch_in_epoch`; "
                    "optimizer/scheduler were restored, but dataloader progress resume is skipped."
                )
            self.accelerator.wait_for_everyone()
            return

        match = re.search(r"step[_-](\d+)$", str(state_dir).rstrip("/"))
        if match:
            self.global_step = int(match.group(1))
        else:
            self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self.train_sampler.clear_resume_batch_offset()
        self.accelerator.wait_for_everyone()
        logger.info("Loaded accelerate training state from %s at step=%d", state_dir, self.global_step)
        logger.warning(
            "State file `%s` is missing; dataloader progress resume is skipped.",
            state_file,
        )

    def train(self):
        self._set_dit_only_train_mode()

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        if self.max_steps is None:
            raise ValueError("`max_steps` must be set before entering the while-step training loop.")

        logger.info("Starting training with max_steps=%d.", self.max_steps)
        data_iter = iter(self.train_loader)
        self.run_start_step = self.global_step
        self.run_start_time = time.perf_counter()

        while self.global_step < self.max_steps:
            timer_active = (
                self.rank_timer_every > 0
                and (self.global_step + 1) % self.rank_timer_every == 0
            )
            step_timings = {}
            self._rank_timer_sync(timer_active)
            step_start = time.perf_counter()
            data_start = step_start
            try:
                sample = next(data_iter)
                self.batch_in_epoch += 1
            except StopIteration:
                self.epoch += 1
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                data_iter = iter(self.train_loader)
                continue
            step_timings["data"] = time.perf_counter() - data_start

            with self.accelerator.accumulate(self.model):
                train_model = self.model if hasattr(self.model, "training_loss") else self.accelerator.unwrap_model(self.model)

                iter_training_losses = getattr(train_model, "iter_training_losses", None)
                loss = None
                loss_dict = {}
                forward_elapsed = 0.0
                backward_elapsed = 0.0

                if callable(iter_training_losses):
                    objective_iter = iter(iter_training_losses(sample))
                    objective_count = 0
                    while True:
                        objective_forward_start = time.perf_counter()
                        try:
                            with self.accelerator.autocast():
                                objective_loss, objective_loss_dict = next(objective_iter)
                        except StopIteration:
                            break
                        forward_elapsed += time.perf_counter() - objective_forward_start
                        objective_count += 1
                        loss = (
                            objective_loss.detach().float()
                            if loss is None
                            else loss + objective_loss.detach().float()
                        )
                        for key, value in objective_loss_dict.items():
                            loss_dict[key] = float(value)

                        objective_backward_start = time.perf_counter()
                        self.accelerator.backward(objective_loss)
                        backward_elapsed += time.perf_counter() - objective_backward_start

                    if objective_count <= 0:
                        raise RuntimeError("`iter_training_losses` yielded no training losses.")
                else:
                    forward_start = time.perf_counter()
                    with self.accelerator.autocast():
                        loss, loss_dict = train_model.training_loss(sample)
                    forward_elapsed = time.perf_counter() - forward_start

                    backward_start = time.perf_counter()
                    self.accelerator.backward(loss)
                    backward_elapsed = time.perf_counter() - backward_start

                self._rank_timer_sync(timer_active)
                step_timings["forward"] = forward_elapsed
                self._maybe_log_omnigen2_forward_profile(loss.device)
                step_timings["backward"] = backward_elapsed

                if self.accelerator.sync_gradients:
                    optimizer_start = time.perf_counter()
                    grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    if not self.accelerator.optimizer_step_was_skipped:
                        self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self._rank_timer_sync(timer_active)
                    step_timings["optimizer"] = time.perf_counter() - optimizer_start

                    self.global_step += 1
                    metrics_start = time.perf_counter()
                    global_loss = float(
                        self.accelerator.gather(loss.detach().float().reshape(1)).mean().item()
                    )
                    global_loss_metrics = {}
                    for key, value in loss_dict.items():
                        metric_tensor = torch.tensor(float(value), device=loss.device, dtype=torch.float32).reshape(1)
                        global_loss_metrics[key] = float(
                            self.accelerator.gather(metric_tensor).mean().item()
                        )
                    grad_norm_tensor = torch.tensor(grad_norm, device=loss.device, dtype=torch.float32)
                    global_grad_norm = float(self.accelerator.gather(grad_norm_tensor).mean().item())
                    self._rank_timer_sync(timer_active)
                    step_timings["metrics"] = time.perf_counter() - metrics_start
                    step_timings["total"] = time.perf_counter() - step_start

                    current_lr = float(self.optimizer.param_groups[0]["lr"])

                    should_log = self.log_every > 0 and self.global_step % self.log_every == 0
                    should_log_timer = self.rank_timer_every > 0 and self.global_step % self.rank_timer_every == 0
                    timer_names = None
                    timer_stats = None
                    if should_log_timer:
                        timer_names, timer_stats = self._gather_rank_timer_stats(step_timings, loss.device)

                    if should_log and self.accelerator.is_main_process:
                        eta_str, steps_per_sec = self._estimate_eta()
                        description = "[train] epoch=%d step=%d/%d loss=%.4f " % (
                            self.epoch,
                            self.global_step,
                            self.max_steps,
                            global_loss,
                        )
                        if global_loss_metrics:
                            detail_str = " ".join([f"{k}={v:.4f}" for k, v in sorted(global_loss_metrics.items())])
                            description += detail_str + " "
                        description += "lr=%.2e speed=%.2f step/s, %.2f samples/s eta=%s" % (
                            current_lr,
                            steps_per_sec,
                            steps_per_sec * self.batch_size * self.accelerator.num_processes,
                            eta_str,
                        )
                        logger.info(description)

                        wandb_payload = {
                            "train/loss": global_loss,
                            "train/grad_norm": global_grad_norm,
                            "train/lr": current_lr,
                            "performance/steps_per_sec": steps_per_sec,
                            "performance/samples_per_sec": steps_per_sec * self.batch_size * self.accelerator.num_processes,
                        }
                        for key, value in global_loss_metrics.items():
                            wandb_payload[f"train/{key}"] = value
                        self._wandb_log(wandb_payload)

                    if should_log_timer and self.accelerator.is_main_process:
                        logger.info(
                            "[timer] step=%d rank-wise %s",
                            self.global_step,
                            self._format_rank_timer_stats(timer_names, timer_stats),
                        )

                    if (
                        self.eval_every > 0
                        and self.val_dataset is not None
                        and self.global_step % self.eval_every == 0
                    ):
                        metrics = self.evaluate()
                        self.accelerator.wait_for_everyone()
                        if metrics is not None and self.accelerator.is_main_process:
                            description = "[eval] step=%d samples=%d val_loss=%.4f infer_psnr=%.4f infer_ssim=%.4f" % (
                                self.global_step,
                                metrics["num_samples"],
                                metrics["val_loss"],
                                metrics["psnr_rd"],
                                metrics["ssim_rd"],
                            )
                            if "action_l2" in metrics:
                                description += " action_l2=%.4f" % metrics["action_l2"]
                            if "action_l1" in metrics:
                                description += " action_l1=%.4f" % metrics["action_l1"]
                            logger.info(description)
                            eval_payload = {
                                "eval/num_samples": int(metrics["num_samples"]),
                                "eval/val_loss": float(metrics["val_loss"]),
                                "eval/psnr_rg": float(metrics["psnr_rg"]),
                                "eval/ssim_rg": float(metrics["ssim_rg"]),
                                "eval/psnr_rd": float(metrics["psnr_rd"]),
                                "eval/ssim_rd": float(metrics["ssim_rd"]),
                                "eval/psnr_dg": float(metrics["psnr_dg"]),
                                "eval/ssim_dg": float(metrics["ssim_dg"]),
                            }
                            if "action_l2" in metrics:
                                eval_payload["eval/action_l2"] = float(metrics["action_l2"])
                            if "action_l1" in metrics:
                                eval_payload["eval/action_l1"] = float(metrics["action_l1"])
                            self._wandb_log(eval_payload)

                    ckpt_info = None
                    if self.save_every > 0 and self.global_step % self.save_every == 0:
                        ckpt_info = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[ckpt] step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )

                    if self.global_step >= self.max_steps:
                        if ckpt_info is None:
                            ckpt_info = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[done] max_steps reached step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )
                        return

        ckpt_info = self.save_checkpoint()
        if self.accelerator.is_main_process:
            logger.info(
                "[done] training finished step=%d weights=%s state=%s",
                self.global_step,
                ckpt_info["weights_path"],
                ckpt_info["state_path"],
            )
        
