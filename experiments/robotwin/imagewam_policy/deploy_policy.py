import importlib
import logging
import os
import sys
import time
import inspect
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from imagewam.datasets.lerobot.processors.imagewam_processor import ImageWAMProcessor
from imagewam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from imagewam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

logger = logging.getLogger(__name__)


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null"}
    return False


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    raise ValueError(f"Cannot parse bool value: {value}")


def _parse_optional_int(value: Any) -> Optional[int]:
    if _is_none_like(value):
        return None
    return int(value)


def _parse_optional_float(value: Any) -> Optional[float]:
    if _is_none_like(value):
        return None
    return float(value)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    key = str(mixed_precision).strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _resolve_sim_cfg_name(sim_cfg_path: Optional[str], sim_cfg_name: Optional[str]) -> str:
    configs_root = (PROJECT_ROOT / "configs").resolve()
    if not _is_none_like(sim_cfg_path):
        cfg_path = Path(str(sim_cfg_path)).expanduser().resolve()
        try:
            relative = cfg_path.relative_to(configs_root)
        except ValueError as exc:
            raise ValueError(
                f"`sim_cfg_path` must be under {configs_root}, got: {cfg_path}"
            ) from exc
        return relative.as_posix()

    if _is_none_like(sim_cfg_name):
        return "sim_robotwin.yaml"
    return str(sim_cfg_name)


def _compose_sim_cfg(
    sim_cfg_path: Optional[str],
    sim_cfg_name: Optional[str],
    sim_task: Optional[str],
) -> DictConfig:
    config_name = _resolve_sim_cfg_name(sim_cfg_path=sim_cfg_path, sim_cfg_name=sim_cfg_name)
    configs_root = (PROJECT_ROOT / "configs").resolve()
    overrides = []
    if not _is_none_like(sim_task):
        overrides.append(f"task={str(sim_task)}")

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(version_base="1.3", config_dir=str(configs_root)):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg


def _apply_model_overrides(cfg: DictConfig, usr_args: Dict[str, Any]) -> None:
    model_overrides = usr_args.get("model_overrides")
    if isinstance(model_overrides, dict):
        for key, value in model_overrides.items():
            cfg.model[key] = value

    # Keep compatibility with older launch wrappers that forwarded only these paths.
    for key in ("omnigen2_model_path", "omnigen2_vae_path", "qwen_path", "ovis_u1_model_path"):
        value = usr_args.get(key)
        if not _is_none_like(value):
            cfg.model[key] = value


def _resolve_target(target: str):
    module_name, _, attr_name = target.rpartition(".")
    if not module_name or not attr_name:
        raise ValueError(f"Invalid Hydra target: {target}")
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def _filter_model_cfg_for_target(model_cfg: DictConfig) -> DictConfig:
    model_dict = OmegaConf.to_container(model_cfg, resolve=True)
    if not isinstance(model_dict, dict):
        raise ValueError(f"`model` config must resolve to a dict, got {type(model_dict)}")

    target = model_dict.get("_target_")
    if _is_none_like(target):
        return OmegaConf.create(model_dict)

    target_fn = _resolve_target(str(target))
    signature = inspect.signature(target_fn)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return OmegaConf.create(model_dict)

    allowed_keys = {"_target_"}
    allowed_keys.update(
        name
        for name, param in signature.parameters.items()
        if param.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
    )
    dropped_keys = sorted(set(model_dict) - allowed_keys)
    if dropped_keys:
        logger.info("Ignoring unsupported model config keys for %s: %s", target, dropped_keys)

    return OmegaConf.create({key: value for key, value in model_dict.items() if key in allowed_keys})


def _resolve_dataset_stats_path(dataset_stats_path: Optional[str]) -> Path:
    if _is_none_like(dataset_stats_path):
        raise FileNotFoundError(
            "`dataset_stats_path` is required. "
            "Please pass it from eval entrypoint overrides."
        )
    resolved = Path(str(dataset_stats_path)).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset stats path not found: {resolved}")
    return resolved


def _resize_rgb(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
    resized = pil_image.resize(size_wh, resample=Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def _robotwin_camera_sizes(layout: str) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    layout = str(layout).strip().lower()
    if layout in {"compact", "compact_288x256", "288x256"}:
        return (256, 192), (128, 96), (128, 96)
    if layout in {"legacy", "legacy_384x320", "384x320"}:
        return (320, 256), (160, 128), (160, 128)
    raise ValueError(
        f"Unsupported robotwin_camera_layout={layout!r}. "
        "Expected one of: compact_288x256, legacy_384x320."
    )


class WorldActionRobotWinPolicy:
    def __init__(
        self,
        model_cfg: DictConfig,
        processor_cfg: DictConfig,
        checkpoint_path: str,
        dataset_stats_path: Path,
        device: str,
        model_dtype: torch.dtype,
        action_horizon: int,
        replan_steps: int,
        tau_star: float,
        num_inference_steps: int,
        sigma_shift: Optional[float],
        seed: Optional[int],
        text_cfg_scale: float,
        negative_prompt: str,
        rand_device: str,
        tiled: bool,
        timing_enabled: bool,
        num_video_frames: int,
        robotwin_camera_layout: str,
    ) -> None:
        model_cfg_copy = OmegaConf.create(OmegaConf.to_container(model_cfg, resolve=True))
        model_cfg_copy.load_text_encoder = True
        model_cfg_copy = _filter_model_cfg_for_target(model_cfg_copy)

        self.model = instantiate(model_cfg_copy, model_dtype=model_dtype, device=device)
        self.model.load_checkpoint(checkpoint_path)
        self.model = self.model.to(device).eval()

        self.processor: ImageWAMProcessor = instantiate(processor_cfg).eval()
        dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
        self.processor.set_normalizer_from_stats(dataset_stats)

        self.action_horizon = int(action_horizon)
        self.replan_steps = int(max(1, min(replan_steps, action_horizon)))
        self.tau_star = float(tau_star)
        self.num_inference_steps = int(num_inference_steps)
        self.sigma_shift = sigma_shift
        self.seed = seed
        self.text_cfg_scale = float(text_cfg_scale)
        self.negative_prompt = str(negative_prompt)
        self.rand_device = str(rand_device)
        self.tiled = bool(tiled)
        self.timing_enabled = bool(timing_enabled)
        self._num_video_frames = int(num_video_frames)
        self.robotwin_camera_layout = str(robotwin_camera_layout)

        self.pending_actions: deque[np.ndarray] = deque()
        self.episode_count = 0
        self.step_count = 0
        self._timing_rollout = {
            "infer_s": 0.0,
            "sim_s": 0.0,
            "infer_count": 0,
            "last_infer_s": 0.0,
            "segment_sums": {},
            "action_predict_total_s": 0.0,
            "action_predict_count": 0,
            "action_predict_min_s": None,
            "action_predict_max_s": None,
            "total_profiled_s": 0.0,
        }

        logger.info(
            "Initialized WorldActionRobotWinPolicy | ckpt=%s | stats=%s | horizon=%d | replan=%d | robotwin_camera_layout=%s",
            checkpoint_path,
            dataset_stats_path,
            self.action_horizon,
            self.replan_steps,
            self.robotwin_camera_layout,
        )

    def _normalize_state(self, state: np.ndarray) -> torch.Tensor:
        state_meta = self.processor.shape_meta["state"]
        if len(state_meta) != 1:
            raise ValueError("Expected exactly one merged state key in shape_meta['state'].")
        state_key = state_meta[0]["key"]

        state_batch = {"state": {state_key: torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)}}
        state_batch = self.processor.action_state_transform(state_batch)
        state_batch = self.processor.normalizer.forward(state_batch)
        return state_batch["state"][state_key]

    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3:
            raise ValueError(f"Expected action tensor [B,T,D], got {tuple(action.shape)}")

        action_meta = self.processor.shape_meta["action"]
        if len(action_meta) != 1:
            raise ValueError("Expected exactly one merged action key in shape_meta['action'].")

        action_key = action_meta[0]["key"]
        normalizer = self.processor.normalizer.normalizers["action"][action_key]
        denorm = normalizer.backward(action.to(dtype=torch.float32, device="cpu"))
        return denorm.numpy()

    def _build_robotwin_image_tensor(self, observation: Dict[str, Any]) -> torch.Tensor:
        obs_data = observation["observation"]
        head_size, left_size, right_size = _robotwin_camera_sizes(self.robotwin_camera_layout)
        head = _resize_rgb(obs_data["head_camera"]["rgb"], head_size)
        left = _resize_rgb(obs_data["left_camera"]["rgb"], left_size)
        right = _resize_rgb(obs_data["right_camera"]["rgb"], right_size)
        bottom = np.concatenate([left, right], axis=1)
        image = np.concatenate([head, bottom], axis=0)

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(
            device=self.model.device,
            dtype=self.model.torch_dtype,
        )
        image_tensor = image_tensor * (2.0 / 255.0) - 1.0
        return image_tensor

    def _infer_action_chunk(self, observation: Dict[str, Any], instruction: str) -> np.ndarray:
        image_tensor = self._build_robotwin_image_tensor(observation)
        state_vector = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        proprio = self._normalize_state(state_vector)

        prompt = DEFAULT_PROMPT.format(task=instruction)
        infer_kwargs = {
            "prompt": prompt,
            "input_image": image_tensor,
            "action_horizon": self.action_horizon,
            "proprio": proprio,
            "negative_prompt": self.negative_prompt,
            "text_cfg_scale": self.text_cfg_scale,
            "tau_star": self.tau_star,
            "num_inference_steps": self.num_inference_steps,
            "sigma_shift": self.sigma_shift,
            "seed": self.seed,
            "rand_device": self.rand_device,
            "tiled": self.tiled,
        }
        infer_action_params = inspect.signature(self.model.infer_action).parameters
        if "num_video_frames" in infer_action_params:
            infer_kwargs["num_video_frames"] = int(self._num_video_frames)
        if "profile_infer_timing" in infer_action_params:
            infer_kwargs["profile_infer_timing"] = self.timing_enabled
        infer_t0 = time.perf_counter() if self.timing_enabled else 0.0
        with torch.no_grad():
            pred = self.model.infer_action(**infer_kwargs)
        if self.timing_enabled:
            infer_s = time.perf_counter() - infer_t0
            self._timing_rollout["infer_s"] += infer_s
            self._timing_rollout["infer_count"] += 1
            self._timing_rollout["last_infer_s"] = infer_s
            self._accumulate_model_timing(pred.get("timing"))

        action_tensor = pred["action"]  # [T, D]
        action_chunk = self._denormalize_action(action_tensor)[0]  # [T, D]
        return action_chunk

    def _fill_action_queue(self, observation: Dict[str, Any], instruction: str) -> None:
        action_chunk = self._infer_action_chunk(observation=observation, instruction=instruction)
        n_exec = min(self.replan_steps, action_chunk.shape[0])
        for i in range(n_exec):
            self.pending_actions.append(np.asarray(action_chunk[i], dtype=np.float32))

    def should_request_observation(self) -> bool:
        return not self.pending_actions

    def step(self, task_env, observation: Optional[Dict[str, Any]]) -> None:
        if not self.pending_actions:
            if observation is None:
                raise ValueError(
                    "Observation is required when action queue is empty "
                    "(replan step for imagewam)."
                )
            instruction = task_env.get_instruction()
            self._fill_action_queue(observation=observation, instruction=instruction)

        if not self.pending_actions:
            logger.warning("No action generated; skip current eval step.")
            return

        action = self.pending_actions.popleft()
        sim_t0 = time.perf_counter() if self.timing_enabled else 0.0
        task_env.take_action(action, action_type="qpos")
        if self.timing_enabled:
            self._timing_rollout["sim_s"] += time.perf_counter() - sim_t0
        self.step_count += 1

    def reset_timing_rollout(self) -> None:
        self._timing_rollout["infer_s"] = 0.0
        self._timing_rollout["sim_s"] = 0.0
        self._timing_rollout["infer_count"] = 0
        self._timing_rollout["last_infer_s"] = 0.0
        self._timing_rollout["segment_sums"] = {}
        self._timing_rollout["action_predict_total_s"] = 0.0
        self._timing_rollout["action_predict_count"] = 0
        self._timing_rollout["action_predict_min_s"] = None
        self._timing_rollout["action_predict_max_s"] = None
        self._timing_rollout["total_profiled_s"] = 0.0

    def _accumulate_model_timing(self, timing: Any) -> None:
        if not isinstance(timing, dict):
            return

        segment_sums = self._timing_rollout["segment_sums"]
        segments = timing.get("segments", {})
        if isinstance(segments, dict):
            for name, value in segments.items():
                segment_sums[str(name)] = float(segment_sums.get(str(name), 0.0)) + float(value)

        action_predict_count = int(timing.get("num_inference_steps", 0) or 0)
        self._timing_rollout["action_predict_total_s"] += float(
            timing.get("action_predict_total_s", 0.0) or 0.0
        )
        self._timing_rollout["action_predict_count"] += action_predict_count
        action_predict_min_s = timing.get("action_predict_min_s")
        action_predict_max_s = timing.get("action_predict_max_s")
        if action_predict_min_s is not None:
            current_min = self._timing_rollout["action_predict_min_s"]
            value = float(action_predict_min_s)
            self._timing_rollout["action_predict_min_s"] = value if current_min is None else min(current_min, value)
        if action_predict_max_s is not None:
            current_max = self._timing_rollout["action_predict_max_s"]
            value = float(action_predict_max_s)
            self._timing_rollout["action_predict_max_s"] = value if current_max is None else max(current_max, value)
        self._timing_rollout["total_profiled_s"] += float(timing.get("total_profiled_s", 0.0) or 0.0)

    def get_timing_rollout(self) -> Dict[str, Any]:
        infer_count = int(self._timing_rollout["infer_count"])
        avg_infer_s = self._timing_rollout["infer_s"] / infer_count if infer_count > 0 else 0.0
        segment_sums = {
            str(name): float(value)
            for name, value in self._timing_rollout["segment_sums"].items()
        }
        segment_avgs = {
            name: value / infer_count if infer_count > 0 else 0.0
            for name, value in segment_sums.items()
        }
        action_predict_count = int(self._timing_rollout["action_predict_count"])
        action_predict_avg_s = (
            self._timing_rollout["action_predict_total_s"] / action_predict_count
            if action_predict_count > 0
            else 0.0
        )
        return {
            "infer_s": float(self._timing_rollout["infer_s"]),
            "sim_s": float(self._timing_rollout["sim_s"]),
            "infer_count": infer_count,
            "last_infer_s": float(self._timing_rollout["last_infer_s"]),
            "avg_infer_s": float(avg_infer_s),
            "segment_sums": segment_sums,
            "segment_avgs": segment_avgs,
            "action_predict_total_s": float(self._timing_rollout["action_predict_total_s"]),
            "action_predict_count": action_predict_count,
            "action_predict_avg_s": float(action_predict_avg_s),
            "action_predict_min_s": self._timing_rollout["action_predict_min_s"],
            "action_predict_max_s": self._timing_rollout["action_predict_max_s"],
            "total_profiled_s": float(self._timing_rollout["total_profiled_s"]),
        }

    def print_timing_rollout(
        self,
        *,
        episode_idx: Optional[int] = None,
        seed: Optional[int] = None,
        success: Optional[bool] = None,
    ) -> None:
        if not self.timing_enabled:
            return

        timing = self.get_timing_rollout()
        infer_count = int(timing["infer_count"])
        episode_text = self.episode_count if episode_idx is None else episode_idx
        print(
            "[timer] episode_model_infer "
            f"episode={episode_text} "
            f"seed={seed} "
            f"success={success} "
            f"env_steps={self.step_count} "
            f"infer_calls={infer_count} "
            f"total_s={timing['infer_s']:.4f} "
            f"avg_s={timing['avg_infer_s']:.4f} "
            f"sim_s={timing['sim_s']:.4f}",
            flush=True,
        )
        if infer_count == 0 or not timing["segment_sums"]:
            return

        segment_order = [
            "prepare_text_s",
            "encode_image_latents_s",
            "sample_latents_s",
            "prepare_video_timestep_s",
            "video_pre_dit_s",
            "build_attention_mask_s",
            "prefill_video_cache_s",
            "build_action_schedule_s",
            "action_denoise_loop_s",
        ]
        segment_parts = []
        for name in segment_order:
            if name not in timing["segment_sums"]:
                continue
            segment_parts.append(f"{name}_total={timing['segment_sums'][name]:.4f}")
            segment_parts.append(f"{name}_avg={timing['segment_avgs'][name]:.4f}")
        action_min = timing["action_predict_min_s"]
        action_max = timing["action_predict_max_s"]
        print(
            "[timer] episode_infer_segments "
            f"episode={episode_text} "
            f"infer_calls={infer_count} "
            f"action_predict_calls={timing['action_predict_count']} "
            + " ".join(segment_parts)
            + " "
            f"action_predict_total_s={timing['action_predict_total_s']:.4f} "
            f"action_predict_avg_s={timing['action_predict_avg_s']:.4f} "
            f"action_predict_min_s={(0.0 if action_min is None else action_min):.4f} "
            f"action_predict_max_s={(0.0 if action_max is None else action_max):.4f} "
            f"total_profiled_s={timing['total_profiled_s']:.4f}",
            flush=True,
        )

    def reset(self) -> None:
        self.pending_actions.clear()
        self.episode_count += 1
        self.step_count = 0
        self.reset_timing_rollout()


def encode_obs(observation: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return observation


def get_model(usr_args: Dict[str, Any]):
    sim_cfg_path = usr_args.get("sim_cfg_path")
    sim_cfg_name = usr_args.get("sim_cfg_name")
    sim_task = usr_args.get("sim_task")
    cfg = _compose_sim_cfg(
        sim_cfg_path=sim_cfg_path,
        sim_cfg_name=sim_cfg_name,
        sim_task=sim_task,
    )
    _apply_model_overrides(cfg, usr_args)

    checkpoint_path = usr_args.get("ckpt_setting")
    if _is_none_like(checkpoint_path):
        raise ValueError("`ckpt_setting` is required and must be a valid checkpoint path.")

    device = str(usr_args.get("device") or cfg.EVALUATION.get("device") or "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable; fallback device to cpu.")
        device = "cpu"

    mixed_precision = str(usr_args.get("mixed_precision") or cfg.get("mixed_precision", "bf16"))
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)

    dataset_stats_path = _resolve_dataset_stats_path(
        dataset_stats_path=usr_args.get("dataset_stats_path"),
    )

    action_horizon = _parse_optional_int(usr_args.get("action_horizon"))
    if action_horizon is None:
        eval_horizon = _parse_optional_int(cfg.EVALUATION.get("action_horizon"))
        action_horizon = eval_horizon if eval_horizon is not None else int(cfg.data.train.num_frames) - 1
    if action_horizon <= 0:
        raise ValueError(f"`action_horizon` must be positive, got {action_horizon}")

    replan_steps = _parse_optional_int(usr_args.get("replan_steps"))
    if replan_steps is None:
        replan_steps = int(cfg.EVALUATION.get("replan_steps", 8))

    tau_star = _parse_optional_float(usr_args.get("tau_star"))
    if tau_star is None:
        tau_star = float(cfg.EVALUATION.get("tau_star", 0.0))

    num_inference_steps = _parse_optional_int(usr_args.get("num_inference_steps"))
    if num_inference_steps is None:
        num_inference_steps = int(cfg.EVALUATION.get("num_inference_steps", cfg.eval_num_inference_steps))

    sigma_shift = _parse_optional_float(usr_args.get("sigma_shift"))
    if sigma_shift is None:
        sigma_shift = _parse_optional_float(cfg.EVALUATION.get("sigma_shift"))

    seed = _parse_optional_int(usr_args.get("seed"))
    text_cfg_scale = float(usr_args.get("text_cfg_scale", cfg.EVALUATION.get("text_cfg_scale", 1.0)))
    negative_prompt = str(usr_args.get("negative_prompt", cfg.EVALUATION.get("negative_prompt", "")))
    rand_device = str(usr_args.get("rand_device", cfg.EVALUATION.get("rand_device", "cpu")))
    tiled = _parse_bool(usr_args.get("tiled", cfg.EVALUATION.get("tiled", False)))
    timing_enabled = _parse_bool(
        usr_args.get("timing_enabled", cfg.EVALUATION.get("timing_enabled", False))
    )
    robotwin_camera_layout = str(
        usr_args.get(
            "robotwin_camera_layout",
            cfg.EVALUATION.get(
                "robotwin_camera_layout",
                cfg.data.train.get("robotwin_camera_layout", "compact_288x256"),
            ),
        )
    )

    policy = WorldActionRobotWinPolicy(
        model_cfg=cfg.model,
        processor_cfg=cfg.data.train.processor,
        checkpoint_path=str(checkpoint_path),
        dataset_stats_path=dataset_stats_path,
        device=device,
        model_dtype=model_dtype,
        action_horizon=action_horizon,
        replan_steps=replan_steps,
        tau_star=tau_star,
        num_inference_steps=num_inference_steps,
        sigma_shift=sigma_shift,
        seed=seed,
        text_cfg_scale=text_cfg_scale,
        negative_prompt=negative_prompt,
        rand_device=rand_device,
        tiled=tiled,
        timing_enabled=timing_enabled,
        num_video_frames=(int(cfg.data.train.num_frames) - 1) // int(cfg.data.train.action_video_freq_ratio) + 1,
        robotwin_camera_layout=robotwin_camera_layout,
    )
    return policy


def eval(TASK_ENV, model, observation: Optional[Dict[str, Any]]):
    obs = encode_obs(observation)
    model.step(TASK_ENV, obs)


def reset_model(model):
    model.reset()
