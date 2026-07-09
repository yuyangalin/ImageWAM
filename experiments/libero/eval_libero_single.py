import json
import inspect
import importlib
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import hydra
import numpy as np
import torch
from accelerate import PartialState
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

# try:
#     import rootutils

#     rootutils.setup_root(__file__, indicator=".python-version", pythonpath=True)
# except ModuleNotFoundError:
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.libero.libero_utils import (
    LIBERO_ENV_RESOLUTION,
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    invert_gripper_action,
    quat2axisangle,
    save_prediction_video,
    save_rollout_video,
)
from imagewam.datasets.lerobot.processors.imagewam_processor import ImageWAMProcessor
from imagewam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from imagewam.utils.pytorch_utils import set_global_seed
from imagewam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from libero.libero import benchmark
from action_ensembler import ActionEnsembler

OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("max", lambda x: max(x))
OmegaConf.register_new_resolver("split", lambda s, idx: s.split("/")[int(idx)])

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null"}
    return False


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


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


def _resolve_eval_device(cfg: DictConfig) -> str:
    eval_device = cfg.EVALUATION.get("device")
    if eval_device is not None:
        return str(eval_device)
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_path(path_str: str, *, base: Path) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(str(path_str))))
    if not path.is_absolute():
        path = (base / path).resolve()
    return path.resolve()


def _resolve_ckpt_tag(ckpt_path: Path) -> str:
    parts = ckpt_path.resolve().parts
    if "runs" in parts:
        runs_idx = parts.index("runs")
        if runs_idx + 2 >= len(parts):
            raise ValueError(
                f"`ckpt` under runs must follow .../runs/<task>/<date_dir>/..., got: {ckpt_path}"
            )
        task_name = parts[runs_idx + 1]
        date_dir = parts[runs_idx + 2]
        if task_name == "" or date_dir == "":
            raise ValueError(
                f"`ckpt` under runs must follow .../runs/<task>/<date_dir>/..., got: {ckpt_path}"
            )
        return f"{task_name}_{date_dir}"
    return ckpt_path.stem


def _resolve_tagged_output_dir(raw_output_dir: Path, ckpt_tag: str) -> Path:
    run_ts = raw_output_dir.name
    if run_ts == "":
        raise ValueError(f"Invalid EVALUATION.output_dir (missing run timestamp): {raw_output_dir}")

    eval_root = (project_root / "evaluate_results").resolve()
    try:
        relative = raw_output_dir.resolve().relative_to(eval_root)
    except ValueError:
        if raw_output_dir.parent.name == ckpt_tag:
            return raw_output_dir.resolve()
        return (raw_output_dir.parent / ckpt_tag / run_ts).resolve()

    if len(relative.parts) >= 2 and relative.parts[1] == ckpt_tag:
        return raw_output_dir.resolve()

    family = relative.parts[0] if len(relative.parts) > 0 else "libero"
    return (eval_root / family / ckpt_tag / run_ts).resolve()


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
        logging.info("Ignoring unsupported model config keys for %s: %s", target, dropped_keys)

    return OmegaConf.create({key: value for key, value in model_dict.items() if key in allowed_keys})


def _load_model_overrides_path(path_value: Any) -> DictConfig | None:
    if _is_none_like(path_value):
        return None
    path = _resolve_path(str(path_value), base=project_root)
    if not path.exists():
        raise FileNotFoundError(f"model_overrides_path not found: {path}")
    loaded = OmegaConf.load(path)
    if not isinstance(OmegaConf.to_container(loaded, resolve=False), dict):
        raise ValueError(f"model_overrides_path must point to a mapping config, got: {path}")
    return loaded


def _resolve_model_cfg(cfg: DictConfig) -> DictConfig:
    # Keep the full config context while applying overrides so model fields that
    # interpolate into data.* (for example proprio_dim) still resolve correctly.
    cfg_for_resolve = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False, throw_on_missing=False))
    model_cfg = cfg_for_resolve.model

    file_overrides = _load_model_overrides_path(cfg.get("model_overrides_path", None))
    if file_overrides is not None:
        model_cfg = OmegaConf.merge(model_cfg, file_overrides)

    inline_overrides = cfg.get("model_overrides", None)
    if not _is_none_like(inline_overrides):
        model_cfg = OmegaConf.merge(model_cfg, inline_overrides)

    cfg_for_resolve.model = model_cfg
    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg_for_resolve.model, resolve=True))
    return _filter_model_cfg_for_target(model_cfg)


def _save_resolved_eval_config(cfg: DictConfig, model_cfg: DictConfig, output_file: Path) -> None:
    resolved_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False))
    resolved_cfg.model_effective = OmegaConf.create(OmegaConf.to_container(model_cfg, resolve=True))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=resolved_cfg, f=str(output_file))


def _resolve_dataset_stats_path(cfg: DictConfig) -> Path:
    explicit = cfg.EVALUATION.get("dataset_stats_path")
    candidates: list[Path] = []

    if explicit is not None:
        candidates.append(Path(os.path.expanduser(os.path.expandvars(str(explicit)))))

    ckpt = Path(os.path.expanduser(os.path.expandvars(str(cfg.ckpt))))
    for parent in list(ckpt.parents)[:4]:
        candidates.append(parent / "dataset_stats.json")

    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved

    msg = (
        "Failed to locate dataset_stats.json. Tried explicit "
        "EVALUATION.dataset_stats_path and checkpoint parent directories. "
        "Please pass EVALUATION.dataset_stats_path=/path/to/dataset_stats.json."
    )
    raise FileNotFoundError(msg)


def _load_task_choices(cfg: DictConfig) -> list[tuple[str, int]]:
    task_chunk_file = cfg.EVALUATION.get("task_chunk_file", None)
    if task_chunk_file is None or str(task_chunk_file).strip() == "":
        return [(str(cfg.EVALUATION.task_suite_name), int(cfg.EVALUATION.task_id))]

    chunk_path = _resolve_path(str(task_chunk_file), base=project_root)
    if not chunk_path.exists():
        raise FileNotFoundError(f"EVALUATION.task_chunk_file not found: {chunk_path}")

    choices: list[tuple[str, int]] = []
    with chunk_path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if line == "" or line.startswith("#"):
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 2 or parts[0] == "" or parts[1] == "":
                raise ValueError(
                    f"Invalid task chunk line {line_no} in {chunk_path}: {raw_line.rstrip()!r}. "
                    "Expected format: suite_name,task_id"
                )
            choices.append((parts[0], int(parts[1])))

    if not choices:
        raise ValueError(f"EVALUATION.task_chunk_file is empty: {chunk_path}")
    return choices


def _load_model_checkpoint(model: torch.nn.Module, ckpt: str) -> None:
    model.load_checkpoint(ckpt)
    logging.info("Loaded checkpoint via model.load_checkpoint: %s", ckpt)
    return

    # deprecated legacy checkpoint loading
    payload = torch.load(ckpt, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Legacy checkpoint payload must be dict, got: {type(payload)}")

    if "mot" in payload and hasattr(model, "mot"):
        missing, unexpected = model.mot.load_state_dict(payload["mot"], strict=False)
        logging.warning(
            "Loaded fallback `mot` state_dict with strict=False. Missing=%d Unexpected=%d",
            len(missing),
            len(unexpected),
        )
        return

    state_dict = None
    for key in ("model_state_dict", "state_dict", "model"):
        value = payload.get(key)
        if isinstance(value, dict):
            state_dict = value
            break
    if state_dict is None and all(torch.is_tensor(v) for v in payload.values()):
        state_dict = payload
    if state_dict is None:
        raise ValueError(f"Cannot parse legacy checkpoint keys from: {ckpt}")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logging.warning(
        "Loaded fallback model state_dict with strict=False. Missing=%d Unexpected=%d",
        len(missing),
        len(unexpected),
    )


def _center_crop_resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    pil_image = Image.fromarray(image)
    src_w, src_h = pil_image.size
    scale = max(width / src_w, height / src_h)
    resized = pil_image.resize((round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR)
    rw, rh = resized.size
    left = max((rw - width) // 2, 0)
    top = max((rh - height) // 2, 0)
    cropped = resized.crop((left, top, left + width, top + height))
    return np.asarray(cropped, dtype=np.uint8)


def _normalize_proprio(
    proprio: np.ndarray,
    processor: ImageWAMProcessor,
) -> torch.Tensor:
    state_meta = processor.shape_meta["state"]
    if len(state_meta) != 1:
        raise ValueError(
            "LIBERO eval currently expects a single merged state key in shape_meta['state']."
        )
    state_key = state_meta[0]["key"]

    state_batch = {"state": {state_key: torch.as_tensor(proprio, dtype=torch.float32).unsqueeze(0)}}
    state_batch = processor.action_state_transform(state_batch)
    state_batch = processor.normalizer.forward(state_batch)
    return state_batch["state"][state_key]


def _obs_to_model_input(
    obs: dict,
    cfg: DictConfig,
    processor: ImageWAMProcessor,
    width: int,
    height: int,
    device: str,
    dtype: torch.dtype,
):
    imgs = get_libero_image(obs)
    image_meta = processor.shape_meta["images"]
    if len(image_meta) < int(processor.num_output_cameras):
        raise ValueError(
            f"shape_meta.images has {len(image_meta)} entries, "
            f"but num_output_cameras={processor.num_output_cameras}."
        )

    def _meta_to_hw(meta: dict, camera_idx: int) -> tuple[int, int]:
        shape = meta["shape"]
        if len(shape) != 3:
            raise ValueError(f"shape_meta.images[{camera_idx}].shape must be [C,H,W], got {shape}")
        return int(shape[1]), int(shape[2])

    concatenation = cfg.data.train.get("concat_multi_camera", "horizontal")
    num_cameras = processor.num_output_cameras
    if num_cameras == 1:
        primary_h, primary_w = _meta_to_hw(image_meta[0], camera_idx=0)
        rgb = _center_crop_resize(imgs["image"], width=primary_w, height=primary_h)
    elif num_cameras == 2:
        primary_h, primary_w = _meta_to_hw(image_meta[0], camera_idx=0)
        wrist_h, wrist_w = _meta_to_hw(image_meta[1], camera_idx=1)
        primary = _center_crop_resize(imgs["image"], width=primary_w, height=primary_h)
        wrist = _center_crop_resize(imgs["wrist_image"], width=wrist_w, height=wrist_h)
        if concatenation == "horizontal":
            rgb = np.concatenate([primary, wrist], axis=1)
        elif concatenation == "vertical":
            rgb = np.concatenate([primary, wrist], axis=0)
        else:
            raise ValueError(f"Invalid concat_multi_camera: {concatenation}")
    else:
        raise ValueError(f"LIBERO eval currently supports num_output_cameras in [1, 2], got {num_cameras}.")

    actual_h, actual_w = int(rgb.shape[0]), int(rgb.shape[1])
    expected_h, expected_w = int(height), int(width)
    image_shapes = [meta["shape"] for meta in image_meta]
    assert actual_h == expected_h and actual_w == expected_w, (
        "Input image size mismatch after per-camera resize + concat: "
        f"got (H,W)=({actual_h},{actual_w}), expected (H,W)=({expected_h},{expected_w}) "
        f"from data.train.video_size={[expected_h, expected_w]}; "
        f"shape_meta.images={image_shapes}, concat_multi_camera={concatenation}."
    )

    x = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
    x = x * (2.0 / 255.0) - 1.0

    proprio = _normalize_proprio(_extract_sim_state(obs), processor)

    return x, proprio, imgs


def _extract_sim_state(obs: dict) -> np.ndarray:
    """Build simulator state from current observation.

    This is used as proprio input for model inference.
    """
    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    ).astype(np.float32)
    return state


def _denormalize_action(action: torch.Tensor, processor: ImageWAMProcessor) -> np.ndarray:
    if action.ndim == 2:
        action = action.unsqueeze(0)
    if action.ndim != 3:
        raise ValueError(f"Expected action tensor [B, T, D], got {tuple(action.shape)}")

    action_meta = processor.shape_meta["action"]
    if len(action_meta) != 1:
        raise ValueError(
            "LIBERO eval currently expects a single merged action key in shape_meta['action']."
        )

    action_key = action_meta[0]["key"]
    normalizer = processor.normalizer.normalizers["action"][action_key]
    action = action.to(dtype=torch.float32, device="cpu")
    denorm = normalizer.backward(action)
    return denorm.numpy()


def _get_num_video_frames(cfg: DictConfig) -> int:
    return (int(cfg.data.train.num_frames) - 1) // int(cfg.data.train.action_video_freq_ratio) + 1


def _validate_visualize_future_video_cfg(cfg: DictConfig) -> None:
    if not bool(cfg.EVALUATION.get("visualize_future_video", False)):
        return

    action_conditioned = cfg.model.video_dit_config.get("action_conditioned", None)
    if action_conditioned is not False:
        raise ValueError(
            "EVALUATION.visualize_future_video=true requires "
            "model.video_dit_config.action_conditioned=false."
        )


def _select_predicted_future_frames(pred_video: list[Image.Image], cfg: DictConfig) -> list[Image.Image]:
    if len(pred_video) == 0:
        raise ValueError("`infer_joint` returned an empty predicted video.")

    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    action_video_freq_ratio = int(cfg.data.train.action_video_freq_ratio)
    num_future_frames = replan_steps // action_video_freq_ratio
    keep_frames = 1 + num_future_frames
    return list(pred_video[:keep_frames])


def _get_future_frame_capture_steps(cfg: DictConfig) -> list[int]:
    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    action_video_freq_ratio = int(cfg.data.train.action_video_freq_ratio)
    num_future_frames = replan_steps // action_video_freq_ratio
    return [step_idx * action_video_freq_ratio for step_idx in range(num_future_frames + 1)]


def _frame_to_rgb_array(frame: Any) -> np.ndarray:
    if isinstance(frame, dict):
        images = []
        for value in frame.values():
            value_array = np.array(value) if isinstance(value, Image.Image) else np.array(value, copy=True)
            images.append(value_array)
        return np.concatenate(images, axis=1)
    if isinstance(frame, Image.Image):
        return np.array(frame.convert("RGB"))
    return np.array(frame, copy=True)


def _compute_clip_mean_psnr(
    gt_frames: list[Any],
    pred_frames: list[Any],
    eps: float = 1e-8,
) -> Optional[float]:
    if len(gt_frames) == 0 or len(pred_frames) == 0:
        return None
    assert len(gt_frames) == len(pred_frames), (
        "GT/pred frame count mismatch for PSNR: "
        f"len(gt_frames)={len(gt_frames)} len(pred_frames)={len(pred_frames)}. "
        "This indicates temporal misalignment in future-video capture."
    )
    num_frames = len(gt_frames)

    frame_psnr_values = []
    for gt_frame, pred_frame in zip(gt_frames[:num_frames], pred_frames[:num_frames]):
        gt_image = _frame_to_rgb_array(gt_frame)
        pred_image = _frame_to_rgb_array(pred_frame)
        target_h, target_w = pred_image.shape[:2]
        if gt_image.shape[:2] != (target_h, target_w):
            gt_image = np.array(
                Image.fromarray(gt_image).resize((target_w, target_h), resample=Image.BILINEAR)
            )

        gt_f32 = gt_image.astype(np.float32)
        pred_f32 = pred_image.astype(np.float32)
        mse = float(np.mean((pred_f32 - gt_f32) ** 2))
        psnr = 10.0 * np.log10((255.0 * 255.0) / max(mse, eps))
        frame_psnr_values.append(float(psnr))

    if len(frame_psnr_values) == 0:
        return None
    return float(np.mean(frame_psnr_values))


def _predict_action_chunk(
    obs: dict,
    task_description: str,
    model: torch.nn.Module,
    processor: ImageWAMProcessor,
    cfg: DictConfig,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
) -> tuple[np.ndarray, dict, Optional[list[Image.Image]]]:
    num_inference_steps_cfg = cfg.EVALUATION.get("num_inference_steps", None)
    if num_inference_steps_cfg is None:
        num_inference_steps = int(cfg.get("eval_num_inference_steps", 20))
    else:
        num_inference_steps = int(num_inference_steps_cfg)
    prompt_template = DEFAULT_PROMPT
    prompt = prompt_template.format(task=task_description)

    image, proprio, imgs = _obs_to_model_input(
        obs,
        cfg=cfg,
        processor=processor,
        width=input_w,
        height=input_h,
        device=model_device,
        dtype=model.torch_dtype,
    )

    infer_kwargs = {
        "prompt": prompt,
        "input_image": image,
        "action_horizon": action_horizon,
        "negative_prompt": str(cfg.EVALUATION.get("negative_prompt", "")),
        "text_cfg_scale": float(cfg.EVALUATION.get("text_cfg_scale", 1.0)),
        "tau_star": float(cfg.EVALUATION.get("tau_star", 0.0)),
        "num_inference_steps": num_inference_steps,
        "proprio": proprio,
        "sigma_shift": (
            None
            if cfg.EVALUATION.get("sigma_shift") is None
            else float(cfg.EVALUATION.get("sigma_shift"))
        ),
        "seed": None if cfg.get("seed") is None else int(cfg.seed),
        "rand_device": str(cfg.EVALUATION.get("rand_device", "cpu")),
        "tiled": bool(cfg.EVALUATION.get("tiled", False)),
    }
    visualize_future_video = bool(cfg.EVALUATION.get("visualize_future_video", False))
    predicted_future_frames = None
    if visualize_future_video:
        infer_kwargs["num_video_frames"] = _get_num_video_frames(cfg)
    elif "num_video_frames" in inspect.signature(model.infer_action).parameters:
        infer_kwargs["num_video_frames"] = _get_num_video_frames(cfg)

    with torch.no_grad():
        if visualize_future_video:
            pred = model.infer_joint(**infer_kwargs)
            predicted_future_frames = _select_predicted_future_frames(pred["video"], cfg)
        else:
            pred = model.infer_action(**infer_kwargs)
    action = pred["action"]  # [T, D]

    action = _denormalize_action(action, processor)[0]  # [T, D]

    # The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    action[..., -1] = action[..., -1] * 2 - 1
    action = invert_gripper_action(action)
    if bool(cfg.EVALUATION.get("binarize_gripper", False)):
        action[..., -1] = np.sign(action[..., -1])
    return action, imgs, predicted_future_frames


def _get_max_steps(task_suite_name: str) -> int:
    suite_steps = {
        "libero_spatial": 400,
        "libero_object": 400,
        "libero_goal": 400,
        "libero_10": 700,
        "libero_90": 700,
    }
    if task_suite_name not in suite_steps:
        raise ValueError(f"Unknown task suite: {task_suite_name}")
    return suite_steps[task_suite_name]


def run_single_episode(
    env,
    initial_state,
    task_description: str,
    model: torch.nn.Module,
    processor: ImageWAMProcessor,
    cfg: DictConfig,
    episode_idx: int,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
) -> tuple[bool, list, list[dict[str, Any]], Optional[float]]:
    max_steps = _get_max_steps(cfg.EVALUATION.task_suite_name)
    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    num_steps_wait = int(cfg.EVALUATION.get("num_steps_wait", 5))
    use_action_ensembler = bool(cfg.EVALUATION.get("use_action_ensembler", False))
    visualize_future_video = bool(cfg.EVALUATION.get("visualize_future_video", False))
    capture_steps = set(_get_future_frame_capture_steps(cfg)[1:])

    env.reset()
    obs = env.set_init_state(initial_state)
    if use_action_ensembler:
        ensembler = ActionEnsembler()
        ensembler.reset()

    replay_images = []
    predicted_future_video_clips: list[dict[str, Any]] = []
    episode_future_clip_psnr: list[float] = []
    pending_actions: list[list[float]] = []
    current_predicted_future_clip: Optional[dict[str, Any]] = None
    current_replan_step = 0
    current_replan_idx = -1

    t = 0
    done = False
    pbar = tqdm(total=max_steps + num_steps_wait, desc=f"Episode {episode_idx + 1}")
    while t < max_steps + num_steps_wait:
        pbar.update(1)
        if t < num_steps_wait:
            obs, _, done, _ = env.step(get_libero_dummy_action())
            t += 1
            continue

        if len(pending_actions) == 0:
            action_chunk, imgs, predicted_future_frames = _predict_action_chunk(
                obs=obs,
                task_description=task_description,
                model=model,
                processor=processor,
                cfg=cfg,
                action_horizon=action_horizon,
                input_w=input_w,
                input_h=input_h,
                model_device=model_device,
            )
            if predicted_future_frames is not None:
                current_replan_idx += 1
                current_predicted_future_clip = {
                    "replan_idx": current_replan_idx,
                    "gt_frames": [imgs.copy()],
                    "pred_frames": predicted_future_frames,
                }
            else:
                current_predicted_future_clip = None
            current_replan_step = 0
            if use_action_ensembler:
                ensembler.add_actions(action_chunk, t)
                pending_actions = [ensembler.get_action(ts).tolist() for ts in range(t, t + replan_steps)]
            else:
                pending_actions = action_chunk[:replan_steps].tolist()
            replay_images.append(imgs.copy())
        else:
            imgs = get_libero_image(obs)
            replay_images.append(imgs.copy())

        obs, _, done, _ = env.step(pending_actions.pop(0))
        if visualize_future_video and current_predicted_future_clip is not None:
            current_replan_step += 1
            if current_replan_step in capture_steps:
                current_predicted_future_clip["gt_frames"].append(get_libero_image(obs))
            if done or len(pending_actions) == 0:
                expected_frame_count = 1 + sum(
                    1 for capture_step in capture_steps if capture_step <= current_replan_step
                )
                gt_len = len(current_predicted_future_clip["gt_frames"])
                pred_len = len(current_predicted_future_clip["pred_frames"])
                assert gt_len == expected_frame_count, (
                    "GT future frames do not match expected capture count: "
                    f"gt_len={gt_len} expected={expected_frame_count} "
                    f"episode={episode_idx} replan={current_predicted_future_clip['replan_idx']} "
                    f"current_replan_step={current_replan_step} capture_steps={sorted(capture_steps)}."
                )
                assert pred_len >= expected_frame_count, (
                    "Predicted future frames shorter than expected capture count: "
                    f"pred_len={pred_len} expected={expected_frame_count} "
                    f"episode={episode_idx} replan={current_predicted_future_clip['replan_idx']}."
                )
                if pred_len != expected_frame_count:
                    logging.info(
                        "Align predicted clip length to executed steps: "
                        "episode=%s replan=%s done=%s expected=%s pred_full=%s",
                        episode_idx,
                        current_predicted_future_clip["replan_idx"],
                        done,
                        expected_frame_count,
                        pred_len,
                    )
                current_predicted_future_clip["pred_frames"] = current_predicted_future_clip["pred_frames"][
                    :expected_frame_count
                ]
                assert len(current_predicted_future_clip["gt_frames"]) == len(
                    current_predicted_future_clip["pred_frames"]
                ), (
                    "GT/pred frame count mismatch after alignment: "
                    f"len(gt_frames)={len(current_predicted_future_clip['gt_frames'])} "
                    f"len(pred_frames)={len(current_predicted_future_clip['pred_frames'])} "
                    f"episode={episode_idx} replan={current_predicted_future_clip['replan_idx']}."
                )
                clip_psnr = _compute_clip_mean_psnr(
                    current_predicted_future_clip["gt_frames"],
                    current_predicted_future_clip["pred_frames"],
                )
                if clip_psnr is not None:
                    episode_future_clip_psnr.append(clip_psnr)
                predicted_future_video_clips.append(current_predicted_future_clip)
                current_predicted_future_clip = None
        if done:
            break
        t += 1
    pbar.close()

    episode_mean_psnr = (
        float(np.mean(episode_future_clip_psnr)) if len(episode_future_clip_psnr) > 0 else None
    )
    return bool(done), replay_images, predicted_future_video_clips, episode_mean_psnr


def run_single_task(
    task,
    initial_states,
    model: torch.nn.Module,
    processor: ImageWAMProcessor,
    cfg: DictConfig,
    video_dir: Path,
    predicted_video_dir: Path,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
) -> dict:
    env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, cfg.get("seed"))
    visualize_future_video = bool(cfg.EVALUATION.get("visualize_future_video", False))
    results = {
        "successes": 0,
        "failure_episodes": [],
        "success_episodes": [],
        "task_description": task_description,
    }
    if visualize_future_video:
        results["episode_future_video_psnr"] = []
        results["future_video_psnr_mean"] = None

    for trial_idx in range(int(cfg.EVALUATION.num_trials)):
        success, replay_images, predicted_future_video_clips, episode_mean_psnr = run_single_episode(
            env=env,
            initial_state=initial_states[trial_idx],
            task_description=task_description,
            model=model,
            processor=processor,
            cfg=cfg,
            episode_idx=trial_idx,
            action_horizon=action_horizon,
            input_w=input_w,
            input_h=input_h,
            model_device=model_device,
        )
        if success:
            results["successes"] += 1
            results["success_episodes"].append(trial_idx)
        else:
            results["failure_episodes"].append(trial_idx)
        if visualize_future_video:
            results["episode_future_video_psnr"].append(episode_mean_psnr)

        save_rollout_video(
            video_dir,
            replay_images,
            f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
            success=success,
            task_description=task_description,
        )
        if visualize_future_video:
            if len(predicted_future_video_clips) == 0:
                logging.warning(
                    "No predicted future frames collected for task %s trial %s.",
                    cfg.EVALUATION.task_id,
                    trial_idx,
                )
            else:
                all_gt_frames = []
                all_pred_frames = []
                for clip in predicted_future_video_clips:
                    all_gt_frames.extend(clip["gt_frames"])
                    all_pred_frames.extend(clip["pred_frames"])
                    save_prediction_video(
                        predicted_video_dir,
                        clip["gt_frames"],
                        clip["pred_frames"],
                        f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
                        clip["replan_idx"],
                        success=success,
                        task_description=task_description,
                    )
                save_prediction_video(
                    predicted_video_dir,
                    all_gt_frames,
                    all_pred_frames,
                    f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
                    "all",
                    success=success,
                    task_description=task_description,
                )

    if visualize_future_video:
        valid_episode_psnr = [x for x in results["episode_future_video_psnr"] if x is not None]
        if len(valid_episode_psnr) > 0:
            results["future_video_psnr_mean"] = float(np.mean(valid_episode_psnr))
    return results


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero.yaml")
def eval_single_process(cfg: DictConfig):
    start_time = time.time()
    partial_state = PartialState()
    partial_state.config = cfg

    if cfg.get("seed") is not None:
        set_global_seed(int(cfg.seed), get_worker_init_fn=False)

    if cfg.ckpt is None:
        raise ValueError("cfg.ckpt must not be None.")
    _validate_visualize_future_video_cfg(cfg)

    ckpt_path = _resolve_path(str(cfg.ckpt), base=project_root)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt_tag = _resolve_ckpt_tag(ckpt_path)
    output_dir = _resolve_tagged_output_dir(
        _resolve_path(str(cfg.EVALUATION.output_dir), base=project_root),
        ckpt_tag,
    )
    cfg.EVALUATION.output_dir = str(output_dir)

    env_num = int(cfg.EVALUATION.get("env_num", 1))
    if env_num != 1:
        raise ValueError(
            "Only env_num=1 is supported in eval_libero_single.py. "
            "Use run_libero_manager/run_libero_parallel_test.sh for multi-GPU task parallelism."
        )

    model_device = _resolve_eval_device(cfg)
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model_cfg = _resolve_model_cfg(cfg)
    model = instantiate(model_cfg, model_dtype=model_dtype, device=model_device)
    _load_model_checkpoint(model, str(ckpt_path))
    model = model.to(model_device).eval()

    dataset_stats_path = _resolve_dataset_stats_path(cfg)
    dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
    processor: ImageWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)
    logging.info("Using dataset stats: %s", dataset_stats_path)

    action_horizon_cfg = cfg.EVALUATION.get("action_horizon", None)
    if action_horizon_cfg is None:
        action_horizon = int(cfg.data.train.num_frames) - 1
    else:
        action_horizon = int(action_horizon_cfg)
    if action_horizon <= 0:
        raise ValueError(f"EVALUATION.action_horizon must be positive, got {action_horizon}")

    video_size = cfg.data.train.get("video_size", [224, 224])
    if len(video_size) != 2:
        raise ValueError(f"data.train.video_size must be [H, W], got {video_size}")
    input_h = int(video_size[0])
    input_w = int(video_size[1])
    concat_multi_camera = cfg.data.train.get("concat_multi_camera", None)
    shape_meta_images = [meta["shape"] for meta in processor.shape_meta["images"]]

    benchmark_dict = benchmark.get_benchmark_dict()
    task_choices = _load_task_choices(cfg)
    chunk_mode = len(task_choices) > 1
    all_results = []

    for suite_name, task_id in task_choices:
        task_start_time = time.time()
        cfg.EVALUATION.task_suite_name = suite_name
        cfg.EVALUATION.task_id = int(task_id)

        local_log_dir = Path(cfg.EVALUATION.output_dir)
        local_log_dir.mkdir(parents=True, exist_ok=True)
        video_dir = local_log_dir / cfg.EVALUATION.task_suite_name / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        predicted_video_dir = local_log_dir / cfg.EVALUATION.task_suite_name / "predicted_videos"
        if bool(cfg.EVALUATION.get("visualize_future_video", False)):
            predicted_video_dir.mkdir(parents=True, exist_ok=True)
        task_output_dir = local_log_dir / cfg.EVALUATION.task_suite_name
        _save_resolved_eval_config(
            cfg,
            model_cfg,
            task_output_dir / f"eval_config_gpu{cfg.gpu_id}_task{cfg.EVALUATION.task_id}.yaml",
        )

        task_suite = benchmark_dict[cfg.EVALUATION.task_suite_name]()
        task = task_suite.get_task(cfg.EVALUATION.task_id)
        initial_states = task_suite.get_task_init_states(cfg.EVALUATION.task_id)

        while len(initial_states) < int(cfg.EVALUATION.num_trials):
            initial_states.extend(initial_states[: (int(cfg.EVALUATION.num_trials) - len(initial_states))])

        results = {
            "task_suite": cfg.EVALUATION.task_suite_name,
            "task_id": cfg.EVALUATION.task_id,
            "task_description": None,
            "successes": 0,
            "total_episodes": int(cfg.EVALUATION.num_trials),
            "gpu_id": int(cfg.gpu_id),
            "success_episodes": [],
            "failure_episodes": [],
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration": 0,
        }

        logging.info(
            "Running LIBERO evaluation with env_num=1 suite=%s task_id=%s chunk=%s/%s",
            cfg.EVALUATION.task_suite_name,
            cfg.EVALUATION.task_id,
            len(all_results) + 1,
            len(task_choices),
        )
        task_results = run_single_task(
            task=task,
            initial_states=initial_states,
            model=model,
            processor=processor,
            cfg=cfg,
            video_dir=video_dir,
            predicted_video_dir=predicted_video_dir,
            action_horizon=action_horizon,
            input_w=input_w,
            input_h=input_h,
            model_device=model_device,
        )
        results.update(task_results)

        results["duration"] = time.time() - (task_start_time if chunk_mode else start_time)
        output_dir = Path(cfg.EVALUATION.output_dir) / cfg.EVALUATION.task_suite_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"gpu{cfg.gpu_id}_task{cfg.EVALUATION.task_id}_results.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, cls=NumpyEncoder)

        print(
            f"Task {cfg.EVALUATION.task_suite_name}/{cfg.EVALUATION.task_id} completed: "
            f"{results['successes']}/{cfg.EVALUATION.num_trials} successes"
        )
        if results.get("future_video_psnr_mean") is not None:
            print(f"Task {cfg.EVALUATION.task_id} future-video PSNR mean: {results['future_video_psnr_mean']:.4f}")
        print(f"Time taken: {results['duration']:.2f} seconds")
        all_results.append(results)

    return all_results[0] if len(all_results) == 1 else all_results


if __name__ == "__main__":
    eval_single_process()
