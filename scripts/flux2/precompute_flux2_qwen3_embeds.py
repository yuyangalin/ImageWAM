#!/usr/bin/env python3
"""Precompute FLUX.2 Qwen3 text embeddings.

Example:
    torchrun --standalone --nproc_per_node=8 scripts/flux2/precompute_flux2_qwen3_embeds.py \
        task=libero_flux2_imagewam \
        data.train.dataset_dirs=[...] \
        data.train.qwen_text_cache_dir=./qwen3_flux2_cache \
        data.train.qwen_context_len=512 \
        data.train.qwen_text_cache_format=qwen3_flux2 \
        model.variant=klein-base-4b
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import hydra
import torch
from einops import rearrange
from omegaconf import DictConfig, ListConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from imagewam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from imagewam.models.backbones.flux2_imports import ensure_flux2_importable
from imagewam.utils.config_resolvers import register_default_resolvers
from imagewam.utils.logging_config import get_logger, setup_logging

register_default_resolvers()
logger = get_logger(__name__)


def _get_rank_info() -> tuple[int, int]:
    return int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0"))), int(os.environ.get("WORLD_SIZE", "1"))


def _iter_dataset_nodes(node: Any, path: str = "data"):
    if isinstance(node, DictConfig):
        if "dataset_dirs" in node and node.get("dataset_dirs") is not None:
            yield path, node
        for key, value in node.items():
            yield from _iter_dataset_nodes(value, f"{path}.{key}")
    elif isinstance(node, ListConfig):
        for idx, value in enumerate(node):
            yield from _iter_dataset_nodes(value, f"{path}[{idx}]")


def _instruction_to_prompt(instruction: Any) -> str | None:
    task = str(instruction).strip()
    if not task:
        return None
    return DEFAULT_PROMPT.format(task=task)


def _configured_prompts(node: DictConfig) -> list[str]:
    prompts: list[str] = []
    override_instruction = node.get("override_instruction")
    if override_instruction is not None:
        prompt = _instruction_to_prompt(override_instruction)
        return [prompt] if prompt is not None else []
    fallback_instructions = node.get("fallback_instructions")
    if not fallback_instructions:
        return prompts
    for instruction in fallback_instructions.values():
        prompt = _instruction_to_prompt(instruction)
        if prompt is not None:
            prompts.append(prompt)
    return prompts


def _collect_dataset_settings(data_cfg: DictConfig) -> tuple[list[str], list[Path], set[int], list[str]]:
    dataset_dirs: list[str] = []
    cache_dirs: list[Path] = []
    context_lens: set[int] = set()
    config_prompts: list[str] = []
    seen_prompts: set[str] = set()
    for node_path, node in _iter_dataset_nodes(data_cfg):
        raw_dirs = node.get("dataset_dirs")
        if raw_dirs is None:
            continue
        cache_dir = node.get("qwen_text_cache_dir")
        if cache_dir is None or not str(cache_dir).strip():
            raise ValueError(f"Missing `qwen_text_cache_dir` for dataset node `{node_path}`.")
        cache_path = Path(str(cache_dir)).expanduser()
        if cache_path not in cache_dirs:
            cache_dirs.append(cache_path)
        context_lens.add(int(node.get("qwen_context_len", 512)))
        for ds in raw_dirs:
            ds_str = str(ds)
            if ds_str not in dataset_dirs:
                dataset_dirs.append(ds_str)
        for prompt in _configured_prompts(node):
            if prompt not in seen_prompts:
                seen_prompts.add(prompt)
                config_prompts.append(prompt)
    return dataset_dirs, cache_dirs, context_lens, config_prompts


def _read_unique_prompts(dataset_dirs: list[str], initial_prompts: list[str] | None = None) -> list[str]:
    prompts: list[str] = []
    seen: set[str] = set()
    for prompt in initial_prompts or []:
        if prompt not in seen:
            seen.add(prompt)
            prompts.append(prompt)
    for dataset_dir in dataset_dirs:
        tasks_path = Path(dataset_dir) / "meta" / "tasks.jsonl"
        tasks_parquet_path = Path(dataset_dir) / "meta" / "tasks.parquet"
        if tasks_path.exists():
            with tasks_path.open("r", encoding="utf-8") as f:
                task_records = []
                for line_idx, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if "task" not in record:
                        raise KeyError(f"Missing `task` field at {tasks_path}:{line_idx}")
                    task_records.append(str(record["task"]))
        elif tasks_parquet_path.exists():
            import pyarrow.parquet as pq

            table = pq.read_table(tasks_parquet_path)
            column_names = set(table.column_names)
            if "task" in column_names:
                task_records = [str(value) for value in table["task"].to_pylist()]
            elif "name" in column_names:
                task_records = [str(value) for value in table["name"].to_pylist()]
            elif "__index_level_0__" in column_names:
                task_records = [str(value) for value in table["__index_level_0__"].to_pylist()]
            else:
                raise KeyError(
                    f"Missing `task`, `name`, or `__index_level_0__` column in LeRobot v3 tasks file: {tasks_parquet_path}"
                )
        else:
            raise FileNotFoundError(f"Missing tasks file: {tasks_path} or {tasks_parquet_path}")

        for task in task_records:
            prompt = DEFAULT_PROMPT.format(task=task)
            if prompt not in seen:
                seen.add(prompt)
                prompts.append(prompt)
    return prompts


def _atomic_torch_save(payload: dict[str, torch.Tensor], path: Path) -> None:
    tmp = path.parent / f".{path.name}.tmp.{os.getpid()}"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _variant_to_model_spec(variant: str) -> str:
    key = str(variant).lower().replace("_", "-")
    if key in {"klein-base-4b", "flux.2-klein-base-4b", "4b", "base-4b"}:
        return "Qwen/Qwen3-4B"
    if key in {"klein-base-9b", "flux.2-klein-base-9b", "9b", "base-9b"}:
        return "Qwen/Qwen3-8B"
    raise ValueError(f"Unsupported FLUX.2 variant for Qwen3 cache: {variant!r}")


@hydra.main(config_path="../../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_logging(log_level=logging.INFO)
    if cfg.data is None:
        raise ValueError("`cfg.data` is required.")
    if cfg.model is None:
        raise ValueError("`cfg.model` is required.")

    local_rank, world_size = _get_rank_info()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    is_main = local_rank == 0

    dataset_dirs, cache_dirs, context_lens, config_prompts = _collect_dataset_settings(cfg.data)
    if len(context_lens) != 1:
        raise ValueError(f"Expected one qwen_context_len, got {sorted(context_lens)}")
    context_len = next(iter(context_lens))
    if context_len != 512 and is_main:
        logger.warning("FLUX.2 official Qwen3 text cache uses max_length=512; got qwen_context_len=%d.", context_len)

    prompts = _read_unique_prompts(dataset_dirs, initial_prompts=config_prompts)
    my_prompts = prompts[local_rank::world_size]
    variant = str(cfg.model.get("variant", "klein-base-4b"))
    model_spec = cfg.get("flux2_qwen3_model_spec")
    if model_spec is None:
        model_spec = cfg.model.get("qwen3_model_spec")
    if model_spec is None:
        model_spec = _variant_to_model_spec(variant)
    model_spec = str(model_spec)
    flux2_src_path = cfg.model.get("flux2_src_path") or os.environ.get("FLUX2_SRC")
    if not flux2_src_path:
        raise ValueError("Set model.flux2_src_path or the FLUX2_SRC environment variable.")
    ensure_flux2_importable(str(flux2_src_path))
    from flux2.text_encoder import OUTPUT_LAYERS_QWEN3

    if is_main:
        logger.info("Caching FLUX.2 Qwen3 embeddings with %s, prompts=%d, len=%d", model_spec, len(prompts), context_len)
    tokenizer = AutoTokenizer.from_pretrained(model_spec)
    model_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_spec, torch_dtype=model_dtype).to(device).eval()

    batch_size = int(cfg.get("qwen_cache_batch_size", 16))
    overwrite = bool(cfg.get("qwen_cache_overwrite", False))
    save_workers = int(cfg.get("qwen_cache_save_workers", 4))
    for cache_dir in cache_dirs:
        cache_dir.mkdir(parents=True, exist_ok=True)

    def _save_one(prompt: str, hidden_i: torch.Tensor, mask_i: torch.Tensor) -> None:
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        payload = {
            "text_hidden_states": hidden_i.clone(),
            "text_attention_mask": mask_i.to(dtype=torch.bool).clone(),
        }
        for cache_dir in cache_dirs:
            _atomic_torch_save(payload, cache_dir / f"{hashed}.qwen3_flux2_len{context_len}.pt")

    written = 0
    skipped = 0
    total_fwd_s = 0.0
    with torch.no_grad(), concurrent.futures.ThreadPoolExecutor(max_workers=save_workers) as pool:
        futs: list[concurrent.futures.Future] = []
        pbar = tqdm(range(0, len(my_prompts), batch_size), desc=f"rank{local_rank}", disable=not is_main)
        for start in pbar:
            batch_prompts = my_prompts[start : start + batch_size]
            needed = []
            for prompt in batch_prompts:
                hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                exists_everywhere = all(
                    (cache_dir / f"{hashed}.qwen3_flux2_len{context_len}.pt").exists()
                    for cache_dir in cache_dirs
                )
                if exists_everywhere and not overwrite:
                    skipped += 1
                else:
                    needed.append(prompt)
            if not needed:
                continue

            rendered = []
            for prompt in needed:
                messages = [{"role": "user", "content": prompt}]
                rendered.append(
                    tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                )
            encoded = tokenizer(
                rendered,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=context_len,
            )
            input_ids = encoded.input_ids.to(device)
            attention_mask = encoded.attention_mask.to(device)
            t0 = time.perf_counter()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            hidden = torch.stack([outputs.hidden_states[k] for k in OUTPUT_LAYERS_QWEN3], dim=1)
            hidden = rearrange(hidden, "b c l d -> b l (c d)")
            hidden_cpu = hidden.detach().to(device="cpu", dtype=torch.bfloat16)
            mask_cpu = attention_mask.detach().cpu()
            total_fwd_s += time.perf_counter() - t0
            for i, prompt in enumerate(needed):
                futs.append(pool.submit(_save_one, prompt, hidden_cpu[i], mask_cpu[i]))
                written += 1
        for fut in concurrent.futures.as_completed(futs):
            fut.result()
    logger.info("[rank%d] Finished FLUX.2 Qwen3 cache: written=%d skipped=%d gpu_fwd=%.1fs", local_rank, written, skipped, total_fwd_s)


if __name__ == "__main__":
    main()
