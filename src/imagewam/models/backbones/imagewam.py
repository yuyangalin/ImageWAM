import os
import time
from typing import Any, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from PIL import Image

from imagewam.utils.logging_config import get_logger

from .action_dit import ActionDiT
from .helpers.loader import load_wan22_ti2v_5b_components
from .mot import MoT
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler

logger = get_logger(__name__)


class ImageWAM(torch.nn.Module):
    """MoT world model with video/action experts."""

    def __init__(
        self,
        video_expert,
        action_expert: ActionDiT,
        mot: MoT,
        vae,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        stack: str = "wan22",
        omnigen2_online_text_cache_compatible: bool = False,
        qwen_context_len: int = 128,
        pack_proprio_after_text: bool = False,
    ):
        super().__init__()
        self.video_expert = video_expert
        self.action_expert = action_expert
        self.mot = mot
        # Keep trainer compatibility: optimizer and freeze logic use `model.dit`.
        self.dit = self.mot

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        if text_dim is None:
            if self.text_encoder is None:
                raise ValueError("`text_dim` is required when `text_encoder` is not loaded.")
            text_dim = int(self.text_encoder.dim)
        self.text_dim = int(text_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        if self.proprio_dim is not None:
            self.proprio_encoder = nn.Linear(self.proprio_dim, self.text_dim).to(torch_dtype)
        else:
            self.proprio_encoder = None

        self.train_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_train_shift,
        )
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_infer_shift,
        )
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_train_shift,
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_infer_shift,
        )
        # Optional aliases for consistency with Wan22Core naming.
        self.train_scheduler = self.train_video_scheduler
        self.infer_scheduler = self.infer_video_scheduler

        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_action = float(loss_lambda_action)
        self.stack = str(stack)
        self.omnigen2_online_text_cache_compatible = bool(omnigen2_online_text_cache_compatible)
        self.qwen_context_len = int(qwen_context_len)
        self.pack_proprio_after_text = bool(pack_proprio_after_text)

        self.to(self.device)

    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        proprio_dim: Optional[int] = None,
        redirect_common_files: bool = True,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = True,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        mot_gqa_implementation: str = "repeat",
        mot_force_flash_attention: bool = False,
        omnigen2_online_text_cache_compatible: bool = False,
        qwen_context_len: int = 128,
        pack_proprio_after_text: bool = False,
    ):
        if video_dit_config is None:
            raise ValueError("`video_dit_config` is required for ImageWAM.from_wan22_pretrained().")
        if "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config['text_dim']` is required for ImageWAM.")

        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
        )

        video_expert = components.dit
        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        if int(action_expert.num_heads) != int(video_expert.num_heads):
            raise ValueError("ActionDiT `num_heads` must match video expert for MoT mixed attention.")
        if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
            raise ValueError("ActionDiT `attn_head_dim` must match video expert for MoT mixed attention.")
        if int(len(action_expert.blocks)) != int(len(video_expert.blocks)):
            raise ValueError("ActionDiT `num_layers` must match video expert.")

        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
            gqa_implementation=mot_gqa_implementation,
            force_flash_attention=mot_force_flash_attention,
        )

        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            stack="wan22",
            pack_proprio_after_text=bool(pack_proprio_after_text),
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else action_dit_pretrained_path
            ),
        }
        return model

    @classmethod
    def from_omnigen2_pretrained(
        cls,
        omnigen2_model_path: str,
        omnigen2_vae_path: str,
        qwen_path: str | None,
        action_dit_config: dict[str, Any],
        action_dit_pretrained_path: str | None = None,
        proprio_dim: Optional[int] = None,
        transformer_subfolder: str | None = "transformer",
        vae_subfolder: str | None = "vae",
        load_text_encoder: bool = True,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        mot_checkpoint_mixed_attn: bool = True,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        mot_gqa_implementation: str = "repeat",
        mot_force_flash_attention: bool = False,
        omnigen2_online_text_cache_compatible: bool = False,
        qwen_context_len: int = 128,
        pack_proprio_after_text: bool = False,
    ):
        from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
        from transformers import AutoTokenizer, Qwen2_5_VLModel

        from .action_dit_omnigen2 import ActionDiTOmnigen2
        from .omnigen2_video_expert import OmniGen2VideoExpert

        video_expert = OmniGen2VideoExpert.from_pretrained(
            omnigen2_model_path,
            subfolder=transformer_subfolder,
            torch_dtype=torch_dtype,
        )
        action_cfg = dict(action_dit_config)
        expected_action_shape = {
            "num_layers": len(video_expert.blocks),
            "num_heads": int(video_expert.num_heads),
            "num_kv_heads": int(video_expert.num_kv_heads),
            "attn_head_dim": int(video_expert.attn_head_dim),
        }
        for key, expected_value in expected_action_shape.items():
            if key in action_cfg and int(action_cfg[key]) != int(expected_value):
                logger.warning(
                    "Overriding action_dit_config.%s=%s to match OmniGen2 video expert value %s.",
                    key,
                    action_cfg[key],
                    expected_value,
                )
            action_cfg[key] = expected_value
        action_expert = ActionDiTOmnigen2.from_pretrained(
            action_dit_config=action_cfg,
            action_dit_pretrained_path=action_dit_pretrained_path,
            device=device,
            torch_dtype=torch_dtype,
        )
        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
            gqa_implementation=mot_gqa_implementation,
            force_flash_attention=mot_force_flash_attention,
        )
        vae = AutoencoderKL.from_pretrained(
            omnigen2_vae_path,
            subfolder=vae_subfolder,
            torch_dtype=torch_dtype,
        )
        tokenizer = None
        text_encoder = None
        if load_text_encoder:
            logger.warning(
                "ImageWAM-OmniGen2 is loading Qwen from `%s`. This follows OmniGen2 official training configs "
                "using upstream Qwen2.5-VL-3B-Instruct. It is not yet confirmed whether the official "
                "inference `mllm/` subfolder should be used instead for this ImageWAM path.",
                qwen_path,
            )
            if qwen_path is None:
                raise ValueError("`qwen_path` is required when `load_text_encoder=True`.")
            tokenizer = AutoTokenizer.from_pretrained(qwen_path)
            tokenizer.padding_side = "right"
            text_encoder = Qwen2_5_VLModel.from_pretrained(qwen_path, torch_dtype=torch_dtype)

        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_dim=2048,
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            stack="omnigen2",
            omnigen2_online_text_cache_compatible=bool(omnigen2_online_text_cache_compatible),
            qwen_context_len=int(qwen_context_len),
            pack_proprio_after_text=bool(pack_proprio_after_text),
        )
        model.model_paths = {
            "omnigen2_transformer": omnigen2_model_path,
            "omnigen2_vae": omnigen2_vae_path,
            "qwen": qwen_path,
            "action_dit": action_dit_pretrained_path,
        }
        return model

    @classmethod
    def from_ovis_u1_pretrained(
        cls,
        ovis_u1_model_path: str,
        action_dit_config: dict[str, Any],
        action_dit_pretrained_path: str | None = None,
        proprio_dim: Optional[int] = None,
        load_condition_encoder: bool = False,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        mot_checkpoint_mixed_attn: bool = True,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        mot_gqa_implementation: str = "repeat",
        mot_force_flash_attention: bool = False,
    ):
        from transformers import AutoModelForCausalLM

        from .action_dit_yak import ActionDiTYak
        from .mot import MoT
        from .ovis_u1_video_expert import OvisU1VideoExpert

        ovis_model = AutoModelForCausalLM.from_pretrained(
            ovis_u1_model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        ).eval()
        if load_condition_encoder:
            ovis_model = ovis_model.to(device=device, dtype=torch_dtype)
        visual_generator = ovis_model.get_visual_generator()
        video_expert = OvisU1VideoExpert.from_visual_generator(visual_generator)

        action_cfg = dict(action_dit_config)
        expected_action_shape = {
            "residual_dim": int(video_expert.hidden_dim),
            "num_heads": int(video_expert.num_heads),
            "num_layers_double": int(video_expert.double_layers),
            "num_layers_single": int(video_expert.single_layers),
        }
        for key, expected_value in expected_action_shape.items():
            if key in action_cfg and int(action_cfg[key]) != int(expected_value):
                logger.warning(
                    "Overriding action_dit_config.%s=%s to match Ovis-U1 Yak value %s.",
                    key,
                    action_cfg[key],
                    expected_value,
                )
            action_cfg[key] = expected_value
        action_expert = ActionDiTYak.from_pretrained(
            action_dit_config=action_cfg,
            action_dit_pretrained_path=action_dit_pretrained_path,
            device=device,
            torch_dtype=torch_dtype,
        )
        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
            gqa_implementation=mot_gqa_implementation,
            force_flash_attention=mot_force_flash_attention,
        )
        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=visual_generator.get_vae(),
            text_encoder=None,
            tokenizer=None,
            text_dim=4096,
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            stack="ovis_u1",
        )
        object.__setattr__(model, "ovis_condition_encoder", ovis_model if load_condition_encoder else None)
        model.model_paths = {
            "ovis_u1": ovis_u1_model_path,
            "action_dit": action_dit_pretrained_path,
        }
        return model

    @classmethod
    def from_flux2_klein_pretrained(
        cls,
        flux2_model_path: str,
        ae_model_path: str,
        action_dit_config: dict[str, Any],
        action_dit_pretrained_path: str | None = None,
        flux2_src_path: str | None = None,
        variant: str = "klein-base-4b",
        proprio_dim: Optional[int] = None,
        load_text_encoder: bool = False,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        mot_checkpoint_mixed_attn: bool = True,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        mot_gqa_implementation: str = "repeat",
        mot_force_flash_attention: bool = False,
        pack_proprio_after_text: bool = True,
        flux2_lora_config: Optional[dict[str, Any]] = None,
        qwen3_model_spec: str | None = None,
        qwen_context_len: int = 512,
    ):
        from safetensors.torch import load_file as load_sft

        from .action_dit_flux2 import ActionDiTFlux2
        from .flux2_imports import ensure_flux2_importable
        from .flux2_video_expert import Flux2VideoExpert
        from .mot import MoT

        ensure_flux2_importable(flux2_src_path)
        from flux2.autoencoder import AutoEncoder, AutoEncoderParams

        key = str(variant).lower().replace("_", "-")
        if key in {"klein-base-4b", "flux.2-klein-base-4b", "4b", "base-4b"}:
            text_dim = 7680
            default_qwen3_model_spec = "Qwen/Qwen3-4B"
        elif key in {"klein-base-9b", "flux.2-klein-base-9b", "9b", "base-9b"}:
            text_dim = 12288
            default_qwen3_model_spec = "Qwen/Qwen3-8B"
        else:
            raise ValueError(f"Unsupported FLUX.2 Klein variant: {variant!r}")

        video_expert = Flux2VideoExpert.from_pretrained(
            flux2_model_path=flux2_model_path,
            variant=key,
            flux2_src_path=flux2_src_path,
            device=device,
            torch_dtype=torch_dtype,
        )
        flux2_lora_config = dict(flux2_lora_config or {})
        if bool(flux2_lora_config.get("enabled", False)):
            from .lora import apply_lora_to_linear_suffixes

            target_suffixes = flux2_lora_config.get(
                "target_suffixes",
                [
                    "qkv",
                    "proj",
                    "linear1",
                    "linear2",
                    "img_mlp.0",
                    "img_mlp.2",
                    "txt_mlp.0",
                    "txt_mlp.2",
                ],
            )
            apply_lora_to_linear_suffixes(
                video_expert.transformer,
                target_suffixes=target_suffixes,
                rank=int(flux2_lora_config.get("rank", 16)),
                alpha=float(flux2_lora_config.get("alpha", 16.0)),
                dropout=float(flux2_lora_config.get("dropout", 0.0)),
            )
            video_expert.flux2_lora_enabled = True
            video_expert.flux2_lora_target_suffixes = tuple(str(item) for item in target_suffixes)
        else:
            video_expert.flux2_lora_enabled = False
        action_cfg = dict(action_dit_config)
        expected_action_shape = {
            "num_heads": int(video_expert.num_heads),
            "attn_head_dim": int(video_expert.attn_head_dim),
            "num_layers_double": int(video_expert.double_layers),
            "num_layers_single": int(video_expert.single_layers),
        }
        action_cfg.setdefault("hidden_dim", 1024)
        for key_name, expected_value in expected_action_shape.items():
            if key_name in action_cfg and int(action_cfg[key_name]) != int(expected_value):
                logger.warning(
                    "Overriding action_dit_config.%s=%s to match FLUX.2 value %s.",
                    key_name,
                    action_cfg[key_name],
                    expected_value,
                )
            action_cfg[key_name] = expected_value
        action_expert = ActionDiTFlux2.from_pretrained(
            action_dit_config=action_cfg,
            action_dit_pretrained_path=action_dit_pretrained_path,
            device=device,
            torch_dtype=torch_dtype,
        )
        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
            gqa_implementation=mot_gqa_implementation,
            force_flash_attention=mot_force_flash_attention,
        )
        with torch.device("meta"):
            ae = AutoEncoder(AutoEncoderParams())
        ae_state = load_sft(str(ae_model_path), device=str(device))
        ae.load_state_dict(ae_state, strict=True, assign=True)
        ae = ae.to(device=device, dtype=torch_dtype).eval()
        if load_text_encoder:
            from types import SimpleNamespace
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_spec = qwen3_model_spec or default_qwen3_model_spec
            qwen3_model = AutoModelForCausalLM.from_pretrained(
                model_spec,
                torch_dtype=torch_dtype,
            ).to(device).eval()
            text_encoder = SimpleNamespace(
                model=qwen3_model,
                tokenizer=AutoTokenizer.from_pretrained(model_spec),
                max_length=int(qwen_context_len),
            )
        else:
            text_encoder = None

        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=ae,
            text_encoder=text_encoder,
            tokenizer=None,
            text_dim=text_dim,
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            stack="flux2",
            qwen_context_len=int(qwen_context_len),
            pack_proprio_after_text=bool(pack_proprio_after_text),
        )
        model.model_paths = {
            "flux2": flux2_model_path,
            "flux2_src": flux2_src_path,
            "ae": ae_model_path,
            "action_dit": action_dit_pretrained_path,
            "qwen3": qwen3_model_spec or default_qwen3_model_spec,
        }
        model.flux2_qwen3_model_spec = qwen3_model_spec or default_qwen3_model_spec
        model.save_lora_merged = bool(flux2_lora_config.get("save_lora_merged", bool(flux2_lora_config.get("enabled", False))))
        model.save_trainable_only = bool(flux2_lora_config.get("save_trainable_only", False))
        return model

    @classmethod
    def from_dim_pretrained(
        cls,
        dim_model_path: str,
        sana_config_path: str,
        qwen_path: str,
        action_dit_config: dict[str, Any],
        action_dit_pretrained_path: str | None = None,
        proprio_dim: Optional[int] = None,
        max_condition_length: int = 8192,
        with_latents_condition: bool = True,
        load_mllm: bool = True,
        qwen_attn_implementation: str = "flash_attention_2",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        mot_checkpoint_mixed_attn: bool = True,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        mot_gqa_implementation: str = "repeat",
        mot_force_flash_attention: bool = False,
        pack_proprio_after_text: bool = True,
    ):
        from safetensors.torch import load_file
        from transformers import AutoProcessor

        from .action_dit_sana import ActionDiTSana
        from .dim_video_expert import DimVideoExpert
        from .mot import MoT

        from models.modeling_dim import Qwen2_5_VLForConditionalGeneration
        from models.multimodal.multimodal_projector.builder import build_projector

        video_expert = DimVideoExpert.from_pretrained(
            dim_model_path=dim_model_path,
            sana_config_path=sana_config_path,
            max_condition_length=int(max_condition_length),
            with_latents_condition=bool(with_latents_condition),
            device=device,
            torch_dtype=torch_dtype,
        )
        action_cfg = dict(action_dit_config)
        expected_action_shape = {
            "attn_dim": int(video_expert.hidden_dim),
            "context_dim": int(video_expert.hidden_dim),
            "num_heads": int(video_expert.num_heads),
            "num_layers": len(video_expert.blocks),
        }
        for key, expected_value in expected_action_shape.items():
            if key in action_cfg and int(action_cfg[key]) != int(expected_value):
                logger.warning(
                    "Overriding action_dit_config.%s=%s to match DIM/SANA value %s.",
                    key,
                    action_cfg[key],
                    expected_value,
                )
            action_cfg[key] = expected_value
        action_expert = ActionDiTSana.from_pretrained(
            action_dit_config=action_cfg,
            action_dit_pretrained_path=action_dit_pretrained_path,
            device=device,
            torch_dtype=torch_dtype,
        )
        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
            gqa_implementation=mot_gqa_implementation,
            force_flash_attention=mot_force_flash_attention,
        )
        processor = AutoProcessor.from_pretrained(qwen_path, padding_side="left")
        processor.tokenizer.padding_side = "left"
        mllm = None
        if load_mllm:
            mllm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                qwen_path,
                torch_dtype=torch_dtype,
                attn_implementation=qwen_attn_implementation,
            ).eval()
            for param in mllm.parameters():
                param.requires_grad = False
        projector = build_projector(proj_in=2048, proj_out=int(video_expert.caption_dim), projector_type="mlp2x_gelu")
        ckpt = os.path.join(dim_model_path, "model.safetensors") if os.path.isdir(dim_model_path) else dim_model_path
        state = load_file(ckpt)
        projector_state = {k[len("projector.") :]: v for k, v in state.items() if k.startswith("projector.")}
        missing, unexpected = projector.load_state_dict(projector_state, strict=False)
        if missing:
            logger.warning("DIM projector missing keys: %s", missing[:20])
        if unexpected:
            logger.warning("DIM projector unexpected keys: %s", unexpected[:20])
        projector = projector.to(device=device, dtype=torch_dtype)

        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=video_expert.vae,
            text_encoder=mllm,
            tokenizer=processor.tokenizer,
            text_dim=int(video_expert.caption_dim),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            stack="dim",
            pack_proprio_after_text=bool(pack_proprio_after_text),
        )
        model.dim_processor = processor
        model.dim_projector = projector
        model.dim_max_condition_length = int(max_condition_length)
        model.dim_with_latents_condition = bool(with_latents_condition)
        model.model_paths = {
            "dim": dim_model_path,
            "sana_config": sana_config_path,
            "qwen": qwen_path,
            "action_dit": action_dit_pretrained_path,
        }
        return model

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.mot.to(*args, **kwargs)
        if self.text_encoder is not None:
            if hasattr(self.text_encoder, "to"):
                self.text_encoder.to(*args, **kwargs)
            elif hasattr(self.text_encoder, "model") and hasattr(self.text_encoder.model, "to"):
                self.text_encoder.model.to(*args, **kwargs)
        if hasattr(self, "dim_projector"):
            self.dim_projector.to(*args, **kwargs)
        self.vae.to(*args, **kwargs)
        return self

    @staticmethod
    def _check_resize_height_width(height, width, num_frames):
        if height % 16 != 0:
            height = (height + 15) // 16 * 16
        if width % 16 != 0:
            width = (width + 15) // 16 * 16
        if num_frames % 4 != 1:
            num_frames = (num_frames + 3) // 4 * 4 + 1
        return height, width, num_frames

    @torch.no_grad()
    def encode_prompt(self, prompt: Union[str, Sequence[str]]):
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError(
                "Prompt encoding requires loaded text encoder/tokenizer. "
                "Set `load_text_encoder=true` or provide precomputed `context/context_mask`."
            )
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)
        # FIXME: original implementation's zero padding is visible in cross-attn.
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        mask = torch.ones_like(mask)
        return prompt_emb.to(device=self.device), mask

    @staticmethod
    def _scheduler_timestep_to_unit(timestep: torch.Tensor, scheduler) -> torch.Tensor:
        """Convert ImageWAM scheduler timesteps back to Yak's [0, 1] time domain."""
        num_train_timesteps = float(getattr(scheduler, "num_train_timesteps", 1000))
        if num_train_timesteps <= 0:
            raise ValueError(f"`num_train_timesteps` must be positive, got {num_train_timesteps}.")
        return timestep / num_train_timesteps

    def _append_proprio_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_encoder is None or proprio is None:
            return context, context_mask
        if proprio.ndim != 2:
            raise ValueError(f"`proprio` must be 2D [B, D], got shape {tuple(proprio.shape)}")
        if self.proprio_dim is None or proprio.shape[1] != self.proprio_dim:
            raise ValueError(
                f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
            )
        proprio_token = self.proprio_encoder(
            proprio.to(device=self.device, dtype=context.dtype).unsqueeze(1)
        ).to(dtype=context.dtype) # [B, 1, D]
        if not getattr(self, "pack_proprio_after_text", False):
            proprio_mask = torch.ones((context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device)
            return (
                torch.cat([context, proprio_token], dim=1),
                torch.cat([context_mask, proprio_mask], dim=1),
            )
        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        if context.shape[:2] != context_mask.shape:
            raise ValueError(
                f"`context/context_mask` leading dims must match, got {tuple(context.shape[:2])} and {tuple(context_mask.shape)}"
            )
        if context.shape[0] != proprio_token.shape[0]:
            raise ValueError(
                f"`proprio` batch size must match context batch size ({context.shape[0]}), got {proprio_token.shape[0]}"
            )

        context_mask = context_mask.to(device=context.device, dtype=torch.bool)
        new_context = context.new_zeros(context.shape[0], context.shape[1] + 1, context.shape[2])

        # OmniGen2 treats context_mask.sum() as a contiguous prefix length, so the
        # proprio token must sit immediately after the valid text tokens.
        valid_counts = context_mask.sum(dim=1)
        valid_rank = context_mask.cumsum(dim=1) - 1
        invalid_mask = ~context_mask
        invalid_rank = invalid_mask.cumsum(dim=1) - 1
        target_indices = torch.where(
            context_mask,
            valid_rank,
            valid_counts[:, None] + 1 + invalid_rank,
        )
        new_context.scatter_(
            dim=1,
            index=target_indices[:, :, None].expand(-1, -1, context.shape[2]),
            src=context,
        )
        batch_indices = torch.arange(context.shape[0], device=context.device)
        new_context[batch_indices, valid_counts] = proprio_token[:, 0]
        positions = torch.arange(context.shape[1] + 1, device=context.device)
        new_context_mask = positions[None, :] <= valid_counts[:, None]

        return new_context, new_context_mask

    def _append_proprio_to_context_if_enabled(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
        source: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_encoder is None:
            return context, context_mask
        if proprio is None:
            raise ValueError(f"`{source}` requires `proprio` when `proprio_dim` is enabled.")

        if proprio.ndim == 3:
            proprio = proprio[:, 0, :]
        elif proprio.ndim == 2:
            pass
        elif proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        else:
            raise ValueError(
                f"`{source}` `proprio` must be [B,T,D], [B,D], or [D], got shape {tuple(proprio.shape)}"
            )
        if proprio.shape[0] != context.shape[0]:
            raise ValueError(
                f"`{source}` `proprio` batch size must match context batch size "
                f"({context.shape[0]}), got {proprio.shape[0]}"
            )
        if self.proprio_dim is None or proprio.shape[1] != self.proprio_dim:
            raise ValueError(
                f"`{source}` `proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
            )
        return self._append_proprio_to_context(
            context=context,
            context_mask=context_mask,
            proprio=proprio.to(device=self.device, dtype=self.torch_dtype),
        )

    def _configure_action_attention_capture(
        self,
        *,
        condition_slice: slice | tuple[int, int] | list[int],
        condition_grid: tuple[int, int],
        prefix_len: int,
        action_len: int,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        capture = getattr(self.mot, "action_attention_capture", None)
        if capture is None or not hasattr(capture, "configure"):
            return
        capture.configure(
            condition_slice=condition_slice,
            condition_grid=condition_grid,
            prefix_len=prefix_len,
            action_len=action_len,
            metadata=metadata or {},
        )

    def _start_action_attention_capture_step(self, step_idx: int) -> None:
        capture = getattr(self.mot, "action_attention_capture", None)
        if capture is not None and hasattr(capture, "start_step"):
            capture.start_step(int(step_idx))

    @staticmethod
    def _image_tensor_to_pil_list(image: torch.Tensor) -> list[Image.Image]:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"`image` must be [B,3,H,W], got {tuple(image.shape)}")
        image = image.detach().float().cpu()
        if image.min() < 0:
            image = (image + 1.0) * 0.5
        image = image.clamp(0, 1)
        arr = (image.permute(0, 2, 3, 1).numpy() * 255.0).round().astype("uint8")
        return [Image.fromarray(item) for item in arr]

    def _build_dim_user_labels(self, input_ids: torch.Tensor) -> torch.Tensor:
        tokenizer = self.dim_processor.tokenizer
        im_start_id = tokenizer("<|im_start|>").input_ids[0]
        im_end_id = tokenizer("<|im_end|>").input_ids[0]
        user_id = tokenizer("user").input_ids[0]
        newline_id = tokenizer("\n").input_ids[0]
        labels = torch.full_like(input_ids, -100, dtype=torch.long)
        for b in range(input_ids.shape[0]):
            user_begin, user_end = [], []
            for idx in range(3, input_ids.shape[1]):
                if (
                    input_ids[b, idx - 3] == im_start_id
                    and input_ids[b, idx - 2] == user_id
                    and input_ids[b, idx - 1] == newline_id
                ):
                    user_begin.append(idx)
                if input_ids[b, idx] == im_end_id and len(user_begin) == len(user_end) + 1:
                    user_end.append(idx)
            if len(user_begin) != len(user_end):
                raise ValueError("DIM Qwen user span parse failed: begin/end counts mismatch.")
            for start, end in zip(user_begin, user_end):
                labels[b, start:end] = input_ids[b, start:end]
        return labels

    def _build_dim_messages(self, instruction: str | Sequence[str], image: torch.Tensor):
        pil_images = self._image_tensor_to_pil_list(image)
        if isinstance(instruction, str):
            instructions = [instruction] * len(pil_images)
        else:
            instructions = list(instruction)
        if len(instructions) != len(pil_images):
            raise ValueError(f"Expected {len(pil_images)} DIM instructions, got {len(instructions)}.")
        messages = []
        for img, prompt in zip(pil_images, instructions):
            messages.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": str(prompt)},
                        ],
                    }
                ]
            )
        return messages

    def _prepare_dim_condition_from_batch(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if self.text_encoder is None or not hasattr(self, "dim_projector"):
            raise ValueError("DIM online conditioning requires loaded Qwen MLLM and projector.")
        batch = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        with torch.no_grad():
            outputs = self.text_encoder(
                **batch,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden = outputs.hidden_states[-1]
        projected = self.dim_projector(hidden.to(dtype=self.torch_dtype))
        user_labels = self._build_dim_user_labels(batch["input_ids"]).to(projected.device)
        valid = (user_labels != -100) & batch["attention_mask"].to(device=projected.device, dtype=torch.bool)
        max_len = int(getattr(self, "dim_max_condition_length", projected.shape[1]))
        out = projected.new_zeros(projected.shape[0], max_len, projected.shape[-1])
        mask = torch.zeros(projected.shape[0], max_len, device=projected.device, dtype=torch.bool)
        for b in range(projected.shape[0]):
            item = projected[b, valid[b]]
            if item.shape[0] > max_len:
                item = item[-max_len:]
            out[b, : item.shape[0]] = item
            mask[b, : item.shape[0]] = True
        return out, mask

    def _encode_dim_condition_online(
        self,
        instruction: str | Sequence[str],
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        messages = self._build_dim_messages(instruction, image)
        texts = [
            self.dim_processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
            for msg in messages
        ]
        pil_images = [msg[0]["content"][0]["image"] for msg in messages]
        batch = self.dim_processor(
            text=texts,
            images=pil_images,
            padding=True,
            return_tensors="pt",
        )
        return self._prepare_dim_condition_from_batch(batch)

    @torch.no_grad()
    def _encode_dim_image_latents(self, image: torch.Tensor) -> torch.Tensor:
        return self.video_expert.encode_image_latents(image.to(device=self.device, dtype=self.torch_dtype))

    @torch.no_grad()
    def _decode_dim_image_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return self.video_expert.decode_image_latents(latents)

    @torch.no_grad()
    def _encode_video_latents(self, video_tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        z = self.vae.encode(
            video_tensor,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        return z

    @torch.no_grad()
    def _encode_input_image_latents_tensor(self, input_image: torch.Tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        image = input_image.to(device=self.device)[0].unsqueeze(1)
        z = self.vae.encode([image], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        if isinstance(z, list):
            z = z[0].unsqueeze(0)
        return z

    def _decode_latents(self, latents, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        video_tensor = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video_tensor = video_tensor.squeeze(0).detach().float().clamp(-1, 1)
        video_tensor = ((video_tensor + 1.0) * 127.5).to(torch.uint8).cpu()
        frames = []
        for t in range(video_tensor.shape[1]):
            frame = video_tensor[:, t].permute(1, 2, 0).numpy()
            frames.append(Image.fromarray(frame))
        return frames

    def build_inputs(self, sample, tiled: bool = False):
        video = sample["video"]
        if "context" not in sample or "context_mask" not in sample:
            raise ValueError(
                "ImageWAM training requires `sample['context']` and `sample['context_mask']`."
            )
        context = sample["context"]
        context_mask = sample["context_mask"]
        proprio = sample.get("proprio", None)
        if video.ndim != 5:
            raise ValueError(f"`sample['video']` must be 5D [B, 3, T, H, W], got shape {tuple(video.shape)}")
        if video.shape[1] != 3:
            raise ValueError(f"`sample['video']` channel dimension must be 3, got shape {tuple(video.shape)}")

        batch_size, _, num_frames, height, width = video.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"Video spatial dims must be multiples of 16, got H={height}, W={width}"
            )
        if num_frames % 4 != 1:
            raise ValueError(f"Video T must satisfy T % 4 == 1, got T={num_frames}")
        if num_frames <= 1:
            raise ValueError(f"Video T must be > 1 for action-conditioned training, got T={num_frames}")

        if "action" not in sample:
            raise ValueError("`sample['action']` is required for ImageWAM training.")

        action = sample["action"]
        if action.ndim != 3:
            raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
        action_horizon = int(action.shape[1])
        if action_horizon % (num_frames - 1) != 0:
            raise ValueError(
                f"`sample['action']` temporal dimension must be divisible by video transitions ({num_frames - 1}), got {action_horizon}"
            )

        action_is_pad = sample.get("action_is_pad", None)
        if action_is_pad is not None:
            if action_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['action_is_pad']` must be 2D [B, T], got shape {tuple(action_is_pad.shape)}"
                )
            if action_is_pad.shape[0] != batch_size or action_is_pad.shape[1] != action_horizon:
                raise ValueError(
                    "`sample['action_is_pad']` shape mismatch: "
                    f"got {tuple(action_is_pad.shape)} vs expected ({batch_size}, {action_horizon})"
                )

        action_dim_is_pad = sample.get("action_dim_is_pad", None)
        if action_dim_is_pad is not None:
            if action_dim_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['action_dim_is_pad']` must be 2D [B, D], got shape {tuple(action_dim_is_pad.shape)}"
                )
            if action_dim_is_pad.shape[0] != batch_size or action_dim_is_pad.shape[1] != action.shape[2]:
                raise ValueError(
                    "`sample['action_dim_is_pad']` shape mismatch: "
                    f"got {tuple(action_dim_is_pad.shape)} vs expected ({batch_size}, {action.shape[2]})"
                )

        image_is_pad = sample.get("image_is_pad", None)
        if image_is_pad is not None:
            if image_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['image_is_pad']` must be 2D [B, T], got shape {tuple(image_is_pad.shape)}"
                )
            if image_is_pad.shape[0] != batch_size or image_is_pad.shape[1] != num_frames:
                raise ValueError(
                    "`sample['image_is_pad']` shape mismatch: "
                    f"got {tuple(image_is_pad.shape)} vs expected ({batch_size}, {num_frames})"
                )
        
        input_video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        input_latents = self._encode_video_latents(input_video, tiled=tiled)

        first_frame_latents = None
        fuse_flag = False
        if getattr(self.video_expert, "fuse_vae_embedding_in_latents", False):
            first_frame_latents = input_latents[:, :, 0:1]
            fuse_flag = True

        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if self.proprio_encoder is not None:
            if proprio is None:
                raise ValueError("`sample['proprio']` is required when `proprio_dim` is enabled.")
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")
            if proprio.shape[2] != self.proprio_dim:
                raise ValueError(
                    f"`sample['proprio']` last dim must be {self.proprio_dim}, got {proprio.shape[2]}"
                )
            proprio = proprio[:, 0, :] # [B, D]
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio.to(device=self.device, dtype=self.torch_dtype),
            )
        action = action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)

        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if action_dim_is_pad is not None:
            action_dim_is_pad = action_dim_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if image_is_pad is not None:
            image_is_pad = image_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)

        return {
            "context": context,
            "context_mask": context_mask,
            "input_latents": input_latents,
            "first_frame_latents": first_frame_latents,
            "fuse_vae_embedding_in_latents": fuse_flag,
            "action": action,
            "action_is_pad": action_is_pad,
            "action_dim_is_pad": action_dim_is_pad,
            "image_is_pad": image_is_pad,
        }

    @torch.no_grad()
    def _encode_omnigen2_image_latents(self, image: torch.Tensor) -> torch.Tensor:
        image = image.to(device=self.device, dtype=getattr(self.vae, "dtype", self.torch_dtype), non_blocking=True)
        latent = self.vae.encode(image).latent_dist.sample()
        if getattr(self.vae.config, "shift_factor", None) is not None:
            latent = latent - self.vae.config.shift_factor
        if getattr(self.vae.config, "scaling_factor", None) is not None:
            latent = latent * self.vae.config.scaling_factor
        return latent.to(dtype=self.torch_dtype)

    @torch.no_grad()
    def _decode_omnigen2_image_latents(self, latents: torch.Tensor) -> torch.Tensor:
        z = latents.to(device=self.device, dtype=getattr(self.vae, "dtype", self.torch_dtype))
        if getattr(self.vae.config, "scaling_factor", None) is not None:
            z = z / self.vae.config.scaling_factor
        if getattr(self.vae.config, "shift_factor", None) is not None:
            z = z + self.vae.config.shift_factor
        decoded = self.vae.decode(z)
        image = decoded.sample if hasattr(decoded, "sample") else decoded[0]
        return image.detach().float().clamp(-1, 1)

    @torch.no_grad()
    def _encode_omnigen2_text(self, sample) -> tuple[torch.Tensor, torch.Tensor]:
        if "text_hidden_states" in sample and "text_attention_mask" in sample:
            return (
                sample["text_hidden_states"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True),
                sample["text_attention_mask"].to(device=self.device, dtype=torch.bool, non_blocking=True),
            )
        if "text_ids" in sample and "text_mask" in sample:
            text_ids = sample["text_ids"].to(device=self.device, non_blocking=True)
            text_mask = sample["text_mask"].to(device=self.device, dtype=torch.bool, non_blocking=True)
        else:
            if self.text_encoder is None or self.tokenizer is None:
                raise ValueError(
                    "OmniGen2 stack needs precomputed text_hidden_states/text_attention_mask "
                    "or a loaded Qwen tokenizer/text_encoder."
                )
            instruction = sample.get("instruction")
            if instruction is None:
                raise ValueError("OmniGen2 stack sample must include `instruction` when text is not precomputed.")
            if self.omnigen2_online_text_cache_compatible:
                encoded = self.tokenizer(
                    instruction,
                    padding="max_length",
                    truncation=True,
                    max_length=self.qwen_context_len,
                    return_tensors="pt",
                )
            else:
                encoded = self.tokenizer(
                    instruction,
                    padding=True,
                    return_tensors="pt",
                )
            text_ids = encoded.input_ids.to(self.device)
            text_mask = encoded.attention_mask.to(self.device, dtype=torch.bool)
        text_hidden_states = self.text_encoder(
            input_ids=text_ids,
            attention_mask=text_mask,
            output_hidden_states=False,
        ).last_hidden_state
        return text_hidden_states.to(dtype=self.torch_dtype), text_mask

    @torch.no_grad()
    def _prepare_omnigen2_infer_text(
        self,
        prompt: Optional[str],
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")
        if use_prompt:
            return self._encode_omnigen2_text({"instruction": [prompt]})
        if context is None or context_mask is None:
            raise ValueError("`context` and `context_mask` must be both provided together.")
        if context.ndim == 2:
            context = context.unsqueeze(0)
        if context_mask.ndim == 1:
            context_mask = context_mask.unsqueeze(0)
        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        return (
            context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True),
            context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True),
        )

    def build_inputs_omnigen2(self, sample, tiled: bool = False):
        debug_forward = (
            int(os.environ.get("IMAGEWAM_DEBUG_OMNIGEN2_FORWARD_EVERY", "0") or "0") > 0
            or float(os.environ.get("IMAGEWAM_DEBUG_OMNIGEN2_FORWARD_THRESHOLD", "0") or "0") > 0
        )
        build_profile: list[tuple[str, float]] = []

        def _sync_and_mark(name: str, start_time: float) -> float:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            now = time.perf_counter()
            build_profile.append((name, now - start_time))
            return now

        t0 = time.perf_counter()
        video = sample.get("video")
        if "target_latent" in sample:
            target_latent = sample["target_latent"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        else:
            next_frame = sample.get("next_frame", sample.get("target_image"))
            if next_frame is None and video is not None:
                if video.ndim != 5:
                    raise ValueError(f"`sample['video']` must be [B,C,T,H,W], got {tuple(video.shape)}")
                next_frame = video[:, :, -1]
            if next_frame is None:
                raise ValueError("OmniGen2 stack sample requires `next_frame`, `target_image`, `video`, or `target_latent`.")
            target_latent = self._encode_omnigen2_image_latents(next_frame)
        if debug_forward:
            t0 = _sync_and_mark("build.target_latent", t0)

        if "ref_image_latents" in sample:
            ref_image_latents = sample["ref_image_latents"]
            if isinstance(ref_image_latents, torch.Tensor):
                ref_image_latents = ref_image_latents.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        else:
            current_frame = sample.get("current_frame", sample.get("input_image"))
            if current_frame is None and video is not None:
                if video.ndim != 5:
                    raise ValueError(f"`sample['video']` must be [B,C,T,H,W], got {tuple(video.shape)}")
                current_frame = video[:, :, 0]
            if current_frame is None:
                raise ValueError("OmniGen2 stack sample requires `current_frame`, `input_image`, `video`, or `ref_image_latents`.")
            ref_image_latents = self._encode_omnigen2_image_latents(current_frame)
        if debug_forward:
            t0 = _sync_and_mark("build.ref_latent", t0)

        text_hidden_states, text_attention_mask = self._encode_omnigen2_text(sample)
        if self.proprio_encoder is not None:
            text_hidden_states, text_attention_mask = self._append_proprio_to_context_if_enabled(
                context=text_hidden_states,
                context_mask=text_attention_mask,
                proprio=sample.get("proprio"),
                source="OmniGen2 training sample",
            )
        if debug_forward:
            t0 = _sync_and_mark("build.text_to_device", t0)
        action = sample["action"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        action_is_pad = sample.get("action_is_pad")
        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        action_dim_is_pad = sample.get("action_dim_is_pad")
        if action_dim_is_pad is not None:
            action_dim_is_pad = action_dim_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if debug_forward:
            _sync_and_mark("build.action_to_device", t0)
            self._last_omnigen2_build_profile = build_profile
        return {
            "target_latent": target_latent,
            "ref_image_latents": ref_image_latents,
            "text_hidden_states": text_hidden_states,
            "text_attention_mask": text_attention_mask,
            "action": action,
            "action_is_pad": action_is_pad,
            "action_dim_is_pad": action_dim_is_pad,
        }

    @torch.no_grad()
    def _encode_ovis_u1_image_tokens(
        self,
        image: torch.Tensor,
        *,
        time_value: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from .ovis_u1_video_expert import OvisU1VideoExpert

        image = image.to(device=self.device, dtype=getattr(self.vae, "dtype", self.torch_dtype), non_blocking=True)
        latent = self.vae.encode(image).latent_dist.sample()
        if getattr(self.vae.config, "shift_factor", None) is not None:
            latent = latent - self.vae.config.shift_factor
        if getattr(self.vae.config, "scaling_factor", None) is not None:
            latent = latent * self.vae.config.scaling_factor
        latent = latent.to(dtype=self.torch_dtype)
        tokens = OvisU1VideoExpert.pack_latents(latent)
        _, _, latent_h, latent_w = latent.shape
        ids = OvisU1VideoExpert.build_img_ids(
            batch_size=int(latent.shape[0]),
            token_height=latent_h // 2,
            token_width=latent_w // 2,
            time_value=float(time_value),
            device=latent.device,
            dtype=latent.dtype,
        )
        return tokens, ids

    @torch.no_grad()
    def _decode_ovis_u1_image_tokens(self, tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        latent_h = 2 * ((int(height) + 15) // 16)
        latent_w = 2 * ((int(width) + 15) // 16)
        latents = rearrange(
            tokens,
            "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            h=latent_h // 2,
            w=latent_w // 2,
            ph=2,
            pw=2,
        )
        z = latents.to(device=self.device, dtype=getattr(self.vae, "dtype", self.torch_dtype))
        if getattr(self.vae.config, "scaling_factor", None) is not None:
            z = z / self.vae.config.scaling_factor
        if getattr(self.vae.config, "shift_factor", None) is not None:
            z = z + self.vae.config.shift_factor
        image = self.vae.decode(z, return_dict=False)[0]
        return image.detach().float().clamp(-1, 1)

    @torch.no_grad()
    def _prepare_ovis_u1_vlm_image_tensors(
        self,
        image: torch.Tensor,
        visual_tokenizer,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"`image` must be [B,3,H,W] or [3,H,W], got {tuple(image.shape)}")
        image = image.to(device=visual_tokenizer.device, dtype=torch.float32, non_blocking=True)
        if image.min() < 0:
            image = (image + 1.0) * 0.5
        image = image.clamp(0, 1)

        patch_size = int(visual_tokenizer.image_processor.patch_size)
        temporal_patch_size = int(visual_tokenizer.image_processor.temporal_patch_size)
        if temporal_patch_size != 1:
            raise ValueError("Ovis-U1 tensor online condition path currently expects temporal_patch_size=1.")
        hidden_stride = int(visual_tokenizer.image_processor.hidden_stride)
        min_pixels = int(visual_tokenizer.image_processor.min_pixels)
        max_pixels = int(visual_tokenizer.image_processor.max_pixels)
        _, _, height, width = image.shape
        resized_height, resized_width = visual_tokenizer.smart_resize(
            int(height),
            int(width),
            factor=patch_size * hidden_stride,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        image = F.interpolate(
            image,
            size=(int(resized_height), int(resized_width)),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).clamp(0, 1)

        mean = torch.tensor(
            visual_tokenizer.image_processor.image_mean,
            device=image.device,
            dtype=image.dtype,
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            visual_tokenizer.image_processor.image_std,
            device=image.device,
            dtype=image.dtype,
        ).view(1, 3, 1, 1)
        image = (image - mean) / std

        batch_size, channel, resized_height, resized_width = image.shape
        grid_h = resized_height // patch_size
        grid_w = resized_width // patch_size
        if grid_h % hidden_stride != 0 or grid_w % hidden_stride != 0:
            raise ValueError(
                "Ovis-U1 VLM resized grid must be divisible by hidden_stride, "
                f"got grid=({grid_h},{grid_w}) hidden_stride={hidden_stride}."
            )
        patches = image.reshape(
            batch_size,
            channel,
            grid_h // hidden_stride,
            hidden_stride,
            patch_size,
            grid_w // hidden_stride,
            hidden_stride,
            patch_size,
        )
        patches = patches.permute(0, 2, 5, 3, 6, 1, 4, 7).contiguous()
        flatten_patches = patches.reshape(batch_size * grid_h * grid_w, channel * patch_size * patch_size)
        grid_thws = torch.tensor(
            [[1, grid_h, grid_w]],
            device=image.device,
            dtype=torch.long,
        ).repeat(batch_size, 1)
        return flatten_patches, grid_thws, int(resized_height), int(resized_width)

    @torch.no_grad()
    def _encode_ovis_u1_condition_online(
        self,
        prompt: str | Sequence[str],
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        condition_encoder = getattr(self, "ovis_condition_encoder", None)
        if condition_encoder is None:
            raise ValueError(
                "Ovis-U1 online condition encoding requires `load_condition_encoder=true`. "
                "Otherwise provide precomputed `text_hidden_states`/`text_attention_mask`."
            )
        if image.ndim == 3:
            batch_size = 1
        elif image.ndim == 4:
            batch_size = int(image.shape[0])
        else:
            raise ValueError(f"`image` must be [B,3,H,W] or [3,H,W], got {tuple(image.shape)}")
        if isinstance(prompt, str):
            prompts = [prompt] * batch_size
        elif isinstance(prompt, Sequence):
            prompts = list(prompt)
            if len(prompts) != batch_size:
                raise ValueError(f"Expected {batch_size} prompts, got {len(prompts)}.")
        else:
            raise ValueError(f"`prompt` must be str or sequence of str, got {type(prompt)}")
        text_tokenizer = condition_encoder.get_text_tokenizer()
        visual_tokenizer = condition_encoder.get_visual_tokenizer()
        pixel_values, grid_thws, _, _ = self._prepare_ovis_u1_vlm_image_tensors(
            image,
            visual_tokenizer,
        )
        prefix_ids = text_tokenizer(
            "<|im_start|>user\n",
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.to(device=self.device)
        suffix_texts = ["\n" + str(item).strip() + "<|im_end|>\n" for item in prompts]
        suffix = text_tokenizer(
            suffix_texts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        model_device = getattr(condition_encoder, "device", next(condition_encoder.parameters()).device)
        prefix_ids = prefix_ids.to(device=model_device).expand(batch_size, -1)
        image_placeholders = torch.tensor(
            visual_tokenizer.construct_image_placeholders((1, 1)),
            device=model_device,
            dtype=torch.long,
        ).unsqueeze(0).expand(batch_size, -1)
        suffix_ids = suffix.input_ids.to(device=model_device)
        suffix_mask = suffix.attention_mask.to(device=model_device, dtype=torch.bool)
        input_ids = torch.cat([prefix_ids, image_placeholders, suffix_ids], dim=1)
        attention_mask = torch.cat(
            [
                torch.ones_like(prefix_ids, dtype=torch.bool),
                torch.ones_like(image_placeholders, dtype=torch.bool),
                suffix_mask,
            ],
            dim=1,
        )
        pixel_values = pixel_values.to(
            device=visual_tokenizer.device,
            dtype=getattr(visual_tokenizer, "dtype", self.torch_dtype),
        )
        grid_thws = grid_thws.to(device=visual_tokenizer.device)
        _, inputs_embeds, labels, merged_attention_mask, _ = condition_encoder.merge_multimodal(
            text_input_ids=input_ids,
            text_attention_masks=attention_mask,
            text_labels=None,
            pixel_values=pixel_values,
            grid_thws=grid_thws,
            left_padding=True,
        )
        inputs_embeds = inputs_embeds.detach()
        torch.cuda.empty_cache()
        llm = condition_encoder.get_llm()
        llm_device = llm.device
        outputs = llm(
            inputs_embeds=inputs_embeds.to(llm_device),
            labels=labels.to(llm_device),
            attention_mask=merged_attention_mask.to(llm_device),
            output_hidden_states=True,
        )
        text_hidden_states = torch.cat([outputs.hidden_states[-1], outputs.hidden_states[-2]], dim=-1).to(
            device=self.device,
            dtype=self.torch_dtype,
        )
        text_attention_mask = merged_attention_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        return text_hidden_states, text_attention_mask

    @torch.no_grad()
    def _encode_ovis_u1_text(self, sample, condition_image: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        cached_text_hidden_states = sample.get("text_hidden_states")
        if cached_text_hidden_states is not None:
            text_hidden_states = cached_text_hidden_states.to(
                device=self.device,
                dtype=self.torch_dtype,
                non_blocking=True,
            )
            text_attention_mask = sample.get("text_attention_mask")
            if text_attention_mask is None:
                text_attention_mask = torch.ones(
                    text_hidden_states.shape[:2],
                    device=self.device,
                    dtype=torch.bool,
                )
            else:
                text_attention_mask = text_attention_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
            return text_hidden_states, text_attention_mask
        instruction = sample.get("instruction")
        if instruction is None:
            instruction = sample.get("prompt")
        if instruction is None:
            instruction = sample.get("task")
        if instruction is not None:
            if condition_image is None:
                raise ValueError("Ovis-U1 online condition encoding requires the condition/input image.")
            return self._encode_ovis_u1_condition_online(instruction, condition_image)
        raise ValueError(
            "Ovis-U1 training currently requires precomputed `text_hidden_states` "
            "from OvisU1.generate_condition(), or `instruction`/`prompt` with `load_condition_encoder=true`. "
            f"Available sample keys: {sorted(sample.keys())}"
        )

    @torch.no_grad()
    def _prepare_ovis_u1_infer_text(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")
        if use_prompt:
            return self._encode_ovis_u1_condition_online(prompt, input_image)
        if context is None or context_mask is None:
            raise ValueError("`context` and `context_mask` must be both provided together.")
        if context.ndim == 2:
            context = context.unsqueeze(0)
        if context_mask.ndim == 1:
            context_mask = context_mask.unsqueeze(0)
        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        return (
            context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True),
            context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True),
        )

    def build_inputs_ovis_u1(self, sample, tiled: bool = False):
        video = sample.get("video")
        condition_image = None
        if "target_latent" in sample and "target_img_ids" in sample:
            target_tokens = sample["target_latent"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            target_img_ids = sample["target_img_ids"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        else:
            next_frame = sample.get("next_frame", sample.get("target_image"))
            if next_frame is None and video is not None:
                if video.ndim != 5:
                    raise ValueError(f"`sample['video']` must be [B,C,T,H,W], got {tuple(video.shape)}")
                next_frame = video[:, :, -1]
            if next_frame is None:
                raise ValueError("Ovis-U1 stack sample requires `next_frame`, `target_image`, or target token fields.")
            target_tokens, target_img_ids = self._encode_ovis_u1_image_tokens(next_frame, time_value=0.0)

        if "ref_image_latents" in sample and "ref_img_ids" in sample:
            ref_tokens = sample["ref_image_latents"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            ref_img_ids = sample["ref_img_ids"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        else:
            current_frame = sample.get("current_frame", sample.get("input_image"))
            if current_frame is None and video is not None:
                if video.ndim != 5:
                    raise ValueError(f"`sample['video']` must be [B,C,T,H,W], got {tuple(video.shape)}")
                current_frame = video[:, :, 0]
            if current_frame is None:
                raise ValueError("Ovis-U1 stack sample requires `current_frame`, `input_image`, or ref token fields.")
            condition_image = current_frame
            ref_tokens, ref_img_ids = self._encode_ovis_u1_image_tokens(current_frame, time_value=1.0)

        if condition_image is None:
            condition_image = sample.get("current_frame", sample.get("input_image"))
            if condition_image is None and video is not None:
                condition_image = video[:, :, 0]
        text_hidden_states, text_attention_mask = self._encode_ovis_u1_text(sample, condition_image=condition_image)
        if self.proprio_encoder is not None:
            text_hidden_states, text_attention_mask = self._append_proprio_to_context_if_enabled(
                context=text_hidden_states,
                context_mask=text_attention_mask,
                proprio=sample.get("proprio"),
                source="Ovis-U1 training sample",
            )
        action = sample["action"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        action_is_pad = sample.get("action_is_pad")
        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        action_dim_is_pad = sample.get("action_dim_is_pad")
        if action_dim_is_pad is not None:
            action_dim_is_pad = action_dim_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        return {
            "target_latent": target_tokens,
            "target_img_ids": target_img_ids,
            "ref_image_latents": ref_tokens,
            "ref_img_ids": ref_img_ids,
            "text_hidden_states": text_hidden_states,
            "text_attention_mask": text_attention_mask,
            "action": action,
            "action_is_pad": action_is_pad,
            "action_dim_is_pad": action_dim_is_pad,
        }

    @torch.no_grad()
    def _encode_flux2_image_tokens(
        self,
        image: torch.Tensor,
        *,
        time_value: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from .flux2_video_expert import Flux2VideoExpert

        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"`image` must be [B,3,H,W] or [3,H,W], got {tuple(image.shape)}")
        if image.shape[-2] % 16 != 0 or image.shape[-1] % 16 != 0:
            raise ValueError(f"FLUX.2 image spatial dims must be multiples of 16, got {tuple(image.shape[-2:])}")
        image = image.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        latents = self.vae.encode(image).to(dtype=self.torch_dtype)
        tokens = Flux2VideoExpert.pack_latents(latents)
        _, _, latent_h, latent_w = latents.shape
        ids = Flux2VideoExpert.build_img_ids(
            batch_size=int(latents.shape[0]),
            token_height=int(latent_h),
            token_width=int(latent_w),
            time_value=float(time_value),
            device=tokens.device,
            dtype=tokens.dtype,
        )
        return tokens, ids

    @torch.no_grad()
    def _decode_flux2_image_tokens(self, tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        from .flux2_video_expert import Flux2VideoExpert

        latent_h = int(height) // 16
        latent_w = int(width) // 16
        latents = Flux2VideoExpert.unpack_latents(tokens, latent_h, latent_w)
        image = self.vae.decode(latents.to(device=self.device, dtype=self.torch_dtype))
        return image.detach().float().clamp(-1, 1)

    @torch.no_grad()
    def _encode_flux2_text(self, sample) -> tuple[torch.Tensor, torch.Tensor]:
        cached_text_hidden_states = sample.get("text_hidden_states")
        if cached_text_hidden_states is not None:
            text_hidden_states = cached_text_hidden_states.to(
                device=self.device,
                dtype=self.torch_dtype,
                non_blocking=True,
            )
            text_attention_mask = sample.get("text_attention_mask")
            if text_attention_mask is None:
                raise ValueError("FLUX.2 cached `text_hidden_states` must be paired with `text_attention_mask`.")
            else:
                text_attention_mask = text_attention_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
            return text_hidden_states, text_attention_mask

        prompt = sample.get("instruction", sample.get("prompt", sample.get("task")))
        if prompt is None:
            raise ValueError(
                "FLUX.2 stack requires precomputed `text_hidden_states` or an `instruction`/`prompt`/`task`."
            )
        if self.text_encoder is None:
            raise ValueError("FLUX.2 online text encoding requires `load_text_encoder=true`.")
        video = sample.get("video")
        batch_size = int(video.shape[0]) if isinstance(video, torch.Tensor) and video.ndim == 5 else 1
        prompts = [prompt] * batch_size if isinstance(prompt, str) else list(prompt)
        return self._encode_flux2_prompts(prompts)

    @torch.no_grad()
    def _encode_flux2_prompts(self, prompts: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        if self.text_encoder is None:
            self._load_flux2_text_encoder_for_inference()
        if not hasattr(self.text_encoder, "tokenizer") or not hasattr(self.text_encoder, "model"):
            text_hidden = self.text_encoder(list(prompts)).to(device=self.device, dtype=self.torch_dtype)
            text_mask = torch.ones(text_hidden.shape[:2], device=self.device, dtype=torch.bool)
            return text_hidden, text_mask

        ensure_flux2_importable_path = getattr(self, "model_paths", {}).get("flux2_src")
        from .flux2_imports import ensure_flux2_importable

        ensure_flux2_importable(ensure_flux2_importable_path)
        from flux2.text_encoder import OUTPUT_LAYERS_QWEN3

        tokenizer = self.text_encoder.tokenizer
        model = self.text_encoder.model
        max_length = int(getattr(self.text_encoder, "max_length", 512))
        all_input_ids = []
        all_attention_masks = []
        for prompt in prompts:
            messages = [{"role": "user", "content": str(prompt)}]
            try:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            model_inputs = tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )
            all_input_ids.append(model_inputs["input_ids"])
            all_attention_masks.append(model_inputs["attention_mask"])

        input_ids = torch.cat(all_input_ids, dim=0).to(model.device)
        attention_mask = torch.cat(all_attention_masks, dim=0).to(model.device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden = torch.stack([outputs.hidden_states[k] for k in OUTPUT_LAYERS_QWEN3], dim=1)
        hidden = rearrange(hidden, "b c l d -> b l (c d)")
        return (
            hidden.to(device=self.device, dtype=self.torch_dtype),
            attention_mask.to(device=self.device, dtype=torch.bool),
        )

    def _load_flux2_text_encoder_for_inference(self) -> None:
        from types import SimpleNamespace
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_spec = getattr(self, "flux2_qwen3_model_spec", None)
        if model_spec is None:
            model_spec = getattr(self, "model_paths", {}).get("qwen3")
        if model_spec is None:
            if int(self.text_dim) == 7680:
                model_spec = "Qwen/Qwen3-4B"
            elif int(self.text_dim) == 12288:
                model_spec = "Qwen/Qwen3-8B"
            else:
                raise ValueError(f"Cannot infer FLUX.2 Qwen3 model spec from text_dim={self.text_dim}")
        logger.info("Lazy-loading FLUX.2 Qwen3 text encoder for prompt inference: %s", model_spec)
        qwen3_model = AutoModelForCausalLM.from_pretrained(
            model_spec,
            torch_dtype=self.torch_dtype,
        ).to(self.device).eval()
        self.text_encoder = SimpleNamespace(
            model=qwen3_model,
            tokenizer=AutoTokenizer.from_pretrained(model_spec),
            max_length=int(getattr(self, "qwen_context_len", 512)),
        )

    @torch.no_grad()
    def _prepare_flux2_infer_text(
        self,
        prompt: Optional[str],
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")
        if use_prompt:
            return self._encode_flux2_prompts([prompt])
        if context is None or context_mask is None:
            raise ValueError("`context` and `context_mask` must be both provided together.")
        if context.ndim == 2:
            context = context.unsqueeze(0)
        if context_mask.ndim == 1:
            context_mask = context_mask.unsqueeze(0)
        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        return (
            context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True),
            context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True),
        )

    def build_inputs_flux2(self, sample, tiled: bool = False):
        del tiled
        video = sample.get("video")
        if "target_latent" in sample and "target_img_ids" in sample:
            target_tokens = sample["target_latent"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            target_img_ids = sample["target_img_ids"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        else:
            next_frame = sample.get("next_frame", sample.get("target_image"))
            if next_frame is None and video is not None:
                if video.ndim != 5:
                    raise ValueError(f"`sample['video']` must be [B,C,T,H,W], got {tuple(video.shape)}")
                next_frame = video[:, :, -1]
            if next_frame is None:
                raise ValueError("FLUX.2 stack sample requires `next_frame`, `target_image`, or target token fields.")
            target_tokens, target_img_ids = self._encode_flux2_image_tokens(next_frame, time_value=0.0)

        if "ref_image_latents" in sample and "ref_img_ids" in sample:
            ref_tokens = sample["ref_image_latents"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            ref_img_ids = sample["ref_img_ids"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        else:
            current_frame = sample.get("current_frame", sample.get("input_image"))
            if current_frame is None and video is not None:
                if video.ndim != 5:
                    raise ValueError(f"`sample['video']` must be [B,C,T,H,W], got {tuple(video.shape)}")
                current_frame = video[:, :, 0]
            if current_frame is None:
                raise ValueError("FLUX.2 stack sample requires `current_frame`, `input_image`, or ref token fields.")
            ref_tokens, ref_img_ids = self._encode_flux2_image_tokens(current_frame, time_value=10.0)

        text_hidden_states, text_attention_mask = self._encode_flux2_text(sample)
        if self.proprio_encoder is not None:
            text_hidden_states, text_attention_mask = self._append_proprio_to_context_if_enabled(
                context=text_hidden_states,
                context_mask=text_attention_mask,
                proprio=sample.get("proprio"),
                source="FLUX.2 training sample",
            )
        action = sample["action"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        action_is_pad = sample.get("action_is_pad")
        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        action_dim_is_pad = sample.get("action_dim_is_pad")
        if action_dim_is_pad is not None:
            action_dim_is_pad = action_dim_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        return {
            "target_latent": target_tokens,
            "target_img_ids": target_img_ids,
            "ref_image_latents": ref_tokens,
            "ref_img_ids": ref_img_ids,
            "text_hidden_states": text_hidden_states,
            "text_attention_mask": text_attention_mask,
            "action": action,
            "action_is_pad": action_is_pad,
            "action_dim_is_pad": action_dim_is_pad,
        }

    def build_inputs_dim(self, sample, tiled: bool = False):
        del tiled
        video = sample.get("video")
        if video is None or video.ndim != 5:
            raise ValueError(f"DIM stack sample requires `video` [B,3,T,H,W], got {None if video is None else tuple(video.shape)}")
        if video.shape[1] != 3:
            raise ValueError(f"DIM stack expects RGB video, got {tuple(video.shape)}")
        source_image = sample.get("current_frame", sample.get("input_image"))
        target_image = sample.get("next_frame", sample.get("target_image"))
        if source_image is None:
            source_image = video[:, :, 0]
        if target_image is None:
            target_image = video[:, :, -1]
        if source_image.shape[-2] % int(self.video_expert.vae_downsample_rate) != 0 or source_image.shape[-1] % int(self.video_expert.vae_downsample_rate) != 0:
            raise ValueError(
                "DIM image spatial dims must be multiples of "
                f"{self.video_expert.vae_downsample_rate}, got {tuple(source_image.shape[-2:])}."
            )
        source_image = source_image.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        target_image = target_image.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        source_latent = self._encode_dim_image_latents(source_image)
        target_latent = self._encode_dim_image_latents(target_image)

        cached_dim_condition = sample.get("dim_condition_hidden_states")
        cached_dim_mask = sample.get("dim_condition_attention_mask")
        cached_text = sample.get("text_hidden_states")
        cached_text_mask = sample.get("text_attention_mask")
        if cached_dim_condition is not None and cached_dim_mask is not None:
            condition, condition_mask = (
                cached_dim_condition.to(device=self.device, dtype=self.torch_dtype, non_blocking=True),
                cached_dim_mask.to(device=self.device, dtype=torch.bool, non_blocking=True),
            )
        elif cached_text is not None and cached_text_mask is not None:
            condition, condition_mask = (
                cached_text.to(device=self.device, dtype=self.torch_dtype, non_blocking=True),
                cached_text_mask.to(device=self.device, dtype=torch.bool, non_blocking=True),
            )
        else:
            instruction = sample.get("instruction", sample.get("prompt"))
            if instruction is None:
                raise ValueError("DIM stack requires `instruction`/`prompt` for online Qwen image+text conditioning.")
            condition, condition_mask = self._encode_dim_condition_online(instruction, source_image)
        if self.proprio_encoder is not None:
            condition, condition_mask = self._append_proprio_to_context_if_enabled(
                context=condition,
                context_mask=condition_mask,
                proprio=sample.get("proprio"),
                source="DIM training sample",
            )

        action = sample["action"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        action_is_pad = sample.get("action_is_pad")
        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        action_dim_is_pad = sample.get("action_dim_is_pad")
        if action_dim_is_pad is not None:
            action_dim_is_pad = action_dim_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        return {
            "target_latent": target_latent,
            "source_latent": source_latent,
            "condition": condition,
            "condition_mask": condition_mask,
            "action": action,
            "action_is_pad": action_is_pad,
            "action_dim_is_pad": action_dim_is_pad,
        }

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)

        # video -> video
        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        # action -> action
        mask[video_seq_len:, video_seq_len:] = True
        # action -> first-frame video only
        first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
        mask[video_seq_len:, :first_frame_tokens] = True
        return mask

    @torch.no_grad()
    def _build_mot_attention_mask_omnigen2(
        self,
        encoder_seq_lengths: list[int],
        seq_lengths: list[int],
        max_video_seq_len: int,
        action_seq_len: int,
        device: torch.device,
        l_effective_ref_img_len: list[list[int]],
    ) -> torch.Tensor:
        batch_size = len(seq_lengths)
        total_seq_len = int(max_video_seq_len) + int(action_seq_len)
        action_start = int(max_video_seq_len)
        mask = torch.zeros(batch_size, total_seq_len, total_seq_len, dtype=torch.bool, device=device)
        for i, (cap_len, video_len) in enumerate(zip(encoder_seq_lengths, seq_lengths)):
            prefix_len = cap_len + sum(l_effective_ref_img_len[i])

            # video -> video: text/ref prefix cannot attend noisy target, matching
            # wan22 first_frame_causal where first-frame tokens cannot see future tokens.
            mask[i, :prefix_len, :prefix_len] = True
            mask[i, prefix_len:video_len, :video_len] = True
            # action -> action
            mask[i, action_start:, action_start:] = True
            # action -> text/ref prefix only. Text is included here because
            # OmniGen2-style ActionDiT has no separate text cross-attention.
            mask[i, action_start:, :prefix_len] = True
        return mask

    @torch.no_grad()
    def _build_mot_attention_mask_ovis_u1(
        self,
        batch_size: int,
        txt_len: int,
        target_len: int,
        cond_len: int,
        action_len: int,
        device: torch.device,
        text_attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        t0 = 0
        x0 = txt_len
        c0 = txt_len + target_len
        a0 = txt_len + target_len + cond_len
        total = a0 + action_len
        joint = torch.zeros(batch_size, total, total, dtype=torch.bool, device=device)
        # Text/condition are stable prefix states and do not attend noisy target.
        joint[:, t0:x0, t0:x0] = True
        joint[:, t0:x0, c0:a0] = True
        joint[:, c0:a0, t0:x0] = True
        joint[:, c0:a0, c0:a0] = True
        # Target/noisy image can use text, condition, and target image tokens.
        joint[:, x0:c0, t0:a0] = True
        # Action uses text, condition, and action self context, but not target/noisy image.
        joint[:, a0:total, t0:x0] = True
        joint[:, a0:total, c0:total] = True
        if text_attention_mask is not None:
            if text_attention_mask.ndim != 2 or tuple(text_attention_mask.shape) != (batch_size, txt_len):
                raise ValueError(
                    "`text_attention_mask` must be [B,txt_len], "
                    f"got {tuple(text_attention_mask.shape)} for B={batch_size}, txt_len={txt_len}"
                )
            text_valid = text_attention_mask.to(device=device, dtype=torch.bool)
            joint[:, :, t0:x0] &= text_valid[:, None, :]

        image_total = target_len + cond_len + action_len
        image_action_start = target_len + cond_len
        image = torch.zeros(batch_size, image_total, image_total, dtype=torch.bool, device=device)
        image[:, :target_len, : target_len + cond_len] = True
        image[:, target_len:image_action_start, target_len:image_action_start] = True
        image[:, image_action_start:, target_len:image_total] = True
        return {"joint_double": joint, "image_double": image, "single": joint.clone()}

    @torch.no_grad()
    def _build_mot_attention_mask_flux2(
        self,
        batch_size: int,
        txt_len: int,
        target_len: int,
        cond_len: int,
        action_len: int,
        device: torch.device,
        text_attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        t0 = 0
        r0 = txt_len
        x0 = txt_len + cond_len
        a0 = txt_len + cond_len + target_len
        total = a0 + action_len
        mask = torch.zeros(batch_size, total, total, dtype=torch.bool, device=device)
        # Stable text/ref prefix cannot attend to noisy target/action tokens.
        mask[:, t0:r0, t0:x0] = True
        mask[:, r0:x0, t0:x0] = True
        # Target/noisy image uses stable prefix and target self-context.
        mask[:, x0:a0, t0:a0] = True
        # Action uses stable prefix and action self-context, not target/noisy image.
        mask[:, a0:total, t0:x0] = True
        mask[:, a0:total, a0:total] = True
        if text_attention_mask is not None:
            if text_attention_mask.ndim != 2 or tuple(text_attention_mask.shape) != (batch_size, txt_len):
                raise ValueError(
                    "`text_attention_mask` must be [B,txt_len], "
                    f"got {tuple(text_attention_mask.shape)} for B={batch_size}, txt_len={txt_len}"
                )
            text_valid = text_attention_mask.to(device=device, dtype=torch.bool)
            mask[:, :, t0:r0] &= text_valid[:, None, :]
        return {"double_joint": mask, "single": mask.clone()}

    def _compute_video_loss_per_sample(
        self,
        pred_video: torch.Tensor,
        target_video: torch.Tensor,
        image_is_pad: Optional[torch.Tensor],
        include_initial_video_step: bool,
    ) -> torch.Tensor:
        video_loss_token = F.mse_loss(pred_video.float(), target_video.float(), reduction="none").mean(dim=(1, 3, 4))
        if image_is_pad is None:
            return video_loss_token.mean(dim=1)

        temporal_factor = int(self.vae.temporal_downsample_factor)
        if temporal_factor <= 0:
            raise ValueError(f"`vae.temporal_downsample_factor` must be positive, got {temporal_factor}.")
        if image_is_pad.shape[1] < 1:
            raise ValueError("`image_is_pad` must contain at least one frame.")
        if (image_is_pad.shape[1] - 1) % temporal_factor != 0:
            raise ValueError(
                "Cannot align `image_is_pad` with video latent steps: "
                f"num_frames={image_is_pad.shape[1]}, temporal_downsample_factor={temporal_factor}."
            )

        tail_is_pad = image_is_pad[:, 1:]
        latent_tail_is_pad = tail_is_pad.view(image_is_pad.shape[0], -1, temporal_factor).all(dim=2)
        if include_initial_video_step:
            video_is_pad = torch.cat([image_is_pad[:, :1], latent_tail_is_pad], dim=1)
        else:
            video_is_pad = latent_tail_is_pad

        if video_is_pad.shape[1] != video_loss_token.shape[1]:
            raise ValueError(
                "Video-loss mask shape mismatch: "
                f"mask steps={video_is_pad.shape[1]}, loss steps={video_loss_token.shape[1]}."
            )

        valid = (~video_is_pad).to(device=video_loss_token.device, dtype=video_loss_token.dtype)
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        return (video_loss_token * valid).sum(dim=1) / valid_sum

    def _compute_action_loss_per_sample(
        self,
        pred_action: torch.Tensor,
        target_action: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
        action_dim_is_pad: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        action_loss_dim = F.mse_loss(pred_action.float(), target_action.float(), reduction="none")
        if action_dim_is_pad is not None:
            dim_valid = (~action_dim_is_pad).to(device=action_loss_dim.device, dtype=action_loss_dim.dtype)
            dim_valid_sum = dim_valid.sum(dim=1).clamp(min=1.0).unsqueeze(1)
            action_loss_token = (action_loss_dim * dim_valid.unsqueeze(1)).sum(dim=2) / dim_valid_sum
        else:
            action_loss_token = action_loss_dim.mean(dim=2)

        if action_is_pad is not None:
            valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            return (action_loss_token * valid).sum(dim=1) / valid_sum
        return action_loss_token.mean(dim=1)

    def _training_loss_omnigen2(self, sample, tiled: bool = False):
        debug_every = int(os.environ.get("IMAGEWAM_DEBUG_OMNIGEN2_FORWARD_EVERY", "0") or "0")
        debug_threshold = float(os.environ.get("IMAGEWAM_DEBUG_OMNIGEN2_FORWARD_THRESHOLD", "0") or "0")
        debug_forward = debug_every > 0 or debug_threshold > 0
        debug_step = int(getattr(self, "_omnigen2_forward_debug_step", 0)) + 1
        self._omnigen2_forward_debug_step = debug_step
        debug_segments: list[tuple[str, float]] = []
        debug_meta: dict[str, Any] = {}

        def _sync_and_mark(name: str, start_time: float) -> float:
            if debug_forward and self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            now = time.perf_counter()
            if debug_forward:
                debug_segments.append((name, now - start_time))
            return now

        if debug_forward and self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()
        inputs = self.build_inputs_omnigen2(sample, tiled=tiled)
        t0 = _sync_and_mark("build_inputs", t0)
        target_latent = inputs["target_latent"]
        action = inputs["action"]
        batch_size = target_latent.shape[0]

        noise_video = torch.randn_like(target_latent)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=target_latent.dtype,
        )
        noisy_latent = self.train_video_scheduler.add_noise(target_latent, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(target_latent, noise_video, timestep_video)

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)
        t0 = _sync_and_mark("noise_sched", t0)

        video_pre = self.video_expert.pre_dit(
            x=noisy_latent,
            timestep=timestep_video,
            context=inputs["text_hidden_states"],
            context_mask=inputs["text_attention_mask"],
            ref_image_hidden_states=inputs["ref_image_latents"],
        )
        t0 = _sync_and_mark("video_pre", t0)
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
        )
        t0 = _sync_and_mark("action_pre", t0)

        video_tokens = video_pre["tokens"]
        action_tokens = action_pre["tokens"]
        attention_mask = self._build_mot_attention_mask_omnigen2(
            encoder_seq_lengths=video_pre["encoder_seq_lengths"],
            seq_lengths=video_pre["seq_lengths"],
            max_video_seq_len=video_tokens.shape[1],
            action_seq_len=action_tokens.shape[1],
            l_effective_ref_img_len=video_pre.get("l_effective_ref_img_len"),
            device=video_tokens.device,
        )
        if debug_forward:
            debug_meta = {
                "B": int(batch_size),
                "video_seq": int(video_tokens.shape[1]),
                "action_seq": int(action_tokens.shape[1]),
                "cap_min": int(min(video_pre["encoder_seq_lengths"])),
                "cap_max": int(max(video_pre["encoder_seq_lengths"])),
                "seq_min": int(min(video_pre["seq_lengths"])),
                "seq_max": int(max(video_pre["seq_lengths"])),
                "target_shape": tuple(target_latent.shape),
                "action_shape": tuple(action.shape),
            }
        t0 = _sync_and_mark("mask", t0)
        tokens_out = self.mot(
            embeds_all={"video": video_tokens, "action": action_tokens},
            attention_mask=attention_mask,
            freqs_all={"video": video_pre["freqs"], "action": action_pre["freqs"]},
            context_all={"video": None, "action": None},
            t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]},
        )
        t0 = _sync_and_mark("mot", t0)

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        video_loss_per_sample = F.mse_loss(pred_video.float(), target_video.float(), reduction="none").mean(dim=(1, 2, 3))
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            video_loss_per_sample.device, dtype=video_loss_per_sample.dtype
        )
        loss_video = (video_loss_per_sample * video_weight).mean()

        action_loss_per_sample = self._compute_action_loss_per_sample(
            pred_action=pred_action,
            target_action=target_action,
            action_is_pad=inputs["action_is_pad"],
            action_dim_is_pad=inputs.get("action_dim_is_pad"),
        )
        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        loss_action = (action_loss_per_sample * action_weight).mean()

        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        t0 = _sync_and_mark("post_loss", t0)
        if debug_forward:
            total_debug = sum(value for _, value in debug_segments)
            should_log = (debug_every > 0 and debug_step % debug_every == 0) or (
                debug_threshold > 0 and total_debug >= debug_threshold
            )
            phase_names = ["build_inputs", "noise_sched", "video_pre", "action_pre", "mask", "mot", "post_loss"]
            segment_dict = dict(debug_segments)
            durations = [float(segment_dict.get(name, 0.0)) for name in phase_names]
            arrivals = []
            cumulative = 0.0
            for value in durations:
                cumulative += value
                arrivals.append(cumulative)
            peak_gib = (
                torch.cuda.max_memory_allocated(self.device) / (1024**3)
                if self.device.type == "cuda"
                else 0.0
            )
            meta_values = [
                float(debug_step),
                1.0 if should_log else 0.0,
                float(debug_meta.get("B", 0)),
                float(debug_meta.get("video_seq", 0)),
                float(debug_meta.get("action_seq", 0)),
                float(debug_meta.get("cap_min", 0)),
                float(debug_meta.get("cap_max", 0)),
                float(debug_meta.get("seq_min", 0)),
                float(debug_meta.get("seq_max", 0)),
                float(peak_gib),
            ]
            profile_values = durations + arrivals + meta_values
            self._last_omnigen2_forward_profile = {
                "phase_names": phase_names,
                "num_phases": len(phase_names),
            }
            self._last_omnigen2_forward_profile_tensor = torch.tensor(
                profile_values,
                device=loss_total.device,
                dtype=torch.float32,
            )
        return loss_total, {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }

    def _training_loss_ovis_u1(self, sample, tiled: bool = False):
        inputs = self.build_inputs_ovis_u1(sample, tiled=tiled)
        target_latent = inputs["target_latent"]
        action = inputs["action"]
        batch_size = int(target_latent.shape[0])

        noise_video = torch.randn_like(target_latent)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=target_latent.dtype,
        )
        noisy_latent = self.train_video_scheduler.add_noise(target_latent, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(target_latent, noise_video, timestep_video)

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        video_pre = self.video_expert.pre_dit(
            x=noisy_latent,
            timestep=self._scheduler_timestep_to_unit(timestep_video, self.train_video_scheduler),
            context=inputs["text_hidden_states"],
            context_mask=inputs["text_attention_mask"],
            ref_image_hidden_states=inputs["ref_image_latents"],
            target_img_ids=inputs["target_img_ids"],
            ref_img_ids=inputs["ref_img_ids"],
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=self._scheduler_timestep_to_unit(timestep_action, self.train_action_scheduler),
        )
        attention_mask = self._build_mot_attention_mask_ovis_u1(
            batch_size=batch_size,
            txt_len=int(video_pre["txt_len"]),
            target_len=int(video_pre["target_len"]),
            cond_len=int(video_pre["cond_len"]),
            action_len=int(action_pre["tokens"].shape[1]),
            device=noisy_latent.device,
            text_attention_mask=video_pre["text_mask"],
        )
        tokens_out = self.mot(
            embeds_all={"video": video_pre["tokens"], "action": action_pre["tokens"]},
            attention_mask=attention_mask,
            freqs_all={"video": video_pre["freqs"]},
            context_all={"video": None, "action": {"ids": action_pre["ids"]}},
            t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]},
        )
        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        video_loss_per_sample = F.mse_loss(pred_video.float(), target_video.float(), reduction="none").flatten(1).mean(dim=1)
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            video_loss_per_sample.device,
            dtype=video_loss_per_sample.dtype,
        )
        loss_video = (video_loss_per_sample * video_weight).mean()
        action_loss_per_sample = self._compute_action_loss_per_sample(
            pred_action=pred_action,
            target_action=target_action,
            action_is_pad=inputs["action_is_pad"],
            action_dim_is_pad=inputs.get("action_dim_is_pad"),
        )
        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device,
            dtype=action_loss_per_sample.dtype,
        )
        loss_action = (action_loss_per_sample * action_weight).mean()
        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        return loss_total, {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }

    def _training_loss_flux2(self, sample, tiled: bool = False):
        inputs = self.build_inputs_flux2(sample, tiled=tiled)
        target_latent = inputs["target_latent"]
        action = inputs["action"]
        batch_size = int(target_latent.shape[0])

        noise_video = torch.randn_like(target_latent)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=target_latent.dtype,
        )
        noisy_latent = self.train_video_scheduler.add_noise(target_latent, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(target_latent, noise_video, timestep_video)

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        video_pre = self.video_expert.pre_dit(
            x=noisy_latent,
            timestep=self._scheduler_timestep_to_unit(timestep_video, self.train_video_scheduler),
            context=inputs["text_hidden_states"],
            context_mask=inputs["text_attention_mask"],
            ref_image_hidden_states=inputs["ref_image_latents"],
            target_img_ids=inputs["target_img_ids"],
            ref_img_ids=inputs["ref_img_ids"],
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=self._scheduler_timestep_to_unit(timestep_action, self.train_action_scheduler),
        )
        attention_mask = self._build_mot_attention_mask_flux2(
            batch_size=batch_size,
            txt_len=int(video_pre["txt_len"]),
            target_len=int(video_pre["target_len"]),
            cond_len=int(video_pre["cond_len"]),
            action_len=int(action_pre["tokens"].shape[1]),
            device=noisy_latent.device,
            text_attention_mask=video_pre["text_mask"],
        )
        tokens_out = self.mot(
            embeds_all={"video": video_pre["tokens"], "action": action_pre["tokens"]},
            attention_mask=attention_mask,
            freqs_all={"video": video_pre["freqs"]},
            context_all={"video": None, "action": {"ids": action_pre["ids"]}},
            t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]},
        )
        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        video_loss_per_sample = F.mse_loss(pred_video.float(), target_video.float(), reduction="none").flatten(1).mean(dim=1)
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            video_loss_per_sample.device,
            dtype=video_loss_per_sample.dtype,
        )
        loss_video = (video_loss_per_sample * video_weight).mean()
        action_loss_per_sample = self._compute_action_loss_per_sample(
            pred_action=pred_action,
            target_action=target_action,
            action_is_pad=inputs["action_is_pad"],
            action_dim_is_pad=inputs.get("action_dim_is_pad"),
        )
        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device,
            dtype=action_loss_per_sample.dtype,
        )
        loss_action = (action_loss_per_sample * action_weight).mean()
        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        return loss_total, {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }

    def _training_loss_dim(self, sample, tiled: bool = False):
        inputs = self.build_inputs_dim(sample, tiled=tiled)
        target_latent = inputs["target_latent"]
        source_latent = inputs["source_latent"]
        condition = inputs["condition"]
        condition_mask = inputs["condition_mask"]
        action = inputs["action"]
        batch_size = int(target_latent.shape[0])

        noise_video = torch.randn_like(target_latent)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=target_latent.dtype,
        )
        noisy_latent = self.train_video_scheduler.add_noise(target_latent, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(target_latent, noise_video, timestep_video)

        cond_latent = torch.randn_like(target_latent)
        timestep_cond = torch.full(
            (batch_size,),
            float(self.train_video_scheduler.num_train_timesteps),
            device=self.device,
            dtype=target_latent.dtype,
        )

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        video_pre = self.video_expert.pre_dit(
            x=noisy_latent,
            timestep=timestep_video,
            context=condition,
            context_mask=condition_mask,
            latents_condition=source_latent if self.dim_with_latents_condition else None,
        )
        cond_pre = self.video_expert.pre_dit(
            x=cond_latent,
            timestep=timestep_cond,
            context=condition,
            context_mask=condition_mask,
            latents_condition=source_latent if self.dim_with_latents_condition else None,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "cond": cond_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=None,
            freqs_all={},
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                    "hw": (video_pre["meta"]["token_height"], video_pre["meta"]["token_width"]),
                },
                "cond": {
                    "context": cond_pre["context"],
                    "mask": cond_pre["context_mask"],
                    "hw": (cond_pre["meta"]["token_height"], cond_pre["meta"]["token_width"]),
                },
                "action": {
                    "context_tokens": cond_pre["context_tokens"],
                    "context_mask_bool": cond_pre["context_mask_bool"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "cond": cond_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )
        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        video_loss_per_sample = F.mse_loss(pred_video.float(), target_video.float(), reduction="none").flatten(1).mean(dim=1)
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            video_loss_per_sample.device,
            dtype=video_loss_per_sample.dtype,
        )
        loss_video = (video_loss_per_sample * video_weight).mean()
        action_loss_per_sample = self._compute_action_loss_per_sample(
            pred_action=pred_action,
            target_action=target_action,
            action_is_pad=inputs["action_is_pad"],
            action_dim_is_pad=inputs.get("action_dim_is_pad"),
        )
        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device,
            dtype=action_loss_per_sample.dtype,
        )
        loss_action = (action_loss_per_sample * action_weight).mean()
        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        return loss_total, {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }

    def training_loss(self, sample, tiled: bool = False):
        if self.stack == "dim":
            return self._training_loss_dim(sample, tiled=tiled)
        if self.stack == "flux2":
            return self._training_loss_flux2(sample, tiled=tiled)
        if self.stack == "ovis_u1":
            return self._training_loss_ovis_u1(sample, tiled=tiled)
        if self.stack == "omnigen2":
            return self._training_loss_omnigen2(sample, tiled=tiled)

        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]

        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)

        if inputs["first_frame_latents"] is not None:
            latents[:, :, 0:1] = inputs["first_frame_latents"]

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        video_pre = self.video_expert.pre_dit(
            x=latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )

        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        video_tokens = video_pre["tokens"]
        action_tokens = action_pre["tokens"]

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_tokens.shape[1],
            action_seq_len=action_tokens.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_tokens.device,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_tokens,
                "action": action_tokens,
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)

        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        include_initial_video_step = inputs["first_frame_latents"] is None
        if inputs["first_frame_latents"] is not None:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]

        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=include_initial_video_step,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )
        loss_video = (loss_video_per_sample * video_weight).mean()

        action_loss_per_sample = self._compute_action_loss_per_sample(
            pred_action=pred_action,
            target_action=target_action,
            action_is_pad=action_is_pad,
            action_dim_is_pad=inputs.get("action_dim_is_pad"),
        )

        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        loss_action = (action_loss_per_sample * action_weight).mean()

        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }
        return loss_total, loss_dict

    @torch.no_grad()
    def _predict_joint_noise(
        self,
        latents_video: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        gt_action: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_pre = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=gt_action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )

        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_video, pred_action

    @torch.no_grad()
    def _predict_action_noise(
        self,
        first_frame_latents: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
    ) -> torch.Tensor:
        timestep_video = torch.zeros_like(timestep_action, dtype=first_frame_latents.dtype, device=self.device)
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_action

    @torch.no_grad()
    def _predict_action_noise_with_cache(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> torch.Tensor:
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
        )
        return self.action_expert.post_dit(action_tokens, action_pre)

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None, # NOTE: this is gt action for conditioning videos, not for action expert
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        test_action_with_infer_action: bool = True,
        tau_star: Optional[float] = 0.0,
    ) -> dict[str, Any]:
        self.eval()
        if test_action_with_infer_action:
            if seed is None:
                raise ValueError("`test_action_with_infer_action=True` requires non-null `seed`.")
            action_only_out = self.infer_action(
                prompt=prompt,
                input_image=input_image.clone(),
                action_horizon=action_horizon,
                context=context.clone() if context is not None else None,
                context_mask=context_mask.clone() if context_mask is not None else None,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tiled=tiled,
                proprio=proprio.clone() if proprio is not None else None,
                tau_star=tau_star,
            )["action"]
        
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(height, width, num_video_frames)
        if (checked_h, checked_w) != (height, width):
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if checked_t != num_video_frames:
            raise ValueError(
                f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}"
            )
        if action is not None:
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3 or action.shape[0] != 1 or action.shape[1] != action_horizon:
                # NOTE: This enforces action condition to have the same shape as action horizon to predict, which may be unnecessary
                raise ValueError(
                    f"`action` must have shape [1, T, a_dim] or [T, a_dim], got {tuple(action.shape)} with action_horizon={action_horizon}"
                )
            action = action.to(device=self.device, dtype=self.torch_dtype)
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor

        video_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_video = torch.randn(
            (1, self.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        latents_video[:, :, 0:1] = first_frame_latents.clone()
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        infer_timesteps_video, infer_deltas_video = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_video, step_delta_video, step_t_action, step_delta_action in zip(
            infer_timesteps_video,
            infer_deltas_video,
            infer_timesteps_action,
            infer_deltas_action,
        ):
            timestep_video = step_t_video.unsqueeze(0).to(dtype=latents_video.dtype, device=self.device)
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            pred_video_posi, pred_action_posi = self._predict_joint_noise(
                latents_video=latents_video,
                latents_action=latents_action,
                timestep_video=timestep_video,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                gt_action=action,
            )
            pred_video = pred_video_posi
            pred_action = pred_action_posi

            latents_video = self.infer_video_scheduler.step(pred_video, step_delta_video, latents_video)
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)
            latents_video[:, :, 0:1] = first_frame_latents.clone()

        action_out = latents_action[0].detach().to(device="cpu", dtype=torch.float32)
        if test_action_with_infer_action:
            if not torch.allclose(action_out, action_only_out, atol=1e-2, rtol=1e-2):
                max_abs_diff = (action_out - action_only_out).abs().max().item()
                logger.warning(
                    f"Action from infer_joint and infer_action differ with max abs diff {max_abs_diff:.6f}. "
                )

        return {
            "video": self._decode_latents(latents_video, tiled=tiled),
            "action": action_out,
        }

    def _tau_star_to_video_timestep(
        self,
        tau_star: Optional[float],
        *,
        batch_size: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        tau = 0.0 if tau_star is None else float(tau_star)
        if tau < 0.0 or tau > 1.0:
            raise ValueError(f"`tau_star` must be in [0, 1], got {tau}.")
        timestep = tau * float(self.infer_video_scheduler.num_train_timesteps)
        return torch.full((batch_size,), timestep, dtype=dtype, device=self.device)

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        profile_infer_timing: bool = False,
        tau_star: Optional[float] = 0.0,
    ) -> dict[str, Any]:
        if self.stack == "dim":
            return self.infer_action_dim(
                prompt=prompt,
                input_image=input_image,
                action_horizon=action_horizon,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
            )
        if self.stack == "flux2":
            return self.infer_action_flux2(
                prompt=prompt,
                input_image=input_image,
                action_horizon=action_horizon,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tau_star=tau_star,
            )
        if self.stack == "ovis_u1":
            return self.infer_action_ovis_u1(
                prompt=prompt,
                input_image=input_image,
                action_horizon=action_horizon,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tau_star=tau_star,
            )
        if self.stack == "omnigen2":
            return self.infer_action_omnigen2(
                prompt=prompt,
                input_image=input_image,
                action_horizon=action_horizon,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                profile_infer_timing=profile_infer_timing,
                tau_star=tau_star,
            )

        self.eval()
        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError(
                "`infer_action` requires `video_attention_mask_mode='first_frame_causal'`."
            )

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        timestep_video = self._tau_star_to_video_timestep(
            tau_star,
            batch_size=first_frame_latents.shape[0],
            dtype=first_frame_latents.dtype,
        )
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        patch_size = tuple(getattr(self.video_expert, "patch_size", (1, 1, 1)))
        condition_grid = (
            int(first_frame_latents.shape[3]) // int(patch_size[1]),
            int(first_frame_latents.shape[4]) // int(patch_size[2]),
        )
        self._configure_action_attention_capture(
            condition_slice=(0, video_seq_len),
            condition_grid=condition_grid,
            prefix_len=video_seq_len,
            action_len=int(latents_action.shape[1]),
            metadata={
                "stack": str(self.stack),
                "source": "imagewam_infer_action",
                "condition": "first_frame_latents",
                "input_size": [int(height), int(width)],
            },
        )
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
        )

        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for action_step_idx, (step_t_action, step_delta_action) in enumerate(zip(infer_timesteps_action, infer_deltas_action)):
            self._start_action_attention_capture_step(action_step_idx)
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            pred_action_posi = self._predict_action_noise_with_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )
            pred_action = pred_action_posi

            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)

        return {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }

    @torch.no_grad()
    def infer_action_dim(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
    ) -> dict[str, Any]:
        self.eval()
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(f"`input_image` must be [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}")
        downsample = int(self.video_expert.vae_downsample_rate)
        if input_image.shape[-2] % downsample != 0 or input_image.shape[-1] % downsample != 0:
            raise ValueError(
                f"DIM input spatial dims must be multiples of {downsample}, got {tuple(input_image.shape[-2:])}."
            )
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        batch_size = int(input_image.shape[0])

        if context is not None or context_mask is not None:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            condition = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            condition_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        else:
            if prompt is None:
                raise ValueError("DIM action inference requires `prompt` unless precomputed context is provided.")
            condition, condition_mask = self._encode_dim_condition_online(prompt, input_image)
        if self.proprio_encoder is not None or proprio is not None:
            condition, condition_mask = self._append_proprio_to_context_if_enabled(
                context=condition,
                context_mask=condition_mask,
                proprio=proprio,
                source="DIM action inference",
            )

        source_latent = self._encode_dim_image_latents(input_image)
        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        cond_latent = torch.randn(
            source_latent.shape,
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_action = torch.randn(
            (batch_size, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        timestep_cond = torch.full(
            (batch_size,),
            float(self.infer_video_scheduler.num_train_timesteps),
            device=self.device,
            dtype=source_latent.dtype,
        )
        cond_pre = self.video_expert.pre_dit(
            x=cond_latent,
            timestep=timestep_cond,
            context=condition,
            context_mask=condition_mask,
            latents_condition=source_latent if self.dim_with_latents_condition else None,
        )
        cond_cache = self.mot.prefill_sana_condition_cache(
            cond_tokens=cond_pre["tokens"],
            cond_t_mod=cond_pre["t_mod"],
            cond_context_payload={
                "context": cond_pre["context"],
                "mask": cond_pre["context_mask"],
                "hw": (cond_pre["meta"]["token_height"], cond_pre["meta"]["token_width"]),
            },
        )
        infer_timesteps, infer_deltas = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        action_context_payload = {
            "context_tokens": cond_pre["context_tokens"],
            "context_mask_bool": cond_pre["context_mask_bool"],
        }
        for step_t, step_delta in zip(infer_timesteps, infer_deltas):
            timestep_action = step_t.expand(batch_size).to(dtype=latents_action.dtype, device=self.device)
            action_pre = self.action_expert.pre_dit(
                action_tokens=latents_action,
                timestep=timestep_action,
            )
            action_tokens = self.mot.forward_sana_action_with_condition_cache(
                action_tokens=action_pre["tokens"],
                action_t_mod=action_pre["t_mod"],
                action_context_payload=action_context_payload,
                condition_cache=cond_cache,
            )
            pred_action = self.action_expert.post_dit(action_tokens, action_pre)
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta, latents_action)
        return {"action": latents_action[0].detach().to(device="cpu", dtype=torch.float32)}

    @torch.no_grad()
    def infer_video_dim(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
    ) -> dict[str, Any]:
        self.eval()
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(f"`input_image` must be [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}")
        downsample = int(self.video_expert.vae_downsample_rate)
        if input_image.shape[-2] % downsample != 0 or input_image.shape[-1] % downsample != 0:
            raise ValueError(
                f"DIM input spatial dims must be multiples of {downsample}, got {tuple(input_image.shape[-2:])}."
            )
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        batch_size = int(input_image.shape[0])

        if context is not None or context_mask is not None:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            condition = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            condition_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        else:
            if prompt is None:
                raise ValueError("DIM video inference requires `prompt` unless precomputed context is provided.")
            condition, condition_mask = self._encode_dim_condition_online(prompt, input_image)
        if self.proprio_encoder is not None or proprio is not None:
            condition, condition_mask = self._append_proprio_to_context_if_enabled(
                context=condition,
                context_mask=condition_mask,
                proprio=proprio,
                source="DIM video inference",
            )

        source_latent = self._encode_dim_image_latents(input_image)
        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_video = torch.randn(
            source_latent.shape,
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        infer_timesteps, infer_deltas = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        for step_t, step_delta in zip(infer_timesteps, infer_deltas):
            timestep_video = step_t.expand(batch_size).to(dtype=latents_video.dtype, device=self.device)
            video_pre = self.video_expert.pre_dit(
                x=latents_video,
                timestep=timestep_video,
                context=condition,
                context_mask=condition_mask,
                latents_condition=source_latent if self.dim_with_latents_condition else None,
            )
            x = video_pre["tokens"]
            context_payload = {
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
                "hw": (video_pre["meta"]["token_height"], video_pre["meta"]["token_width"]),
            }
            hw = tuple(context_payload["hw"])
            for layer_idx, block in enumerate(self.video_expert.blocks):
                state = self.mot._sana_video_io(block, x, video_pre["t_mod"], hw)
                attn = self.mot._sana_linear_attention(block.attn, state["q"], state["k"], state["v"])
                x = self.mot._sana_video_post(block, attn, state, context_payload)
            pred_video = self.video_expert.post_dit(x, video_pre)
            latents_video = self.infer_video_scheduler.step(pred_video, step_delta, latents_video)

        image = self._decode_dim_image_latents(latents_video)
        return {"image": image[0].detach().cpu()}

    @torch.no_grad()
    def infer_dim_separate(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
    ) -> dict[str, Any]:
        action_out = self.infer_action_dim(
            prompt=prompt,
            input_image=input_image,
            action_horizon=action_horizon,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
        )
        video_out = self.infer_video_dim(
            prompt=prompt,
            input_image=input_image,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
        )
        image = video_out["image"].detach().float().clamp(-1, 1)
        image = ((image + 1.0) * 127.5).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        return {"action": action_out["action"], "video": [Image.fromarray(image)]}

    @torch.no_grad()
    def infer_action_flux2(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tau_star: Optional[float] = 0.0,
    ) -> dict[str, Any]:
        self.eval()
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(f"`input_image` must be [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}")

        text_hidden, text_mask = self._prepare_flux2_infer_text(prompt, context, context_mask)
        if self.proprio_encoder is not None or proprio is not None:
            text_hidden, text_mask = self._append_proprio_to_context_if_enabled(
                context=text_hidden,
                context_mask=text_mask,
                proprio=proprio,
                source="FLUX.2 action inference",
            )
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        ref_tokens, ref_img_ids = self._encode_flux2_image_tokens(input_image, time_value=10.0)
        batch_size = int(ref_tokens.shape[0])
        empty_target = ref_tokens.new_zeros(batch_size, 0, ref_tokens.shape[-1])
        empty_target_ids = ref_img_ids.new_zeros(batch_size, 0, ref_img_ids.shape[-1])

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (batch_size, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        video_timestep = self._tau_star_to_video_timestep(
            tau_star,
            batch_size=batch_size,
            dtype=ref_tokens.dtype,
        )
        video_pre = self.video_expert.pre_dit(
            x=empty_target,
            timestep=video_timestep,
            context=text_hidden,
            context_mask=text_mask,
            ref_image_hidden_states=ref_tokens,
            target_img_ids=empty_target_ids,
            ref_img_ids=ref_img_ids,
        )
        prefix_attention_mask = self._build_mot_attention_mask_flux2(
            batch_size=batch_size,
            txt_len=int(video_pre["txt_len"]),
            target_len=0,
            cond_len=int(video_pre["cond_len"]),
            action_len=0,
            device=latents_action.device,
            text_attention_mask=video_pre["text_mask"],
        )
        video_kv_cache = self.mot.prefill_flux2_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            attention_mask=prefix_attention_mask,
        )
        full_attention_mask = self._build_mot_attention_mask_flux2(
            batch_size=batch_size,
            txt_len=int(video_pre["txt_len"]),
            target_len=0,
            cond_len=int(video_pre["cond_len"]),
            action_len=int(latents_action.shape[1]),
            device=latents_action.device,
            text_attention_mask=video_pre["text_mask"],
        )
        prefix_len = int(video_pre["txt_len"]) + int(video_pre["cond_len"])
        self._configure_action_attention_capture(
            condition_slice=(int(video_pre["txt_len"]), int(video_pre["txt_len"]) + int(video_pre["cond_len"])),
            condition_grid=(int(input_image.shape[-2]) // 16, int(input_image.shape[-1]) // 16),
            prefix_len=prefix_len,
            action_len=int(latents_action.shape[1]),
            metadata={
                "stack": str(self.stack),
                "source": "flux2_infer_action",
                "condition": "ref_image_tokens",
                "txt_len": int(video_pre["txt_len"]),
                "cond_len": int(video_pre["cond_len"]),
                "input_size": [int(input_image.shape[-2]), int(input_image.shape[-1])],
            },
        )
        for action_step_idx, (step_t_action, step_delta_action) in enumerate(zip(infer_timesteps_action, infer_deltas_action)):
            self._start_action_attention_capture_step(action_step_idx)
            timestep_action = step_t_action.expand(batch_size).to(dtype=latents_action.dtype, device=self.device)
            action_pre = self.action_expert.pre_dit(
                action_tokens=latents_action,
                timestep=self._scheduler_timestep_to_unit(timestep_action, self.infer_action_scheduler),
            )
            action_tokens = self.mot.forward_action_with_video_cache(
                action_tokens=action_pre["tokens"],
                action_freqs=None,
                action_t_mod=action_pre["t_mod"],
                action_context_payload={"ids": action_pre["ids"]},
                video_kv_cache=video_kv_cache,
                attention_mask=full_attention_mask,
                video_seq_len=prefix_len,
            )
            pred_action = self.action_expert.post_dit(action_tokens, action_pre)
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)

        return {"action": latents_action[0].detach().to(device="cpu", dtype=torch.float32)}

    def _forward_flux2_video_only(
        self,
        video_pre: dict[str, Any],
        attention_mask: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        video_state = video_pre["tokens"]
        if not isinstance(video_state, dict):
            raise ValueError("FLUX.2 video tokens must be a dict with `txt` and `img` tensors.")

        txt = video_state["txt"]
        img = video_state["img"]
        video_freqs = video_pre["freqs"]
        txt_pe = video_freqs["txt"]
        img_pe = video_freqs["img"]
        video_t_mod = video_pre["t_mod"]
        video_expert = self.video_expert

        def _flatten_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.transpose(1, 2).reshape(tensor.shape[0], tensor.shape[2], tensor.shape[1] * tensor.shape[3])

        from flux2.model import apply_rope

        for layer_idx in range(int(getattr(video_expert, "double_layers"))):
            block = video_expert.double_blocks[layer_idx]
            q, k, v, pe_full, num_txt_tokens, mods = block._prepare_qkv(
                img,
                txt,
                img_pe,
                txt_pe,
                video_t_mod["double_img"],
                video_t_mod["double_txt"],
            )
            q, k = apply_rope(q, k, pe_full)
            mixed = self.mot._mixed_attention(
                _flatten_heads(q),
                _flatten_heads(k),
                _flatten_heads(v),
                attention_mask["double_joint"],
            )
            txt_attn, img_attn = torch.split(mixed, [num_txt_tokens, img.shape[1]], dim=1)
            img, txt = block._apply_residuals(img, txt, img_attn, txt_attn, mods)

        video_stream = torch.cat([txt, img], dim=1)
        stream_pe = torch.cat([txt_pe, img_pe], dim=2)
        for layer_idx in range(int(getattr(video_expert, "single_layers"))):
            block = video_expert.single_blocks[layer_idx]
            q, k, v, mlp, gate = block._qkv(video_stream, video_t_mod["single"])
            q, k = apply_rope(q, k, stream_pe)
            mixed = self.mot._mixed_attention(
                _flatten_heads(q),
                _flatten_heads(k),
                _flatten_heads(v),
                attention_mask["single"],
            )
            video_stream = block._out(video_stream, mixed, mlp, gate)

        txt_len = int(txt.shape[1])
        txt, img = video_stream[:, :txt_len], video_stream[:, txt_len:]
        return {"txt": txt, "img": img}

    @torch.no_grad()
    def infer_video_flux2(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
    ) -> dict[str, Any]:
        from .flux2_video_expert import Flux2VideoExpert

        self.eval()
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(f"`input_image` must be [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}")
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"FLUX.2 image spatial dims must be multiples of 16, got HxW=({height},{width})")

        text_hidden, text_mask = self._prepare_flux2_infer_text(prompt, context, context_mask)
        if self.proprio_encoder is not None or proprio is not None:
            text_hidden, text_mask = self._append_proprio_to_context_if_enabled(
                context=text_hidden,
                context_mask=text_mask,
                proprio=proprio,
                source="FLUX.2 video inference",
            )
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        ref_tokens, ref_img_ids = self._encode_flux2_image_tokens(input_image, time_value=10.0)
        batch_size = int(ref_tokens.shape[0])
        latent_h = int(height) // 16
        latent_w = int(width) // 16

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_video = torch.randn(
            ref_tokens.shape,
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        target_img_ids = Flux2VideoExpert.build_img_ids(
            batch_size=batch_size,
            token_height=latent_h,
            token_width=latent_w,
            time_value=0.0,
            device=self.device,
            dtype=self.torch_dtype,
        )

        infer_timesteps, infer_deltas = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        for step_t, step_delta in zip(infer_timesteps, infer_deltas):
            timestep_video = step_t.expand(batch_size).to(dtype=latents_video.dtype, device=self.device)
            video_pre = self.video_expert.pre_dit(
                x=latents_video,
                timestep=self._scheduler_timestep_to_unit(timestep_video, self.infer_video_scheduler),
                context=text_hidden,
                context_mask=text_mask,
                ref_image_hidden_states=ref_tokens,
                target_img_ids=target_img_ids,
                ref_img_ids=ref_img_ids,
            )
            attention_mask = self._build_mot_attention_mask_flux2(
                batch_size=batch_size,
                txt_len=int(video_pre["txt_len"]),
                target_len=int(video_pre["target_len"]),
                cond_len=int(video_pre["cond_len"]),
                action_len=0,
                device=latents_video.device,
                text_attention_mask=video_pre["text_mask"],
            )
            tokens_out = self._forward_flux2_video_only(video_pre, attention_mask)
            pred_video = self.video_expert.post_dit(tokens_out, video_pre)
            latents_video = self.infer_video_scheduler.step(pred_video, step_delta, latents_video)

        image = self._decode_flux2_image_tokens(latents_video, height=height, width=width)
        return {"image": image[0].detach().cpu()}

    @torch.no_grad()
    def infer_action_omnigen2(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        profile_infer_timing: bool = False,
        tau_star: Optional[float] = 0.0,
    ) -> dict[str, Any]:
        self.eval()
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(f"`input_image` must be [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}")
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`input_image` spatial dims must be multiples of 16, got HxW=({height},{width})")

        profile_segments: list[tuple[str, float]] = []
        profile_enabled = bool(profile_infer_timing)
        profile_last = 0.0

        def _sync_profile_device() -> None:
            if not profile_enabled:
                return
            device = torch.device(self.device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        def _mark_profile(name: str) -> None:
            nonlocal profile_last
            if not profile_enabled:
                return
            _sync_profile_device()
            now = time.perf_counter()
            profile_segments.append((name, now - profile_last))
            profile_last = now

        if profile_enabled:
            _sync_profile_device()
            profile_last = time.perf_counter()

        text_hidden, text_mask = self._prepare_omnigen2_infer_text(prompt, context, context_mask)
        _mark_profile("prepare_text_s")
        if self.proprio_encoder is not None or proprio is not None:
            text_hidden, text_mask = self._append_proprio_to_context_if_enabled(
                context=text_hidden,
                context_mask=text_mask,
                proprio=proprio,
                source="OmniGen2 action inference",
            )
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        ref_latent = self._encode_omnigen2_image_latents(input_image)
        _mark_profile("encode_image_latents_s")
        batch_size = int(ref_latent.shape[0])

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (batch_size, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        _mark_profile("sample_latents_s")

        timestep_video = self._tau_star_to_video_timestep(
            tau_star,
            batch_size=batch_size,
            dtype=ref_latent.dtype,
        )
        _mark_profile("prepare_video_timestep_s")
        video_pre = self.video_expert.pre_dit(
            x=None,
            timestep=timestep_video,
            context=text_hidden,
            context_mask=text_mask,
            ref_image_hidden_states=ref_latent,
        )
        _mark_profile("video_pre_dit_s")
        video_seq_len = int(video_pre["tokens"].shape[1])
        text_len = int(video_pre["encoder_seq_lengths"][0])
        ref_len = int(sum(video_pre["l_effective_ref_img_len"][0]))
        self._configure_action_attention_capture(
            condition_slice=(text_len, text_len + ref_len),
            condition_grid=(int(input_image.shape[-2]) // 16, int(input_image.shape[-1]) // 16),
            prefix_len=video_seq_len,
            action_len=int(latents_action.shape[1]),
            metadata={
                "stack": str(self.stack),
                "source": "omnigen2_infer_action",
                "condition": "ref_image_latents",
                "text_len": text_len,
                "ref_len": ref_len,
                "input_size": [int(input_image.shape[-2]), int(input_image.shape[-1])],
            },
        )
        attention_mask = self._build_mot_attention_mask_omnigen2(
            encoder_seq_lengths=video_pre["encoder_seq_lengths"],
            seq_lengths=video_pre["seq_lengths"],
            max_video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            l_effective_ref_img_len=video_pre["l_effective_ref_img_len"],
            device=video_pre["tokens"].device,
        )
        _mark_profile("build_attention_mask_s")
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload=None,
            video_attention_mask=attention_mask[:, :video_seq_len, :video_seq_len],
        )
        _mark_profile("prefill_video_cache_s")

        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        _mark_profile("build_action_schedule_s")
        action_predict_times: list[float] = []
        action_loop_start = 0.0
        timing_payload = None
        if profile_enabled:
            _sync_profile_device()
            action_loop_start = time.perf_counter()
        for action_step_idx, (step_t_action, step_delta_action) in enumerate(zip(infer_timesteps_action, infer_deltas_action)):
            self._start_action_attention_capture_step(action_step_idx)
            timestep_action = step_t_action.expand(batch_size).to(dtype=latents_action.dtype, device=self.device)
            if profile_enabled:
                _sync_profile_device()
                action_predict_start = time.perf_counter()
            pred_action = self._predict_action_noise_with_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=text_hidden,
                context_mask=text_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )
            if profile_enabled:
                _sync_profile_device()
                action_predict_times.append(time.perf_counter() - action_predict_start)
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)
        if profile_enabled:
            _sync_profile_device()
            action_denoise_loop_s = time.perf_counter() - action_loop_start
            profile_segments.append(("action_denoise_loop_s", action_denoise_loop_s))
            action_predict_total_s = sum(action_predict_times)
            action_predict_count = len(action_predict_times)
            action_predict_avg_s = action_predict_total_s / action_predict_count if action_predict_count > 0 else 0.0
            action_predict_min_s = min(action_predict_times) if action_predict_times else 0.0
            action_predict_max_s = max(action_predict_times) if action_predict_times else 0.0
            total_profiled_s = sum(value for _, value in profile_segments)
            timing_payload = {
                "segments": {name: float(value) for name, value in profile_segments},
                "action_horizon": int(action_horizon),
                "num_inference_steps": int(action_predict_count),
                "action_predict_total_s": float(action_predict_total_s),
                "action_predict_avg_s": float(action_predict_avg_s),
                "action_predict_min_s": float(action_predict_min_s),
                "action_predict_max_s": float(action_predict_max_s),
                "total_profiled_s": float(total_profiled_s),
            }

        result = {"action": latents_action[0].detach().to(device="cpu", dtype=torch.float32)}
        if timing_payload is not None:
            result["timing"] = timing_payload
        return result

    @torch.no_grad()
    def infer_action_ovis_u1(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tau_star: Optional[float] = 0.0,
    ) -> dict[str, Any]:
        self.eval()
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(f"`input_image` must be [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}")
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`input_image` spatial dims must be multiples of 16, got HxW=({height},{width})")

        text_hidden, text_mask = self._prepare_ovis_u1_infer_text(prompt, input_image, context, context_mask)
        if self.proprio_encoder is not None or proprio is not None:
            text_hidden, text_mask = self._append_proprio_to_context_if_enabled(
                context=text_hidden,
                context_mask=text_mask,
                proprio=proprio,
                source="Ovis-U1 action inference",
            )
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        ref_tokens, ref_img_ids = self._encode_ovis_u1_image_tokens(input_image, time_value=1.0)
        batch_size = int(ref_tokens.shape[0])
        empty_target = torch.empty(
            batch_size,
            0,
            int(self.video_expert.transformer.in_channels),
            device=self.device,
            dtype=self.torch_dtype,
        )
        empty_target_ids = torch.empty(batch_size, 0, 3, device=self.device, dtype=self.torch_dtype)
        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (batch_size, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        timestep_video = self._tau_star_to_video_timestep(
            tau_star,
            batch_size=batch_size,
            dtype=self.torch_dtype,
        )
        video_pre = self.video_expert.pre_dit(
            x=empty_target,
            timestep=timestep_video,
            context=text_hidden,
            context_mask=text_mask,
            ref_image_hidden_states=ref_tokens,
            target_img_ids=empty_target_ids,
            ref_img_ids=ref_img_ids,
        )
        prefix_attention_mask = self._build_mot_attention_mask_ovis_u1(
            batch_size=batch_size,
            txt_len=int(video_pre["txt_len"]),
            target_len=int(video_pre["target_len"]),
            cond_len=int(video_pre["cond_len"]),
            action_len=0,
            device=self.device,
            text_attention_mask=video_pre["text_mask"],
        )
        video_kv_cache = self.mot.prefill_yak_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            attention_mask=prefix_attention_mask,
        )
        full_attention_mask = self._build_mot_attention_mask_ovis_u1(
            batch_size=batch_size,
            txt_len=int(video_pre["txt_len"]),
            target_len=int(video_pre["target_len"]),
            cond_len=int(video_pre["cond_len"]),
            action_len=int(latents_action.shape[1]),
            device=self.device,
            text_attention_mask=video_pre["text_mask"],
        )
        prefix_len = int(video_pre["txt_len"]) + int(video_pre["cond_len"])
        cond_len = int(video_pre["cond_len"])
        action_len = int(latents_action.shape[1])
        action_attention_mask = {
            "joint_double": full_attention_mask["joint_double"][:, prefix_len : prefix_len + action_len, : prefix_len + action_len],
            "image_double": full_attention_mask["image_double"][:, cond_len : cond_len + action_len, : cond_len + action_len],
            "single": full_attention_mask["single"][:, prefix_len : prefix_len + action_len, : prefix_len + action_len],
        }
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
            timestep_action = step_t_action.expand(batch_size).to(dtype=latents_action.dtype, device=self.device)
            action_pre = self.action_expert.pre_dit(
                action_tokens=latents_action,
                timestep=self._scheduler_timestep_to_unit(timestep_action, self.infer_action_scheduler),
            )
            action_tokens = self.mot.forward_action_with_video_cache(
                action_tokens=action_pre["tokens"],
                action_freqs=None,
                action_t_mod=action_pre["t_mod"],
                action_context_payload={"ids": action_pre["ids"]},
                video_kv_cache=video_kv_cache,
                attention_mask=action_attention_mask,
                video_seq_len=prefix_len,
            )
            pred_action = self.action_expert.post_dit(action_tokens, action_pre)
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)

        return {"action": latents_action[0].detach().to(device="cpu", dtype=torch.float32)}

    @torch.no_grad()
    def infer_video_ovis_u1(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
    ) -> dict[str, Any]:
        from .ovis_u1_video_expert import OvisU1VideoExpert

        self.eval()
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(f"`input_image` must be [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}")
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`input_image` spatial dims must be multiples of 16, got HxW=({height},{width})")
        text_hidden, text_mask = self._prepare_ovis_u1_infer_text(prompt, input_image, context, context_mask)
        if self.proprio_encoder is not None or proprio is not None:
            text_hidden, text_mask = self._append_proprio_to_context_if_enabled(
                context=text_hidden,
                context_mask=text_mask,
                proprio=proprio,
                source="Ovis-U1 video inference",
            )
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        ref_tokens, ref_img_ids = self._encode_ovis_u1_image_tokens(input_image, time_value=1.0)
        batch_size = int(ref_tokens.shape[0])
        latent_h = 2 * ((int(height) + 15) // 16)
        latent_w = 2 * ((int(width) + 15) // 16)
        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_video = torch.randn(
            (batch_size, (latent_h // 2) * (latent_w // 2), int(self.video_expert.transformer.in_channels)),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        target_img_ids = OvisU1VideoExpert.build_img_ids(
            batch_size=batch_size,
            token_height=latent_h // 2,
            token_width=latent_w // 2,
            time_value=0.0,
            device=self.device,
            dtype=self.torch_dtype,
        )
        infer_timesteps, infer_deltas = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        for step_t, step_delta in zip(infer_timesteps, infer_deltas):
            timestep_video = step_t.expand(batch_size).to(dtype=latents_video.dtype, device=self.device)
            video_pre = self.video_expert.pre_dit(
                x=latents_video,
                timestep=self._scheduler_timestep_to_unit(timestep_video, self.infer_video_scheduler),
                context=text_hidden,
                context_mask=text_mask,
                ref_image_hidden_states=ref_tokens,
                target_img_ids=target_img_ids,
                ref_img_ids=ref_img_ids,
            )
            pred_video = self.video_expert.native_forward_from_pre_state(video_pre)
            latents_video = self.infer_video_scheduler.step(pred_video, step_delta, latents_video)

        image = self._decode_ovis_u1_image_tokens(latents_video, height=height, width=width)
        return {"image": image[0].detach().cpu()}

    @torch.no_grad()
    def infer_video_omnigen2(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
    ) -> dict[str, Any]:
        self.eval()
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(f"`input_image` must be [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}")

        text_hidden, text_mask = self._prepare_omnigen2_infer_text(prompt, context, context_mask)
        if self.proprio_encoder is not None or proprio is not None:
            text_hidden, text_mask = self._append_proprio_to_context_if_enabled(
                context=text_hidden,
                context_mask=text_mask,
                proprio=proprio,
                source="OmniGen2 video inference",
            )
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        ref_latent = self._encode_omnigen2_image_latents(input_image)
        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_video = torch.randn(
            ref_latent.shape,
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        video_only_mot = MoT({"video": self.video_expert}, mot_checkpoint_mixed_attn=False).to(self.device)
        infer_timesteps, infer_deltas = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        for step_t, step_delta in zip(infer_timesteps, infer_deltas):
            timestep_video = step_t.expand(ref_latent.shape[0]).to(dtype=latents_video.dtype, device=self.device)
            video_pre = self.video_expert.pre_dit(
                x=latents_video,
                timestep=timestep_video,
                context=text_hidden,
                context_mask=text_mask,
                ref_image_hidden_states=ref_latent,
            )
            attention_mask = self._build_mot_attention_mask_omnigen2(
                encoder_seq_lengths=video_pre["encoder_seq_lengths"],
                seq_lengths=video_pre["seq_lengths"],
                max_video_seq_len=video_pre["tokens"].shape[1],
                action_seq_len=0,
                l_effective_ref_img_len=video_pre["l_effective_ref_img_len"],
                device=video_pre["tokens"].device,
            )
            tokens_out = video_only_mot(
                embeds_all={"video": video_pre["tokens"]},
                attention_mask=attention_mask,
                freqs_all={"video": video_pre["freqs"]},
                context_all={"video": None},
                t_mod_all={"video": video_pre["t_mod"]},
            )
            pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
            latents_video = self.infer_video_scheduler.step(pred_video, step_delta, latents_video)

        image = self._decode_omnigen2_image_latents(latents_video)
        return {"image": image[0].detach().cpu()}

    @torch.no_grad()
    def infer(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_frames: int,
        action: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 5.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        tau_star: Optional[float] = 0.0,
    ):
        if self.stack == "ovis_u1":
            if action_horizon is None:
                raise ValueError("`action_horizon` is required for Ovis-U1 action inference.")
            action_out = self.infer_action_ovis_u1(
                prompt=prompt,
                input_image=input_image,
                action_horizon=action_horizon,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tau_star=tau_star,
            )
            video_out = self.infer_video_ovis_u1(
                prompt=prompt,
                input_image=input_image,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
            )
            image = video_out["image"].detach().float().clamp(-1, 1)
            image = ((image + 1.0) * 127.5).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            return {
                "action": action_out["action"],
                "video": [Image.fromarray(image)],
            }
        if self.stack == "omnigen2":
            if action_horizon is None:
                raise ValueError("`action_horizon` is required for OmniGen2 action inference.")
            action_out = self.infer_action_omnigen2(
                prompt=prompt,
                input_image=input_image,
                action_horizon=action_horizon,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tau_star=tau_star,
            )
            video_out = self.infer_video_omnigen2(
                prompt=prompt,
                input_image=input_image,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
            )
            image = video_out["image"].detach().float().clamp(-1, 1)
            image = ((image + 1.0) * 127.5).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            return {
                "action": action_out["action"],
                "video": [Image.fromarray(image)],
            }
        if self.stack == "dim":
            if action_horizon is None:
                raise ValueError("`action_horizon` is required for DIM action inference.")
            return self.infer_dim_separate(
                prompt=prompt,
                input_image=input_image,
                action_horizon=action_horizon,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
            )
        if self.stack == "flux2":
            if action_horizon is None:
                raise ValueError("`action_horizon` is required for FLUX.2 action inference.")
            action_out = self.infer_action_flux2(
                prompt=prompt,
                input_image=input_image,
                action_horizon=action_horizon,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tau_star=tau_star,
            )
            video_out = self.infer_video_flux2(
                prompt=prompt,
                input_image=input_image,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
            )
            image = video_out["image"].detach().float().clamp(-1, 1)
            image = ((image + 1.0) * 127.5).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            return {"action": action_out["action"], "video": [Image.fromarray(image)]}

        return self.infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_frames,
            action_horizon=action_horizon,
            action=action,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
            tau_star=tau_star,
        )

    def save_checkpoint(self, path, optimizer=None, step=None):
        if bool(getattr(self, "save_lora_merged", False)):
            from .lora import lora_merged_state_dict

            mot_state = lora_merged_state_dict(self.mot)
            checkpoint_format = "lora_merged"
        elif bool(getattr(self, "save_trainable_only", False)):
            trainable_names = {name for name, param in self.mot.named_parameters() if param.requires_grad}
            mot_state = {
                key: value
                for key, value in self.mot.state_dict().items()
                if key in trainable_names
            }
            checkpoint_format = "trainable_only"
        else:
            mot_state = self.mot.state_dict()
            checkpoint_format = "full"
        payload = {
            "mot": mot_state,
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        if checkpoint_format != "full":
            payload["checkpoint_format"] = checkpoint_format
            payload["save_trainable_only"] = bool(getattr(self, "save_trainable_only", False))
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location="cpu")
        logger.info("Loading ImageWAM checkpoint from %s with payload keys=%s step=%s", path, sorted(payload.keys()), payload.get("step"))
        if "mot" in payload:
            mot_state = payload["mot"]
            if self.stack == "flux2":
                from .lora import merge_lora_state_dict_to_plain, remap_plain_linear_keys_to_lora_base

                mot_state = merge_lora_state_dict_to_plain(mot_state)
                mot_state = remap_plain_linear_keys_to_lora_base(self.mot, mot_state)
            load_result = self.mot.load_state_dict(mot_state, strict=False)
            missing_keys = list(load_result.missing_keys)
            unexpected_keys = list(load_result.unexpected_keys)
            logger.info(
                "Loaded MoT weights from checkpoint: missing_keys=%d unexpected_keys=%d",
                len(missing_keys),
                len(unexpected_keys),
            )
            if missing_keys:
                logger.warning("First missing MoT keys: %s", missing_keys[:20])
            if unexpected_keys:
                logger.warning("First unexpected MoT keys: %s", unexpected_keys[:20])
        elif "dit" in payload:
            logger.warning("Loading legacy `dit` checkpoint into video expert only.")
            load_result = self.video_expert.load_state_dict(payload["dit"], strict=False)
            logger.info(
                "Loaded legacy video expert weights: missing_keys=%d unexpected_keys=%d",
                len(load_result.missing_keys),
                len(load_result.unexpected_keys),
            )
        else:
            raise ValueError(f"Checkpoint missing both `mot` and `dit` keys: {path}")
        if self.proprio_encoder is not None:
            if "proprio_encoder" in payload:
                self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
                logger.info("Loaded proprio_encoder weights from checkpoint.")
            else:
                logger.warning("Checkpoint has no `proprio_encoder` weights; keeping current `proprio_encoder` params.")
        elif "proprio_encoder" in payload:
            logger.warning("Checkpoint contains `proprio_encoder` weights but current model has `proprio_dim=None`; ignoring.")

        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    def apply_trainable_policy(self) -> None:
        """Refine trainer's default DiT-only policy for parameter-efficient modes."""
        if self.stack != "flux2":
            return
        video_expert = self.mot.mixtures["video"] if "video" in self.mot.mixtures else None
        action_expert = self.mot.mixtures["action"] if "action" in self.mot.mixtures else None
        if action_expert is not None:
            action_expert.train()
            action_expert.requires_grad_(True)
        if video_expert is None or not bool(getattr(video_expert, "flux2_lora_enabled", False)):
            return
        video_expert.train()
        for param in video_expert.parameters():
            param.requires_grad = False
        for name, param in video_expert.named_parameters():
            if ".lora_A" in name or ".lora_B" in name:
                param.requires_grad = True

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)
