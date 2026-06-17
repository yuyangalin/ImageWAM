import time
from typing import Any, Optional

import torch
import torch.nn.functional as F

from imagewam.utils.logging_config import get_logger

from .imagewam_idm import ImageWAMIDM

logger = get_logger(__name__)


class ImageWAMCacheIDM(ImageWAMIDM):
    """IDM-style variant where actions condition on first frame + cached future video latents."""

    def __init__(
        self,
        *args,
        cache_idm_train_schedule: str = "sequential_micro",
        cache_idm_alternate_loss_scale: float = 1.0,
        cache_idm_ddp_zero_anchor: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._configure_cache_idm_training(
            cache_idm_train_schedule=cache_idm_train_schedule,
            cache_idm_alternate_loss_scale=cache_idm_alternate_loss_scale,
            cache_idm_ddp_zero_anchor=cache_idm_ddp_zero_anchor,
        )

    @classmethod
    def from_wan22_pretrained(
        cls,
        cache_idm_train_schedule: str = "sequential_micro",
        cache_idm_alternate_loss_scale: float = 1.0,
        cache_idm_ddp_zero_anchor: bool = False,
        **kwargs,
    ):
        model = super().from_wan22_pretrained(**kwargs)
        model._configure_cache_idm_training(
            cache_idm_train_schedule=cache_idm_train_schedule,
            cache_idm_alternate_loss_scale=cache_idm_alternate_loss_scale,
            cache_idm_ddp_zero_anchor=cache_idm_ddp_zero_anchor,
        )
        return model

    @classmethod
    def from_omnigen2_pretrained(
        cls,
        cache_idm_train_schedule: str = "sequential_micro",
        cache_idm_alternate_loss_scale: float = 1.0,
        cache_idm_ddp_zero_anchor: bool = False,
        **kwargs,
    ):
        model = super().from_omnigen2_pretrained(**kwargs)
        model._configure_cache_idm_training(
            cache_idm_train_schedule=cache_idm_train_schedule,
            cache_idm_alternate_loss_scale=cache_idm_alternate_loss_scale,
            cache_idm_ddp_zero_anchor=cache_idm_ddp_zero_anchor,
        )
        return model

    def _configure_cache_idm_training(
        self,
        cache_idm_train_schedule: str,
        cache_idm_alternate_loss_scale: float,
        cache_idm_ddp_zero_anchor: bool,
    ) -> None:
        schedule = str(cache_idm_train_schedule).strip().lower()
        if schedule not in {"joint", "alternate_micro", "sequential_micro"}:
            raise ValueError(
                "`cache_idm_train_schedule` must be 'joint', 'alternate_micro', or 'sequential_micro', "
                f"got {cache_idm_train_schedule!r}."
            )
        self.cache_idm_train_schedule = schedule
        self.cache_idm_alternate_loss_scale = float(cache_idm_alternate_loss_scale)
        self.cache_idm_ddp_zero_anchor = bool(cache_idm_ddp_zero_anchor)
        self._cache_idm_micro_step = int(getattr(self, "_cache_idm_micro_step", 0))
        self._cache_idm_log_sums = getattr(self, "_cache_idm_log_sums", {"video": 0.0, "action": 0.0})
        self._cache_idm_log_counts = getattr(self, "_cache_idm_log_counts", {"video": 0, "action": 0})

    def _next_cache_idm_train_phase(self) -> str:
        if self.cache_idm_train_schedule in {"joint", "sequential_micro"}:
            return "joint"
        micro_step = int(getattr(self, "_cache_idm_micro_step", 0))
        self._cache_idm_micro_step = micro_step + 1
        return "video" if micro_step % 2 == 0 else "action"

    def _cache_idm_loss_scale_for_phase(self, train_phase: str) -> float:
        if train_phase == "joint" or self.cache_idm_train_schedule == "sequential_micro":
            return 1.0
        return self.cache_idm_alternate_loss_scale

    def _ddp_zero_anchor(self, reference: torch.Tensor) -> torch.Tensor:
        if not self.cache_idm_ddp_zero_anchor:
            return reference.new_zeros(())
        anchor = reference.new_zeros(())
        for param in self.parameters():
            if param.requires_grad and param.numel() > 0:
                anchor = anchor + param.reshape(-1)[0].to(dtype=reference.dtype) * 0.0
        return anchor

    def _update_cache_idm_loss_log_average(
        self,
        name: str,
        value: Optional[torch.Tensor],
        weight: float,
        active: bool,
    ) -> float:
        if active:
            logged_value = float(weight * value.detach().item())
            self._cache_idm_log_sums[name] = float(self._cache_idm_log_sums.get(name, 0.0)) + logged_value
            self._cache_idm_log_counts[name] = int(self._cache_idm_log_counts.get(name, 0)) + 1
        count = int(self._cache_idm_log_counts.get(name, 0))
        if count <= 0:
            return 0.0
        return float(self._cache_idm_log_sums[name]) / float(count)

    def _full_video_timestep(
        self,
        batch_size: int,
        dtype: torch.dtype,
        scheduler,
    ) -> torch.Tensor:
        return torch.full(
            (batch_size,),
            float(scheduler.num_train_timesteps),
            dtype=dtype,
            device=self.device,
        )

    def _half_video_timestep(
        self,
        batch_size: int,
        dtype: torch.dtype,
        scheduler,
    ) -> torch.Tensor:
        return torch.full(
            (batch_size,),
            0.5 * float(scheduler.num_train_timesteps),
            dtype=dtype,
            device=self.device,
        )

    def _sample_future_condition_latents(
        self,
        target_latent: torch.Tensor,
        scheduler,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        batch_size = int(target_latent.shape[0])
        clean_mask = torch.rand((batch_size,), device=target_latent.device) < 0.5
        noisy_sigma = 0.5 + 0.5 * torch.rand(
            (batch_size,),
            device=target_latent.device,
            dtype=torch.float32,
        )
        sigma = torch.where(clean_mask, torch.zeros_like(noisy_sigma), noisy_sigma)
        timestep = (sigma * float(scheduler.num_train_timesteps)).to(dtype=target_latent.dtype)
        noise = torch.randn_like(target_latent)
        cond_latent = scheduler.add_noise(target_latent, noise, timestep)
        stats = {
            "cache_idm_future_cond_sigma_mean": float(sigma.detach().mean().item()),
            "cache_idm_future_cond_clean_ratio": float(clean_mask.detach().float().mean().item()),
        }
        return cond_latent, timestep, stats

    @torch.no_grad()
    def _build_noise_condition_attention_mask(
        self,
        noisy_video_seq_len: int,
        cond_video_seq_len: int,
        action_seq_len: int,
        noisy_video_tokens_per_frame: int,
        cond_video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        if noisy_video_tokens_per_frame != cond_video_tokens_per_frame:
            raise ValueError(
                "CacheIDM requires identical `tokens_per_frame` for noisy and cond video branches, "
                f"got {noisy_video_tokens_per_frame} and {cond_video_tokens_per_frame}."
            )

        noisy_end = noisy_video_seq_len
        cond_end = noisy_video_seq_len + cond_video_seq_len
        total_seq_len = cond_end + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)

        mask[:noisy_end, :noisy_end] = self.video_expert.build_video_to_video_mask(
            video_seq_len=noisy_video_seq_len,
            video_tokens_per_frame=noisy_video_tokens_per_frame,
            device=device,
        )
        mask[noisy_end:cond_end, noisy_end:cond_end] = self.video_expert.build_video_to_video_mask(
            video_seq_len=cond_video_seq_len,
            video_tokens_per_frame=cond_video_tokens_per_frame,
            device=device,
        )
        mask[cond_end:, cond_end:] = True
        mask[cond_end:, noisy_end:cond_end] = True
        return mask

    @torch.no_grad()
    def _build_omnigen2_action_condition_attention_mask(
        self,
        encoder_seq_lengths: list[int],
        seq_lengths: list[int],
        max_video_seq_len: int,
        action_seq_len: int,
        l_effective_ref_img_len: list[list[int]],
        device: torch.device,
    ) -> torch.Tensor:
        batch_size = len(seq_lengths)
        total_seq_len = int(max_video_seq_len) + int(action_seq_len)
        action_start = int(max_video_seq_len)
        mask = torch.zeros(batch_size, total_seq_len, total_seq_len, dtype=torch.bool, device=device)

        for i, (cap_len, video_len) in enumerate(zip(encoder_seq_lengths, seq_lengths)):
            prefix_len = int(cap_len + sum(l_effective_ref_img_len[i]))
            video_len = int(video_len)
            mask[i, :prefix_len, :prefix_len] = True
            mask[i, prefix_len:video_len, :video_len] = True
            mask[i, action_start:, action_start:] = True
            mask[i, action_start:, :video_len] = True
        return mask

    @torch.no_grad()
    def _build_omnigen2_noise_condition_attention_mask(
        self,
        noisy_encoder_seq_lengths: list[int],
        noisy_seq_lengths: list[int],
        noisy_max_video_seq_len: int,
        cond_encoder_seq_lengths: list[int],
        cond_seq_lengths: list[int],
        cond_max_video_seq_len: int,
        action_seq_len: int,
        noisy_l_effective_ref_img_len: list[list[int]],
        cond_l_effective_ref_img_len: list[list[int]],
        device: torch.device,
    ) -> torch.Tensor:
        batch_size = len(noisy_seq_lengths)
        noisy_end = int(noisy_max_video_seq_len)
        cond_end = noisy_end + int(cond_max_video_seq_len)
        total_seq_len = cond_end + int(action_seq_len)
        mask = torch.zeros(batch_size, total_seq_len, total_seq_len, dtype=torch.bool, device=device)

        for i in range(batch_size):
            noisy_prefix_len = int(noisy_encoder_seq_lengths[i] + sum(noisy_l_effective_ref_img_len[i]))
            noisy_len = int(noisy_seq_lengths[i])
            cond_prefix_len = int(cond_encoder_seq_lengths[i] + sum(cond_l_effective_ref_img_len[i]))
            cond_len = int(cond_seq_lengths[i])

            mask[i, :noisy_prefix_len, :noisy_prefix_len] = True
            mask[i, noisy_prefix_len:noisy_len, :noisy_len] = True

            cond_start = noisy_end
            cond_prefix_end = cond_start + cond_prefix_len
            cond_seq_end = cond_start + cond_len
            mask[i, cond_start:cond_prefix_end, cond_start:cond_prefix_end] = True
            mask[i, cond_prefix_end:cond_seq_end, cond_start:cond_seq_end] = True

            mask[i, cond_end:, cond_end:] = True
            mask[i, cond_end:, cond_start:cond_seq_end] = True
        return mask

    def _forward_omnigen2_video_only(self, video_pre: dict[str, Any]) -> torch.Tensor:
        attention_mask = self._build_mot_attention_mask_omnigen2(
            encoder_seq_lengths=video_pre["encoder_seq_lengths"],
            seq_lengths=video_pre["seq_lengths"],
            max_video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=0,
            l_effective_ref_img_len=video_pre["l_effective_ref_img_len"],
            device=video_pre["tokens"].device,
        )

        expert = self.video_expert
        x = video_pre["tokens"]
        for layer_idx in range(self.mot.num_layers):
            block = expert.blocks[layer_idx]
            built = self.mot._build_expert_attention_io(
                expert=expert,
                block=block,
                x=x,
                freqs=video_pre["freqs"],
                t_mod=video_pre["t_mod"],
            )
            mixed = self.mot._mixed_attention(
                q_cat=built["q"],
                k_cat=built["k"],
                v_cat=built["v"],
                attention_mask=attention_mask,
            )
            x = self.mot._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=built["residual_x"],
                post_state=built["post_state"],
                use_gradient_checkpointing=built["use_gradient_checkpointing"],
                mixed_slice=mixed,
                context_payload=None,
            )
        return self.video_expert.post_dit(x, video_pre)

    def training_loss(self, sample, tiled: bool = False):
        if self.stack == "omnigen2":
            return self._training_loss_omnigen2_cache_idm(sample, tiled=tiled)

        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]
        fuse_flag = inputs["fuse_vae_embedding_in_latents"]

        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        latents_noisy = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)
        if inputs["first_frame_latents"] is not None:
            latents_noisy[:, :, 0:1] = inputs["first_frame_latents"]

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        latents_cond = torch.randn_like(input_latents)
        timestep_video_cond = self._full_video_timestep(
            batch_size=batch_size,
            dtype=input_latents.dtype,
            scheduler=self.train_video_scheduler,
        )
        if inputs["first_frame_latents"] is not None:
            latents_cond[:, :, 0:1] = inputs["first_frame_latents"]

        video_pre_noisy = self.video_expert.pre_dit(
            x=latents_noisy,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        video_pre_cond = self.video_expert.pre_dit(
            x=latents_cond,
            timestep=timestep_video_cond,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        if video_pre_noisy["t_mod"].ndim != 4 or video_pre_cond["t_mod"].ndim != 4:
            raise ValueError(
                "CacheIDM requires token-wise `t_mod`; "
                "ensure `seperated_timestep=true` and `fuse_vae_embedding_in_latents=true`."
            )

        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        noisy_video_seq_len = int(video_pre_noisy["tokens"].shape[1])
        cond_video_seq_len = int(video_pre_cond["tokens"].shape[1])
        merged_video_tokens = torch.cat([video_pre_noisy["tokens"], video_pre_cond["tokens"]], dim=1)
        merged_video_freqs = torch.cat([video_pre_noisy["freqs"], video_pre_cond["freqs"]], dim=0)
        merged_video_t_mod = torch.cat([video_pre_noisy["t_mod"], video_pre_cond["t_mod"]], dim=1)
        merged_video_context_mask = torch.cat([video_pre_noisy["context_mask"], video_pre_cond["context_mask"]], dim=1)

        attention_mask = self._build_noise_condition_attention_mask(
            noisy_video_seq_len=noisy_video_seq_len,
            cond_video_seq_len=cond_video_seq_len,
            action_seq_len=action_pre["tokens"].shape[1],
            noisy_video_tokens_per_frame=int(video_pre_noisy["meta"]["tokens_per_frame"]),
            cond_video_tokens_per_frame=int(video_pre_cond["meta"]["tokens_per_frame"]),
            device=merged_video_tokens.device,
        )

        tokens_out = self.mot(
            embeds_all={
                "video": merged_video_tokens,
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": merged_video_freqs,
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre_noisy["context"],
                    "mask": merged_video_context_mask,
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": merged_video_t_mod,
                "action": action_pre["t_mod"],
            },
        )

        pred_video_tokens = tokens_out["video"][:, :noisy_video_seq_len]
        pred_video = self.video_expert.post_dit(pred_video_tokens, video_pre_noisy)
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

    def _training_loss_omnigen2_cache_idm(self, sample, tiled: bool = False):
        inputs = self.build_inputs_omnigen2(sample, tiled=tiled)
        target_latent = inputs["target_latent"]
        action = inputs["action"]
        batch_size = target_latent.shape[0]
        train_phase = self._next_cache_idm_train_phase()

        loss_video = target_latent.new_zeros(())
        loss_action = target_latent.new_zeros(())

        if train_phase in {"joint", "video"}:
            noise_video = torch.randn_like(target_latent)
            timestep_video = self.train_video_scheduler.sample_training_t(
                batch_size=batch_size,
                device=self.device,
                dtype=target_latent.dtype,
            )
            noisy_latent = self.train_video_scheduler.add_noise(target_latent, noise_video, timestep_video)
            target_video = self.train_video_scheduler.training_target(target_latent, noise_video, timestep_video)

            video_pre_noisy = self.video_expert.pre_dit(
                x=noisy_latent,
                timestep=timestep_video,
                context=inputs["text_hidden_states"],
                context_mask=inputs["text_attention_mask"],
                ref_image_hidden_states=inputs["ref_image_latents"],
            )
            pred_video = self._forward_omnigen2_video_only(video_pre_noisy)

            video_loss_per_sample = F.mse_loss(pred_video.float(), target_video.float(), reduction="none").mean(
                dim=(1, 2, 3)
            )
            video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
                video_loss_per_sample.device, dtype=video_loss_per_sample.dtype
            )
            loss_video = (video_loss_per_sample * video_weight).mean()

        if train_phase in {"joint", "action"}:
            noise_action = torch.randn_like(action)
            timestep_action = self.train_action_scheduler.sample_training_t(
                batch_size=batch_size,
                device=self.device,
                dtype=action.dtype,
            )
            noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
            target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

            latents_cond = torch.randn_like(target_latent)
            timestep_video_cond = self._full_video_timestep(
                batch_size=batch_size,
                dtype=target_latent.dtype,
                scheduler=self.train_video_scheduler,
            )
            video_pre_cond = self.video_expert.pre_dit(
                x=latents_cond,
                timestep=timestep_video_cond,
                context=inputs["text_hidden_states"],
                context_mask=inputs["text_attention_mask"],
                ref_image_hidden_states=inputs["ref_image_latents"],
            )
            action_pre = self.action_expert.pre_dit(
                action_tokens=noisy_action,
                timestep=timestep_action,
            )

            attention_mask = self._build_omnigen2_action_condition_attention_mask(
                encoder_seq_lengths=video_pre_cond["encoder_seq_lengths"],
                seq_lengths=video_pre_cond["seq_lengths"],
                max_video_seq_len=video_pre_cond["tokens"].shape[1],
                action_seq_len=action_pre["tokens"].shape[1],
                l_effective_ref_img_len=video_pre_cond["l_effective_ref_img_len"],
                device=video_pre_cond["tokens"].device,
            )
            tokens_out = self.mot(
                embeds_all={"video": video_pre_cond["tokens"], "action": action_pre["tokens"]},
                attention_mask=attention_mask,
                freqs_all={"video": video_pre_cond["freqs"], "action": action_pre["freqs"]},
                context_all={"video": None, "action": None},
                t_mod_all={"video": video_pre_cond["t_mod"], "action": action_pre["t_mod"]},
            )
            pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

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

        if train_phase == "joint":
            scale = 1.0
            loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        elif train_phase == "video":
            scale = self.cache_idm_alternate_loss_scale
            loss_total = scale * self.loss_lambda_video * loss_video + self._ddp_zero_anchor(loss_video)
        else:
            scale = self.cache_idm_alternate_loss_scale
            loss_total = scale * self.loss_lambda_action * loss_action + self._ddp_zero_anchor(loss_action)
        logged_loss_video = self._update_cache_idm_loss_log_average(
            name="video",
            value=loss_video,
            weight=self.loss_lambda_video,
            active=train_phase in {"joint", "video"},
        )
        logged_loss_action = self._update_cache_idm_loss_log_average(
            name="action",
            value=loss_action,
            weight=self.loss_lambda_action,
            active=train_phase in {"joint", "action"},
        )
        return loss_total, {
            "loss_video_running_avg": logged_loss_video,
            "loss_action_running_avg": logged_loss_action,
            "loss_video_current": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action_current": self.loss_lambda_action * float(loss_action.detach().item()),
            "cache_idm_train_phase": {"joint": 0.0, "video": 1.0, "action": 2.0}[train_phase],
            "cache_idm_loss_scale": scale,
        }

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        num_video_frames: int,
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
    ) -> dict[str, Any]:
        del negative_prompt, text_cfg_scale
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
            )
        return self._infer_action_wan22_noise_cond(
            prompt=prompt,
            input_image=input_image,
            action_horizon=action_horizon,
            num_video_frames=num_video_frames,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
        )

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None,
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
    ) -> dict[str, Any]:
        del action, negative_prompt, text_cfg_scale, test_action_with_infer_action
        if self.stack == "omnigen2":
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
            return {"video": video_out["image"], "action": action_out["action"]}

        action_out = self._infer_action_wan22_noise_cond(
            prompt=prompt,
            input_image=input_image,
            action_horizon=action_horizon,
            num_video_frames=num_video_frames,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
        )
        video_out = super().infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_video_frames,
            action_horizon=action_horizon,
            action=None,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
            test_action_with_infer_action=False,
        )
        return {"video": video_out["video"], "action": action_out["action"]}

    @torch.no_grad()
    def _infer_action_wan22_noise_cond(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        num_video_frames: int,
        proprio: Optional[torch.Tensor],
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
        num_inference_steps: int,
        sigma_shift: Optional[float],
        seed: Optional[int],
        rand_device: str,
        tiled: bool,
    ) -> dict[str, Any]:
        self.eval()
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
            raise ValueError(f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}")

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

        timestep_video_cond = self._full_video_timestep(
            batch_size=latents_video.shape[0],
            dtype=latents_video.dtype,
            scheduler=self.infer_video_scheduler,
        )
        video_pre_cond = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video_cond,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        video_seq_len = int(video_pre_cond["tokens"].shape[1])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=int(video_pre_cond["meta"]["tokens_per_frame"]),
            device=video_pre_cond["tokens"].device,
        )
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre_cond["tokens"],
            video_freqs=video_pre_cond["freqs"],
            video_t_mod=video_pre_cond["t_mod"],
            video_context_payload={
                "context": video_pre_cond["context"],
                "mask": video_pre_cond["context_mask"],
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
            pred_action = self._predict_action_noise_with_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)

        return {"action": latents_action[0].detach().to(device="cpu", dtype=torch.float32)}

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
                source="OmniGen2 CacheIDM action inference",
            )
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        ref_latent = self._encode_omnigen2_image_latents(input_image)
        _mark_profile("encode_image_latents_s")
        batch_size = int(ref_latent.shape[0])

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_video = torch.randn(
            ref_latent.shape,
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
        _mark_profile("sample_latents_s")

        timestep_video_start = self._full_video_timestep(
            batch_size=batch_size,
            dtype=latents_video.dtype,
            scheduler=self.infer_video_scheduler,
        )
        _mark_profile("prepare_video_start_timestep_s")
        video_pre_start = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video_start,
            context=text_hidden,
            context_mask=text_mask,
            ref_image_hidden_states=ref_latent,
        )
        _mark_profile("video_pre_dit_start_s")
        pred_video = self._forward_omnigen2_video_only(video_pre_start)
        latents_video = self.infer_video_scheduler.step(
            pred_video,
            latents_video.new_tensor(-0.5),
            latents_video,
        )
        _mark_profile("video_single_step_to_half_s")

        timestep_video_cond = self._half_video_timestep(
            batch_size=batch_size,
            dtype=latents_video.dtype,
            scheduler=self.infer_video_scheduler,
        )
        _mark_profile("prepare_video_half_timestep_s")
        video_pre_cond = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video_cond,
            context=text_hidden,
            context_mask=text_mask,
            ref_image_hidden_states=ref_latent,
        )
        _mark_profile("video_pre_dit_half_s")
        video_seq_len = int(video_pre_cond["tokens"].shape[1])
        text_len = int(video_pre_cond["encoder_seq_lengths"][0])
        ref_len = int(sum(video_pre_cond["l_effective_ref_img_len"][0]))
        self._configure_action_attention_capture(
            condition_slice=(text_len, text_len + ref_len),
            condition_grid=(int(input_image.shape[-2]) // 16, int(input_image.shape[-1]) // 16),
            prefix_len=video_seq_len,
            action_len=int(latents_action.shape[1]),
            metadata={
                "stack": str(self.stack),
                "source": "omnigen2_cache_idm_infer_action",
                "condition": "ref_image_latents_plus_half_denoised_future",
                "text_len": text_len,
                "ref_len": ref_len,
                "input_size": [int(input_image.shape[-2]), int(input_image.shape[-1])],
            },
        )
        attention_mask = self._build_omnigen2_action_condition_attention_mask(
            encoder_seq_lengths=video_pre_cond["encoder_seq_lengths"],
            seq_lengths=video_pre_cond["seq_lengths"],
            max_video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            l_effective_ref_img_len=video_pre_cond["l_effective_ref_img_len"],
            device=video_pre_cond["tokens"].device,
        )
        _mark_profile("build_attention_mask_s")
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre_cond["tokens"],
            video_freqs=video_pre_cond["freqs"],
            video_t_mod=video_pre_cond["t_mod"],
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
