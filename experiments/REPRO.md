# Phase 0 Reproduction Notes

## P0-T2 - tau* provenance

Verified on 2026-07-07 in this fork.

The released FLUX.2-4B LIBERO path is:

`experiments/libero/eval_libero_single.py`
-> `_predict_action_chunk(...)`
-> `model.infer_action(**infer_kwargs)`
-> `src/imagewam/models/backbones/imagewam.py:ImageWAM.infer_action(...)`
-> `ImageWAM.infer_action_flux2(...)`.

The RobotWin policy path mirrors this through:

`experiments/robotwin/imagewam_policy/deploy_policy.py`
-> `WorldActionRobotWinPolicy._infer_action_chunk(...)`
-> `self.model.infer_action(**infer_kwargs)`.

Before P0-T3, the FLUX.2 editing/video cache prefill used:

`video_timestep = torch.zeros((batch_size,), dtype=ref_tokens.dtype, device=self.device)`

inside `src/imagewam/models/backbones/imagewam.py:infer_action_flux2(...)`, then passed that timestep to `self.video_expert.pre_dit(...)` and materialized the cache via `self.mot.prefill_flux2_video_cache(...)`.

Value: `0.0` scheduler-timestep units. With the current `num_train_timesteps: 1000`, this is normalized `tau_star = 0.0`.

Config status before P0-T3: no explicit tau* appeared in `configs/sim_libero.yaml`, `configs/task/libero_flux2_klein_4b_base_imagewam.yaml`, or `configs/model/imagewam_flux2_klein_4b_base.yaml`. `EVALUATION.num_inference_steps` controlled the action denoising schedule, not the fixed editing/video cache timestep.

I did not find comments explaining how `tau_star = 0.0` was chosen.
