"""Hydra training entry for RoboDojo and 28D real-robot data."""
from __future__ import annotations
import math
import hydra
from omegaconf import OmegaConf


def register_resolvers():
    OmegaConf.register_new_resolver('mul', lambda a, b: int(a) * int(b), replace=True)


def validate_training_options(cfg):
    if cfg.optimizer.dit_mode != 'full_train':
        raise ValueError('These recipes use full_train; LoRA needs a separately validated recipe')
    if str(cfg.task.name) not in {'robodojo', 'real_robot'}:
        raise ValueError('task.name must match a supported training profile')
    for key in ['start_ckpt_path', 'start_normalizer_path']:
        if getattr(cfg.optimizer, key, None):
            raise ValueError('Use training.resume and training.resume_path instead of ignored optimizer start paths')
    if cfg.optimizer.obs_encoder_mode != 'freeze':
        raise ValueError('The Wan VAE is frozen; obs_encoder_mode must be freeze')
    if int(cfg.policy.puga_n_samples) != 1:
        raise ValueError('This response-credit implementation supports one perturbation')
    for key in ['video_loss_weight', 'action_loss_weight', 'align_loss_weight', 'ccrt_lambda_max',
                'puga_weight', 'puga_act_weight', 'puga_w_unc', 'puga_w_lev', 'puga_w_diff', 'puga_act_delta']:
        value = float(cfg.policy[key])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'policy.{key} must be finite and nonnegative')
    monitor = str(cfg.checkpoint.topk.monitor_key)
    if monitor not in {'train_loss', 'val_loss', 'sample_visual_mse', 'sample_action_mse_error'}:
        raise ValueError('Unsupported checkpoint monitor key')
    if monitor == 'val_loss' and (int(cfg.training.val_every) <= 0 or (cfg.task.name == 'real_robot' and float(cfg.task.dataset.val_split_ratio) <= 0)):
        raise ValueError('val_loss checkpoint selection requires validation')
    if monitor.startswith('sample_') and not any(int(getattr(cfg.training, key, 0)) > 0 for key in ['sample_every', 'sample_every_steps']):
        raise ValueError('Sample metric checkpoint selection requires enabled sampling')
    for key in ['num_epochs', 'gradient_accumulate_every']:
        if int(cfg.training[key]) < 1:
            raise ValueError(f'training.{key} must be positive')
    for key in ['max_train_steps', 'max_val_steps']:
        value = getattr(cfg.training, key, None)
        if value is not None and int(value) < 1:
            raise ValueError(f'training.{key} must be positive or null')
    for name in ['train_dataloader', 'val_dataloader']:
        loader = cfg[name]
        if int(loader.batch_size) < 1 or int(loader.num_workers) < 0:
            raise ValueError(f'{name}: batch_size must be positive and num_workers nonnegative')
        if int(loader.num_workers) == 0 and bool(getattr(loader, 'persistent_workers', False)):
            raise ValueError('persistent_workers requires at least one data worker')
    for key in ['ccrt_p_max', 'opsr_p_max', 'puga_p']:
        if not 0 <= float(cfg.policy[key]) <= 1:
            raise ValueError(f'policy.{key} must be in [0, 1]')
    if not bool(cfg.policy.strict_auxiliary_errors):
        import warnings
        warnings.warn('Auxiliary errors will use the explicit tolerant fallback', RuntimeWarning)


register_resolvers()


@hydra.main(version_base=None, config_path='configs', config_name='robodojo')
def main(cfg):
    OmegaConf.resolve(cfg)
    validate_training_options(cfg)
    if cfg.policy.obs_encoder._target_ == 'rap_wam.core.model.wan.official_obs_encoder.WanOfficialObsEncoder':
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('Wan observation encoding requires CUDA; use preflight/config inspection for CPU checks')
    from rap_wam.preflight import validate_config
    validate_config(cfg, check_weights=True)
    workspace = hydra.utils.get_class(cfg._target_)(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
