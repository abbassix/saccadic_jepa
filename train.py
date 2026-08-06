# train.py
import os
from pathlib import Path
from time import time
from collections import defaultdict

import fire

import torch
import torch.nn as nn

import wandb
from omegaconf import OmegaConf

from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from tqdm import tqdm

from eb_jepa.architectures import (
    MobileNetV2Encoder,
    MobileNetV4Encoder,
    InverseDynamicsModel,
    Projector,
    GatedPredictor,
)
from eb_jepa.datasets.utils import init_data
from eb_jepa.jepa import JEPA
from eb_jepa.logging import get_logger
from eb_jepa.losses import VC_IDM_Regularizer, SIG_IDM_Regularizer
from eb_jepa.schedulers import CosineWithWarmup
from eb_jepa.training_utils import (
    get_default_dev_name,
    get_exp_name,
    get_unified_experiment_dir,
    load_checkpoint,
    load_config,
    log_config,
    log_data_info,
    log_epoch,
    log_model_info,
    save_checkpoint,
    setup_device,
    setup_seed,
    setup_wandb,
)
from eval import run_patch_localization_eval

logger = get_logger(__name__)


def run(
    fname: str = "cfgs/train.yaml",
    cfg=None,
    folder=None,
    **overrides,
):
    """
    Train an action-conditioned Video JEPA model.

    Args:
        fname: Path to the YAML config file.
        cfg: Pre-loaded config object (optional, overrides config file).
        folder: Experiment folder path (optional, auto-generated if not provided).
        **overrides: Config overrides in dot notation (e.g., model.henc=64).
    """
    if cfg is None:
        cfg = load_config(fname, overrides if overrides else None)

    # Create experiment directory using unified structure (if not provided)
    if folder is None:
        if cfg.meta.get("model_folder"):
            folder = Path(cfg.meta.model_folder)
            folder_name = folder.name
            exp_name = folder_name.rsplit("_seed", 1)[0]
        else:
            sweep_name = get_default_dev_name()
            exp_name = get_exp_name("ac_video_jepa", cfg)
            folder = get_unified_experiment_dir(
                example_name="ac_video_jepa",
                sweep_name=sweep_name,
                exp_name=exp_name,
                seed=cfg.meta.seed,
            )
    else:
        folder = Path(folder)
        folder_name = folder.name
        exp_name = folder_name.rsplit("_seed", 1)[0]

    os.makedirs(folder, exist_ok=True)

    loader, val_loader, cfg = init_data(
        env_name=cfg.data.env_name, config=cfg
    )

    # -- SETUP
    device = setup_device(cfg.meta.get("device", "auto"))
    setup_seed(cfg.meta.seed)

    # -- WANDB
    wandb_run = setup_wandb(
        project="eb_jepa",
        config={
            "example": "ac_video_jepa",
            **OmegaConf.to_container(cfg, resolve=True),
        },
        run_dir=folder,
        run_name=exp_name,
        tags=[f"seed_{cfg.meta.seed}", "ac_video_jepa"],
        group=cfg.logging.get("wandb_group"),
        enabled=cfg.logging.get("log_wandb", False),
        sweep_id=cfg.logging.get("wandb_sweep_id"),
    )

    log_data_info(
        cfg.data.env_name,
        len(loader),
        cfg.data.batch_size,
    )

    # Mixed precision setup
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map.get(cfg.training.get("dtype", "float16").lower(), torch.float16)
    use_amp = cfg.training.get("use_amp", True)
    scaler = GradScaler(device.type, enabled=use_amp)
    logger.info(f"Using AMP with {dtype=}" if use_amp else f"AMP disabled")

    # -- EVAL SETUP
    enable_eval = cfg.meta.get("enable_eval", False)

    # -- SAVE CONFIG
    latest_ckpt_path = folder / "latest.pth.tar"
    total_steps = cfg.optim.epochs * len(loader)
    config_path = folder / "config.yaml"
    with open(config_path, "w") as f:
        OmegaConf.save(cfg, config_path)
    print(f"Saved complete config to {config_path}")

    # -- MODEL --

    # -- ENCODER --
    if cfg.model.encoder_architecture == "mobilenetv2":
        encoder = MobileNetV2Encoder(
            width_mult=1.0,
        )
    elif cfg.model.encoder_architecture == "mobilenetv4":
        encoder = MobileNetV4Encoder(
            width_mult=1.0,
        )
    else:
        raise ValueError(f"Unsupported encoder type: {cfg.model.encoder_architecture}")
    
    logger.info(f"Number of features: {encoder.feature_dim}")

    action_dim = 2

    # -- PROJECTOR --
    if cfg.model.regularizer.use_proj:
        projector = Projector(f"{encoder.feature_dim}-{encoder.feature_dim*4}-{encoder.feature_dim*4}")
    else:
        projector = nn.Identity()

    # -- PREDICTOR --
    # predictor = StatePredictor(hidden_size=encoder.projector_output_dim, action_dim=action_dim)
    predictor = GatedPredictor(state_size=encoder.feature_dim, action_size=action_dim)
    
    # -- IDM --
    idm = InverseDynamicsModel(
        state_dim=projector.out_dim if cfg.model.regularizer.idm_after_proj else encoder.feature_dim,
        hidden_dim=cfg.model.idm.hidden_dim,
        action_dim=action_dim,
    ).to(device)
    
    # -- REGULARIZER --
    if cfg.model.regularizer.type == "vicreg":
        regularizer = VC_IDM_Regularizer(
            idm=idm,
            projector=projector,
            idm_after_proj=cfg.model.regularizer.idm_after_proj,
        )
    elif cfg.model.regularizer.type == "sigreg":
        regularizer = SIG_IDM_Regularizer(
            idm=idm,
            projector=projector,
            idm_after_proj=cfg.model.regularizer.idm_after_proj,
        )
    else:
        raise ValueError(f"Unsupported regularizer type: {cfg.model.regularizer.type}")
    
    jepa = JEPA(encoder, predictor, regularizer, predcost=nn.GaussianNLLLoss()).to(device)

    # Log model structure and parameters
    encoder_params = sum(p.numel() for p in encoder.parameters())
    projector_params = sum(p.numel() for p in projector.parameters())
    predictor_params = sum(p.numel() for p in predictor.parameters())
    idm_params = sum(p.numel() for p in idm.parameters())
    log_model_info(jepa, {"encoder": encoder_params, "projector": projector_params, "predictor": predictor_params, "idm": idm_params})

    log_config(cfg)

    def split_decay_params(module):
        decay, no_decay = [], []
        for p in module.parameters():
            if not p.requires_grad:
                continue
            (no_decay if p.ndim <= 1 else decay).append(p)
        return decay, no_decay

    wd = cfg.optim.get("weight_decay", 1e-6)

    enc_decay, enc_no_decay = split_decay_params(encoder)
    proj_decay, proj_no_decay = split_decay_params(projector)
    pred_decay, pred_no_decay = split_decay_params(predictor)
    idm_decay, idm_no_decay = split_decay_params(idm)

    jepa_optimizer = AdamW(
        [
            {"params": enc_decay, "lr": cfg.optim.enc_lr, "weight_decay": wd},
            {"params": enc_no_decay, "lr": cfg.optim.enc_lr, "weight_decay": 0.0},
            {"params": proj_decay, "lr": cfg.optim.proj_lr, "weight_decay": wd},
            {"params": proj_no_decay, "lr": cfg.optim.proj_lr, "weight_decay": 0.0},
            {"params": pred_decay, "lr": cfg.optim.pred_lr, "weight_decay": wd},
            {"params": pred_no_decay, "lr": cfg.optim.pred_lr, "weight_decay": 0.0},
            {"params": idm_decay, "lr": cfg.optim.idm_lr, "weight_decay": wd},
            {"params": idm_no_decay, "lr": cfg.optim.idm_lr, "weight_decay": 0.0},
        ]
    )
    jepa_scheduler = CosineWithWarmup(jepa_optimizer, total_steps, warmup_ratio=0.1)

    # -- LOAD CKPT
    start_epoch = 0
    ckpt_info = {}
    if cfg.meta.load_model:
        checkpoint_path = folder / cfg.meta.get("load_checkpoint", "latest.pth.tar")
        ckpt_info = load_checkpoint(
            checkpoint_path, jepa, jepa_optimizer, jepa_scheduler, device=device
        )
        start_epoch = ckpt_info.get("epoch", 0)

    # Compile
    if torch.cuda.is_available() and cfg.model.compile:
        logger.info("✅ Compiling model with torch.compile")
        jepa = torch.compile(jepa)

    # -- EVAL ONLY MODE
    if cfg.meta.get("eval_only_mode", False):
        if not enable_eval:
            raise ValueError("eval_only_mode requires enable_eval=True")
        logger.info("Running evaluation only (no training)")
        
        eval_results = run_patch_localization_eval(
            jepa, cfg, device, val_loader,
            vis_dir=str(folder / "eval_visualizations"),
            epoch=ckpt_info.get("epoch", start_epoch),
        )
        
        logger.info(f"Evaluation complete. Results: {eval_results}")
        return eval_results

    # eval_results = run_patch_localization_eval(jepa, cfg, device, val_loader, vis_dir=str(folder / "random_eval_visualizations"),)
    # logger.info(f"Evaluation completed on randomly initialized model as a baseline. Results: {eval_results}")

    # -- TRAINING LOOP
    for epoch in range(start_epoch, cfg.optim.epochs):
        epoch_start_time = time()
        epoch_stats = defaultdict(float)

        num_batches = 0

        pbar = tqdm(
            enumerate(loader),
            total=len(loader),
            desc=f"Epoch {epoch}/{cfg.optim.epochs - 1}",
            disable=cfg.logging.get("tqdm_silent", False),
            bar_format='{desc} | {n_fmt}/{total_fmt} batches | {postfix}',
            leave=False,
        )
        for _, (ref_crop, action, goal_crop, _, _) in pbar:
            ref_crop = ref_crop.to(device)
            action = action.to(device)
            goal_crop = goal_crop.to(device)
            
            # Calculate JEPA loss
            jepa_optimizer.zero_grad()

            with autocast(device.type, enabled=use_amp, dtype=dtype):
                reg_loss_dict, pred_loss = jepa(ref_crop, action, goal_crop)
            
            if cfg.model.regularizer.type == "vicreg":
                reg_loss = cfg.model.regularizer.cov_coeff * reg_loss_dict["cov_loss"] + cfg.model.regularizer.std_coeff * reg_loss_dict["std_loss"] + cfg.model.regularizer.idm_coeff * reg_loss_dict["idm_loss"]
                epoch_stats["cov_loss"] += reg_loss_dict["cov_loss"].item()
                epoch_stats["std_loss"] += reg_loss_dict["std_loss"].item()
            elif cfg.model.regularizer.type == "sigreg":
                reg_loss = cfg.model.regularizer.bcs_coeff * reg_loss_dict["bcs_loss"] + cfg.model.regularizer.idm_coeff * reg_loss_dict["idm_loss"]
                epoch_stats["bcs_loss"] += reg_loss_dict["bcs_loss"].item()
            epoch_stats["idm_loss"] += reg_loss_dict["idm_loss"].item()

            total_loss = reg_loss + cfg.model.regularizer.pred_coeff * pred_loss
            # Mixed precision backward pass
            scaler.scale(total_loss).backward()

            epoch_stats["total_loss"] += total_loss.item()
            epoch_stats["pred_loss"] += pred_loss.item()
            epoch_stats["reg_loss"] += reg_loss.item()
            
            for k, v in reg_loss_dict.items():
                epoch_stats[f"regloss/{k}"] += v.item()

            num_batches += 1

            # Using .get() with a default of 0 or False ensures smooth truthiness checks
            if cfg.optim.get("grad_clip_enc") or cfg.optim.get("grad_clip_proj") or cfg.optim.get("grad_clip_pred") or cfg.optim.get("grad_clip_idm"):
                scaler.unscale_(jepa_optimizer)
                
                torch.nn.utils.clip_grad_norm_(
                    jepa.encoder.parameters(), cfg.optim.get("grad_clip_enc", torch.inf)
                )
                torch.nn.utils.clip_grad_norm_(
                    projector.parameters(), cfg.optim.get("grad_clip_proj", torch.inf)
                )
                torch.nn.utils.clip_grad_norm_(
                    jepa.predictor.parameters(), cfg.optim.get("grad_clip_pred", torch.inf)
                )
                torch.nn.utils.clip_grad_norm_(
                    idm.parameters(), cfg.optim.get("grad_clip_idm", torch.inf)
                )

            scaler.step(jepa_optimizer)
            # Make sure you have this somewhere right after stepping!
            scaler.update()

            jepa_scheduler.step()

            # Update progress bar
            pbar.set_postfix(
                {
                    "bcs_loss": f"{reg_loss_dict['bcs_loss']:.4f}" if cfg.model.regularizer.type == "sigreg" else None,
                    "cov_loss": f"{reg_loss_dict['cov_loss']:.4f}" if cfg.model.regularizer.type == "vicreg" else None,
                    "std_loss": f"{reg_loss_dict['std_loss']:.4f}" if cfg.model.regularizer.type == "vicreg" else None,
                    "idm_loss": f"{int(reg_loss_dict['idm_loss'])}",
                    "pred_loss": f"{pred_loss.item():.4f}",
                }
            )

        if epoch % cfg.logging.log_every == 0:
            log_data = {
                f"train/{k}": v / num_batches
                for k, v in epoch_stats.items()
            }

            if cfg.logging.get("log_wandb"):
                wandb.log(log_data, step=epoch)

        # Patch Localization Eval
        if (enable_eval and epoch % cfg.meta.eval_every_n_epochs == 0):
            logger.info(f"Running patch localization eval at epoch {epoch}...")
            jepa.eval()  # Set model to evaluation mode
            with torch.no_grad():
                eval_results = run_patch_localization_eval(
                    jepa, cfg, device, val_loader,
                    vis_dir=str(folder / "eval_visualizations"),
                    epoch=epoch,
                )
            jepa.train()  # Set model back to training mode

            # Create a formatted dictionary
            formatted_eval_results = {
                k: (round(v * 100, 2)) for k, v in eval_results.items()
            }

            wandb.log(eval_results, step=epoch)
            logger.info(f"Evaluation results: {formatted_eval_results}")

        epoch_time = time() - epoch_start_time

        pbar.close()
        
        # Log epoch summary
        log_epoch(
            epoch,
            {
                "loss": epoch_stats["total_loss"] / num_batches,
                "reg": epoch_stats["reg_loss"] / num_batches,
                "pred": epoch_stats["pred_loss"] / num_batches,
            },
            total_epochs=cfg.optim.epochs,
            elapsed_time=epoch_time,
        )

        if cfg.logging.get("log_wandb"):
            wandb.log(
                {"epoch": epoch+1, "epoch_time": epoch_time},
                step=epoch,
            )

        # Save checkpoint
        save_checkpoint(
            latest_ckpt_path,
            model=jepa,
            optimizer=jepa_optimizer,
            scheduler=jepa_scheduler,
            epoch=epoch+1,
        )
        if epoch % cfg.logging.save_every_n_epochs == 0:
            save_checkpoint(
                folder / f"e-{epoch}.pth.tar",
                model=jepa,
                optimizer=jepa_optimizer,
                scheduler=jepa_scheduler,
                epoch=epoch+1,
            )


if __name__ == "__main__":
    fire.Fire(run)
