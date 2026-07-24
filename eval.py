# eval.py
"""
Patch-localization eval for the single-step (state, action) JEPA.

ref_crop is given; the true goal is goal_crop's center (unknown to the model at
eval time).

Two prediction methods are evaluated and compared:
  - IDM   : idm(ref_state, goal_state) -> predicted (dx, dy) directly.
  - Predictor grid-search: for a set of equi-spaced candidate goal top-left
    corners (stride configurable via cfg.eval.candidate_stride), the predictor
    is asked to predict the resulting embedding from (ref_state, candidate
    action). The candidate whose predicted embedding is closest (MSE) to the
    true goal embedding is taken as the prediction. This never touches actual
    pixels -- it is a pure embedding-space search -- so it runs on the fast
    5-tuple dataset just like the IDM metric.

The bulk metrics (eval/idm_ratio*, eval/pred_ratio*, eval/mse_ref_goal) use the
original, fast dataset (5-tuple: ref_crop, offset, goal_crop, ref_origin,
goal_origin) and never touch the full source image.

Visualization needs the *actual* pixels at an arbitrary predicted location
(so we can draw the predicted patch and compute an image-space MSE against
the goal patch), which the fast dataset doesn't provide. Rather than slow
down the whole eval loop by carrying the full image for every sample, we
re-load just the `num_vis_samples` chosen images directly from disk in a
separate, cheap pass (see `_load_sample_with_full_image`). The visualization
subset metrics (eval/vis_subset_*) are therefore only reported over that
small subset, not the full val set.
"""

import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

from eb_jepa.datasets.imagenet_patches.dataset import ImagePatchLocatingDataset
from eb_jepa.logging import get_logger

logger = get_logger(__name__)

_IDM_COLOR = "yellow"
_PREDICTOR_COLOR = "orange"


def _per_sample_mse(a, b):
    """Mean-squared-error per sample. a, b: [B, C, H, W] -> [B]"""
    return torch.mean((a - b) ** 2, dim=[1, 2, 3])


def _idm_predict_offset(idm_module, projector, idm_after_proj, ref_state, goal_state):
    """Predict ref -> goal pixel offset via the IDM head, given precomputed states."""
    pred = (
        idm_module.idm(projector(ref_state), projector(goal_state))
        if idm_after_proj
        else idm_module.idm(ref_state, goal_state)
    )
    return pred


def _predictor_grid_search(
    jepa, ref_state, goal_state, ref_origin, max_offset, cfg,
    stride, device, refine_iters, refine_window,
):
    """
    Coarse-to-fine candidate search:
      1. Full equi-spaced grid at `stride`, pick the best candidate per sample.
      2. For `refine_iters` rounds: halve the stride and search only a small
         (2*refine_window+1)^2 window around the current best corner. Update
         the best corner/mse only if a nearby candidate improves on it.
    Stops early once the stride would drop below 1px.

    stride, refine_iters, refine_window: read from cfg.eval.* by the caller.
    """
    B = ref_state.shape[0]
    state_shape = ref_state.shape[1:]

    def eval_candidates(corners):
        # corners: [B, C, 2]
        C = corners.shape[1]
        action = corners - ref_origin.unsqueeze(1)  # [B, C, 2]
        ref_exp = ref_state.unsqueeze(1).expand(B, C, *state_shape).reshape(B * C, *state_shape)
        goal_exp = goal_state.unsqueeze(1).expand(B, C, *state_shape).reshape(B * C, *state_shape)
        action_flat = action.reshape(B * C, 2)
        pred_mean, pred_log_var = jepa.predictor(ref_exp, action_flat)
        pred_var = torch.exp(pred_log_var)
        nll = F.gaussian_nll_loss(pred_mean, goal_exp, pred_var, reduction='none', full=False)
        reduce_dims = list(range(1, nll.dim()))
        return torch.mean(nll, dim=reduce_dims).reshape(B, C)

    # --- coarse pass (same as before) ---
    xs = torch.arange(0, max_offset + 1e-3, stride, device=device).clamp(max=max_offset)
    ys = torch.arange(0, max_offset + 1e-3, stride, device=device).clamp(max=max_offset)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    corners = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1).unsqueeze(0).expand(B, -1, -1)  # [B, C, 2]

    mse = eval_candidates(corners)
    best_idx = torch.argmin(mse, dim=1)
    best_corner = corners[torch.arange(B, device=device), best_idx]  # [B, 2]
    best_mse = mse.gather(1, best_idx.unsqueeze(1)).squeeze(1)

    # --- refinement rounds: shrink window around current best ---
    cur_stride = stride
    for _ in range(refine_iters):
        cur_stride /= 2.0
        if cur_stride < 1.0:
            break

        offsets_1d = torch.arange(-refine_window, refine_window + 1, device=device) * cur_stride
        ox, oy = torch.meshgrid(offsets_1d, offsets_1d, indexing="xy")
        local_offsets = torch.stack([ox.reshape(-1), oy.reshape(-1)], dim=-1)  # [K, 2]
        local_corners = (best_corner.unsqueeze(1) + local_offsets.unsqueeze(0)).clamp(0, max_offset)  # [B, K, 2]

        mse_local = eval_candidates(local_corners)
        best_idx_local = torch.argmin(mse_local, dim=1)
        cand_corner = local_corners[torch.arange(B, device=device), best_idx_local]
        cand_mse = mse_local.gather(1, best_idx_local.unsqueeze(1)).squeeze(1)

        improved = cand_mse < best_mse
        best_corner = torch.where(improved.unsqueeze(1), cand_corner, best_corner)
        best_mse = torch.where(improved, cand_mse, best_mse)

    best_offset_px = best_corner - ref_origin
    return best_offset_px, best_mse


def _load_sample_with_full_image(dataset, idx):
    """
    Re-load a single val-split sample directly from disk, replicating
    ImagePatchLocatingDataset's own ref/goal crop sampling exactly (same
    idx-seeded RNG the dataset uses for the val split), but additionally
    returning the full resized source image. Only meant to be called for a
    handful of visualization samples -- not the whole eval set.

    Returns: ref_crop [3,S,S], goal_crop [3,S,S], ref_origin [2], goal_origin [2],
             img [3,R,R]  (all CPU tensors, unbatched)
    """
    # TODO
    # assert getattr(dataset.config, "split", None) == "val", (
    #     "_load_sample_with_full_image relies on the dataset's idx-seeded RNG, "
    #     "which is only deterministic for split='val'."
    # )

    path, _ = dataset.samples[idx]
    img = Image.open(path).convert("RGB")
    img = dataset.transform(img)

    rng = random.Random(idx)
    x_ref, y_ref = dataset._get_random_crop_bounds(rng)
    x_goal, y_goal = dataset._get_random_crop_bounds(rng)

    crop_size = dataset.crop_size
    ref_crop = img[:, y_ref : y_ref + crop_size, x_ref : x_ref + crop_size]
    goal_crop = img[:, y_goal : y_goal + crop_size, x_goal : x_goal + crop_size]
    ref_origin = torch.tensor([x_ref, y_ref], dtype=torch.float32)
    goal_origin = torch.tensor([x_goal, y_goal], dtype=torch.float32)

    return ref_crop, goal_crop, ref_origin, goal_origin, img


def _draw_box(draw, origin_xy, size, color, width=2):
    """Draw a rectangle outline of `size`x`size` starting at origin_xy=(x, y)."""
    x, y = origin_xy
    draw.rectangle([x, y, x + size, y + size], outline=color, width=width)


def _save_localization_visualization(
    full_image,               # [3, img_size, img_size], the actual source image
    ref_origin,
    goal_origin,
    idm_pred_origin,          # (x, y) or None
    predictor_pred_origin,    # (x, y) or None
    mse_ref_goal,             # float
    mse_idm_pred_goal,        # float or None
    mse_predictor_pred_goal,  # float or None
    img_size,
    crop_size,
    out_path,
):
    """
    Draw the actual resized source image (not a blank canvas) with:
      - blue box   = reference patch (given)
      - green box  = true goal patch
      - yellow box = IDM-predicted goal location
      - orange box = predictor grid-search predicted goal location
    and print MSE(ref,goal) / MSE(idm_pred,goal) / MSE(pred_search,goal) below
    the image, each to 4 decimal places. Saves as PNG.
    """
    x_ref, y_ref = int(ref_origin[0].item()), int(ref_origin[1].item())
    x_goal, y_goal = int(goal_origin[0].item()), int(goal_origin[1].item())

    img_np = (full_image.clamp(0, 1).cpu().permute(1, 2, 0) * 255).byte().numpy()
    canvas_img = Image.fromarray(img_np, mode="RGB")

    num_text_lines = 1 + int(idm_pred_origin is not None) + int(predictor_pred_origin is not None)
    text_h = 16 * num_text_lines + 8
    out_img = Image.new("RGB", (img_size, img_size + text_h), color=(0, 0, 0))
    out_img.paste(canvas_img, (0, 0))
    draw = ImageDraw.Draw(out_img)

    _draw_box(draw, (x_ref, y_ref), crop_size, "blue")
    _draw_box(draw, (x_goal, y_goal), crop_size, "green")

    if idm_pred_origin is not None:
        px, py = idm_pred_origin
        px_i, py_i = int(round(px)), int(round(py))
        _draw_box(draw, (px_i, py_i), crop_size, _IDM_COLOR)
        draw.text((px_i + 2, py_i + 2), "IDM", fill=_IDM_COLOR)

    if predictor_pred_origin is not None:
        px, py = predictor_pred_origin
        px_i, py_i = int(round(px)), int(round(py))
        _draw_box(draw, (px_i, py_i), crop_size, _PREDICTOR_COLOR)
        draw.text((px_i + 2, py_i + 2), "PRED", fill=_PREDICTOR_COLOR)

    y_text = img_size + 4
    draw.text((4, y_text), f"MSE(ref,goal)          = {mse_ref_goal:.4f}", fill="white")
    if mse_idm_pred_goal is not None:
        y_text += 16
        draw.text((4, y_text), f"MSE(idm_pred,goal)     = {mse_idm_pred_goal:.4f}", fill=_IDM_COLOR)
    if mse_predictor_pred_goal is not None:
        y_text += 16
        draw.text((4, y_text), f"MSE(pred_search,goal)  = {mse_predictor_pred_goal:.4f}", fill=_PREDICTOR_COLOR)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(out_path)


@torch.no_grad()
def _run_visualization_pass(
    jepa,
    cfg,
    device,
    dataset,
    vis_indices,
    idm_module,
    projector,
    idm_after_proj,
    has_idm,
    has_predictor,
    vis_out_dir,
):
    """
    For each index in vis_indices: reload the sample + its full source image
    from disk, run the (single-level) IDM and/or predictor grid-search
    prediction, and save a PNG. Purely for visual inspection -- no metrics
    are aggregated or returned from this pass.
    """
    max_offset = float(cfg.data.img_size - cfg.data.crop_size)
    crop_size = cfg.data.crop_size

    for global_i in sorted(vis_indices):
        ref_crop, goal_crop, ref_origin, goal_origin, img = _load_sample_with_full_image(dataset, global_i)

        ref_crop = ref_crop.unsqueeze(0).to(device)
        goal_crop = goal_crop.unsqueeze(0).to(device)
        ref_origin = ref_origin.unsqueeze(0).to(device)
        goal_origin = goal_origin.unsqueeze(0).to(device)
        img = img.unsqueeze(0).to(device)

        mse_ref_goal = _per_sample_mse(ref_crop, goal_crop).item()

        ref_state = jepa.encoder(ref_crop)
        goal_state = jepa.encoder(goal_crop)

        candidate_stride = float(OmegaConf.select(cfg, "eval.candidate_stride", default=32))
        refine_iters = int(OmegaConf.select(cfg, "eval.refine_iters", default=4))
        refine_window = int(OmegaConf.select(cfg, "eval.refine_window", default=1))

        idm_pred_origin = None
        mse_idm_pred_goal = None
        if has_idm:
            idm_offset = _idm_predict_offset(
                idm_module, projector, idm_after_proj, ref_state, goal_state
            )
            idm_origin_clamped = (ref_origin + idm_offset).clamp(0, max_offset)
            x, y = int(round(idm_origin_clamped[0, 0].item())), int(round(idm_origin_clamped[0, 1].item()))
            idm_crop = img[:, :, y : y + crop_size, x : x + crop_size]
            mse_idm_pred_goal = _per_sample_mse(idm_crop, goal_crop).item()
            idm_pred_origin = (x, y)

        predictor_pred_origin = None
        mse_predictor_pred_goal = None
        if has_predictor:
            pred_offset, _ = _predictor_grid_search(
                jepa, ref_state, goal_state, ref_origin, max_offset, cfg, candidate_stride, device, refine_iters=refine_iters, refine_window=refine_window,
            )
            pred_origin_clamped = (ref_origin + pred_offset).clamp(0, max_offset)
            x, y = int(round(pred_origin_clamped[0, 0].item())), int(round(pred_origin_clamped[0, 1].item()))
            pred_crop = img[:, :, y : y + crop_size, x : x + crop_size]
            mse_predictor_pred_goal = _per_sample_mse(pred_crop, goal_crop).item()
            predictor_pred_origin = (x, y)

        _save_localization_visualization(
            full_image=img[0],
            ref_origin=ref_origin[0],
            goal_origin=goal_origin[0],
            idm_pred_origin=idm_pred_origin,
            predictor_pred_origin=predictor_pred_origin,
            mse_ref_goal=mse_ref_goal,
            mse_idm_pred_goal=mse_idm_pred_goal,
            mse_predictor_pred_goal=mse_predictor_pred_goal,
            img_size=cfg.data.img_size,
            crop_size=crop_size,
            out_path=vis_out_dir / f"sample_{global_i:05d}.png",
        )


@torch.no_grad()
def run_patch_localization_eval(
    jepa,
    cfg,
    device,
    val_loader: DataLoader = None,  # Preferred: Pass the loader directly from init_data
    visualize: bool = True,
    num_vis_samples: int = 32,
    vis_dir: str = "eval_visualizations",
    vis_seed: int = 0,
    epoch: int = None,
):
    # Fallback dataset instantiation if val_loader isn't provided directly
    if val_loader is None:
        val_dset = ImagePatchLocatingDataset(
            config=OmegaConf.merge(cfg.data, {"split": "val"})
        )
        val_loader = DataLoader(
            val_dset,
            batch_size=cfg.eval.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=cfg.data.get("num_workers", 0),
            pin_memory=cfg.data.get("pin_mem", False),
            persistent_workers=True, # Safe default for single runs
        )

    idm_module = getattr(jepa.regularizer, "idm_loss_fn", None)
    idm_after_proj = jepa.regularizer.idm_after_proj
    projector = jepa.regularizer.projector
    has_idm = idm_module is not None and idm_module.idm is not None
    has_predictor = getattr(jepa, "predictor", None) is not None
    if not (has_idm or has_predictor):
        raise ValueError("We should have at least one predictor!")

    candidate_stride = float(OmegaConf.select(cfg, "eval.candidate_stride", default=32))
    refine_iters = int(OmegaConf.select(cfg, "eval.refine_iters", default=4))
    refine_window = int(OmegaConf.select(cfg, "eval.refine_window", default=1))

    # --- Initialize storage for 4 distance bins ---
    num_bins = 4
    max_offset = float(cfg.data.img_size - cfg.data.crop_size)
    # Ensure we don't divide by zero if resize == crop
    bin_width = max(1.0, max_offset / num_bins)

    idm_ratios_by_bin = {i: [] for i in range(num_bins)}
    pred_ratios_by_bin = {i: [] for i in range(num_bins)}
    # ------------------------------------------------

    idm_ratios = []
    pred_ratios = []

    # --- Visualization sample selection (indices only -- images are loaded
    # separately, after the main loop, by _run_visualization_pass). ---
    vis_indices = set()
    if visualize and (has_idm or has_predictor):
        dataset_len = len(val_loader.dataset)
        rng = random.Random(vis_seed)
        k = min(num_vis_samples, dataset_len)
        vis_indices = set(rng.sample(range(dataset_len), k))

    pbar = tqdm(
            enumerate(val_loader),
            total=len(val_loader),
            desc="Validating Locating...",
            disable=cfg.logging.get("tqdm_silent", False),
            bar_format='{desc} | {n_fmt}/{total_fmt} batches | {postfix}',
        )
    # Compatible unpacking format matching our 5-tuple output
    for idx, (ref_crop, offset_px, goal_crop, ref_origin, goal_origin) in pbar:
        # ... (Move to device) ...
        ref_crop = ref_crop.to(device)
        goal_crop = goal_crop.to(device)
        offset_px = offset_px.to(device)
        ref_origin = ref_origin.to(device)
        goal_origin = goal_origin.to(device)

        baseline_dist = torch.norm(offset_px, dim=-1)
        valid = baseline_dist > 4.0

        if not valid.any():
            continue

        # Shared bin assignment (by baseline distance) used by both methods.
        valid_baseline = baseline_dist[valid]
        bin_indices = (valid_baseline / bin_width).long().clamp(0, num_bins - 1)

        ref_state = jepa.encoder(ref_crop)
        goal_state = jepa.encoder(goal_crop)

        # ---- IDM-BASED PREDICTION ----
        if has_idm:
            pred_offset_px = _idm_predict_offset(
                idm_module, projector, idm_after_proj, ref_state, goal_state
            )
            idm_err = torch.norm(pred_offset_px - offset_px, dim=-1)
            idm_ratio = idm_err / baseline_dist

            idm_valid_ratios = idm_ratio[valid]
            for b in range(num_bins):
                mask = (bin_indices == b)
                if mask.any():
                    idm_ratios_by_bin[b].append(idm_valid_ratios[mask].cpu())

            idm_ratios.append(idm_ratio[valid].cpu())

        # ---- PREDICTOR GRID-SEARCH PREDICTION ----
        if has_predictor:
            pred_offset_px, _ = _predictor_grid_search(
                jepa, ref_state, goal_state, ref_origin, max_offset, cfg,
                candidate_stride, device, refine_iters=refine_iters, refine_window=refine_window,
            )

            pred_err = torch.norm(pred_offset_px - offset_px, dim=-1)
            pred_ratio = pred_err / baseline_dist

            pred_valid_ratios = pred_ratio[valid]
            for b in range(num_bins):
                mask = (bin_indices == b)
                if mask.any():
                    pred_ratios_by_bin[b].append(pred_valid_ratios[mask].cpu())

            pred_ratios.append(pred_ratio[valid].cpu())

    def _stats(x, prefix):
        if len(x) == 0:
            return {}
        q = torch.quantile(x, torch.tensor([0.25, 0.5, 0.75]))
        return {
            f"{prefix}_mean": x.mean().item(),
            f"{prefix}_q25": q[0].item(),
            f"{prefix}_med": q[1].item(),
            f"{prefix}_q75": q[2].item(),
        }

    results = {}
    if has_idm and idm_ratios:
        for b in range(num_bins):
            if idm_ratios_by_bin[b]:
                combined_ratios = torch.cat(idm_ratios_by_bin[b])
                results.update(_stats(combined_ratios, f"eval/idm_ratio_bin_{b}"))
        results.update(_stats(torch.cat(idm_ratios), "eval/idm_ratio"))

    if has_predictor and pred_ratios:
        for b in range(num_bins):
            if pred_ratios_by_bin[b]:
                combined_ratios = torch.cat(pred_ratios_by_bin[b])
                results.update(_stats(combined_ratios, f"eval/pred_ratio_bin_{b}"))
        results.update(_stats(torch.cat(pred_ratios), "eval/pred_ratio"))

    # --- Visualization pass (visual inspection only; no metrics reported) ---
    if visualize and (has_idm or has_predictor) and vis_indices:
        idm_mean = results.get("eval/idm_ratio_med")
        pred_mean = results.get("eval/pred_ratio_med")

        name_parts = []
        if epoch is not None:
            name_parts.append(f"epoch{epoch}")
        if idm_mean is not None:
            name_parts.append(f"idm{idm_mean:.3f}")
        if pred_mean is not None:
            name_parts.append(f"pred{pred_mean:.3f}")
        subdir = "_".join(name_parts) if name_parts else "eval"

        vis_out_dir = Path(vis_dir) / subdir
        _run_visualization_pass(
            jepa=jepa,
            cfg=cfg,
            device=device,
            dataset=val_loader.dataset,
            vis_indices=vis_indices,
            idm_module=idm_module,
            projector=projector,
            idm_after_proj=idm_after_proj,
            has_idm=has_idm,
            has_predictor=has_predictor,
            vis_out_dir=vis_out_dir,
        )
        logger.info(f"Saved {len(vis_indices)} localization visualizations to {vis_out_dir.resolve()}")

    return results
