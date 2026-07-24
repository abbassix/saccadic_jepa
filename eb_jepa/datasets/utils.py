# eb_jepa/datasets/utils.py
from pathlib import Path

import torch
from omegaconf import OmegaConf

from eb_jepa.datasets.imagenet_patches.dataset import (
    ImagePatchLocatingDataset,
    ImageClassifierDataset,
)

DATASETS_DIR = Path(__file__).parent


def init_data(env_name, config=None, **kwargs):
    """Initialize data loaders for the specified environment and task.

    Args:
        env_name: Name of the environment (e.g., "imagenet_patches").
        config: Configuration object containing data properties.

    Returns:
        Tuple of (train_loader, val_loader, config).
    """
    if env_name == "imagenet_patches":
        num_workers = config.data.get("num_workers", 0)
        pin_mem = config.data.get("pin_mem", False)
        persistent_workers = config.data.get("persistent_workers", False) and num_workers > 0
        
        # Determine which dataset variant to construct based on the config task
        # Expected task values: "locating" or "classification"
        task = config.data.get("task", "locating")
        if task == "locating":
            dataset_cls = ImagePatchLocatingDataset
        elif task == "classification":
            dataset_cls = ImageClassifierDataset
        else:
            raise ValueError(f"Unknown data task: {task}. Choose 'locating' or 'classification'.")

        # 1. Build Training Dataset & Loader
        train_config = OmegaConf.merge(config.data, {"split": "train"})
        dset = dataset_cls(config=train_config)
        
        loader = torch.utils.data.DataLoader(
            dset,
            batch_size=config.data.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_mem,
            drop_last=True,
            persistent_workers=persistent_workers,
        )

        # 2. Build Validation Dataset & Loader
        val_config = OmegaConf.merge(config.data, {"split": "val"})
        # val_config = OmegaConf.merge(config.data, {"split": "val", "return_full_image": True})
        val_dset = dataset_cls(config=val_config)
        
        # Pull evaluation batch size if provided, otherwise default to a safe value
        batch_size = config.data.get("batch_size", 512)

        # TODO
        val_loader = torch.utils.data.DataLoader(
            val_dset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_mem,
            drop_last=False,  # Don't drop remaining samples during strict evaluations
            persistent_workers=False,
        )

        # Track sizes dynamically inside config tracking
        config.training.size = len(loader)
        # TODO
        config.eval.size = len(val_loader)

        # TODO
        return loader, val_loader, config
        # return loader, config

    else:
        raise ValueError(f"Unknown env: {env_name}.")
