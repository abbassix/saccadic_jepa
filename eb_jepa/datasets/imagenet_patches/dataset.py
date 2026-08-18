# eb_jepa/imagenet_patches/dataset.py
import random
import math
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.v2 as v2

IMG_EXTENSIONS = {".jpeg", ".jpg", ".png"}



class ImageNetBaseDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.root = Path(config.dataset_path).expanduser() / config.split

        if not self.root.exists():
            raise ValueError(f"Dataset split directory missing: {self.root}")

        # 1. Map class directories directly (No heavy recursion)
        classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        # 2. Shallow index files folder-by-folder (Fastest runtime approach without cache files)
        self.samples = []
        for cls_name in classes:
            class_dir = self.root / cls_name
            label = self.class_to_idx[cls_name]
            for p in class_dir.iterdir(): # .iterdir() is vastly faster than .rglob() because it doesn't search recursively
                if p.suffix.lower() in IMG_EXTENSIONS:
                    self.samples.append((p, label))

        if not self.samples:
            raise ValueError(f"No images found under {self.root}")

    def __len__(self):
        return len(self.samples)


class ImagePatchLocatingDataset(ImageNetBaseDataset):
    """
    For JEPA Pretraining and Localization Evaluation.
    Resizes images, crops a 'reference' and a 'goal' patch entirely in PIL space,
    and returns both patches along with their relative and absolute spatial coordinates.
    """

    def __init__(self, config):
        super().__init__(config)
        if config.crop_size > config.img_size:
            raise ValueError("crop_size must be <= img_size")

        # 1. Resize PIL image directly without converting to Tensor yet
        self.resize_transform = transforms.Compose([
            transforms.Resize(config.img_size),
            transforms.CenterCrop(config.img_size),
        ])

        # 2. Appearance transforms on PIL crop + convert to normalized Tensor
        self.appearance_transform = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # 3. Standard transform for validation/unaugmented crops
        self.eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.viz_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.crop_size = config.crop_size
        self.img_size = config.img_size

    def _get_random_crop_bounds(self, width, height, rng=None):
        """
        Sample a valid crop origin from a resized image.

        If `rng` is provided (a random.Random instance), use it instead of the
        global `random` module so callers can obtain deterministic,
        reproducible crops (e.g. validation) without affecting global RNG
        state.
        """
        max_x = width - self.crop_size
        max_y = height - self.crop_size

        if max_x < 0 or max_y < 0:
            raise ValueError(
                f"Resized image ({width}x{height}) is smaller than "
                f"crop_size={self.crop_size}"
            )

        source = rng if rng is not None else random
        x = source.randint(0, max_x)
        y = source.randint(0, max_y)
        return x, y

    def __getitem__(self, idx):
        path, _ = self.samples[idx]

        # Load and resize PIL Image
        img = Image.open(path).convert("RGB")
        img = self.resize_transform(img)
        
        height, width = img.size

        rng = random.Random(idx) if self.config.split == "val" else None
        # Sample crop origins...
        x_ref, y_ref = self._get_random_crop_bounds(width, height, rng)
        x_goal, y_goal = self._get_random_crop_bounds(width, height, rng)

        # Slice crops in PIL space: (left, upper, right, lower)
        ref_crop = img.crop((x_ref, y_ref, x_ref + self.crop_size, y_ref + self.crop_size))
        goal_crop = img.crop((x_goal, y_goal, x_goal + self.crop_size, y_goal + self.crop_size))

        # Crop raw PIL image, then convert to basic unaugmented float tensor
        ref_crop = v2.functional.to_image(ref_crop)    # uint8 Tensor [3, H, W]
        goal_crop = v2.functional.to_image(goal_crop)  # uint8 Tensor [3, H, W]

        # 1. Calculate the raw pixel differences
        dx_pixels = float(x_goal - x_ref)
        dy_pixels = float(y_goal - y_ref)

        # This becomes your "action" target
        action = torch.tensor([dx_pixels, dy_pixels], dtype=torch.float32)

        # Keep your original origins around for evaluation metrics if needed
        ref_origin = torch.tensor([x_ref, y_ref], dtype=torch.float32)
        goal_origin = torch.tensor([x_goal, y_goal], dtype=torch.float32)

        # if self.return_full_image:
        #     return ref_crop, action, goal_crop, ref_origin, goal_origin, img
        return ref_crop, action, goal_crop, ref_origin, goal_origin


class ImageClassifierDataset(ImageNetBaseDataset):
    """
    For Downstream Linear Probe / Evaluation.
    Returns a standard single augmented image view and its respective class label.
    """

    def __init__(self, config):
        super().__init__(config)

        # Standard ImageNet classification augmentations based on split
        if config.split == "train":
            self.transform = transforms.Compose([
                transforms.Resize(config.img_size),
                transforms.CenterCrop(config.img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # Optional: Add transforms.Normalize here if needed
            ])
        else:  # val / test
            self.transform = transforms.Compose([
                transforms.Resize(config.img_size),
                transforms.CenterCrop(config.img_size),
                transforms.ToTensor(),
                # Optional: Add transforms.Normalize here if needed
            ])

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")

        img_tensor = self.transform(img)
        return img_tensor, label
