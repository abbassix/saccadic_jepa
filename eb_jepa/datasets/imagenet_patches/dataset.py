# eb_jepa/imagenet_patches/dataset.py
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

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

        # Resize only—no cropping here. We handle cropping manually for efficiency.        
        self.transforms = transforms.Compose([
            transforms.Resize(config.img_size),  # shortest side = img_size
            transforms.CenterCrop(config.img_size),
            transforms.ToTensor(),
        ])

        self.crop_size = config.crop_size
        self.img_size = config.img_size

        # Off by default: the original fast 5-tuple return, no behavior change for
        # existing callers (training loop, default val loader). When True, __getitem__
        # additionally returns the full resized source image -- used by eval code
        # that needs real pixels at an arbitrary (e.g. model-predicted) location.
        self.return_full_image = bool(config.get("return_full_image", False))

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

        # Load and resize
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        
        _, height, width = img.shape

        # Sample crop origins. For the val split, seed a per-sample local RNG
        # off `idx` so the same (ref, goal) crop pair is used every epoch,
        # making eval/idm_ratio comparable across epochs/runs instead of
        # being computed against a different random benchmark each time.
        # A local random.Random is used (not the global `random` module) so
        # this doesn't affect train-time augmentation randomness or reseed
        # global state in a way that could interact with data loader workers.
        if self.config.split == "val":
            rng = random.Random(idx)
            x_ref, y_ref = self._get_random_crop_bounds(width, height, rng)
            x_goal, y_goal = self._get_random_crop_bounds(width, height, rng)
        else:
            x_ref, y_ref = self._get_random_crop_bounds(width, height)
            x_goal, y_goal = self._get_random_crop_bounds(width, height)

        # Slice tensor views instead of cropping in PIL
        ref_crop = img[
            :,
            y_ref : y_ref + self.crop_size,
            x_ref : x_ref + self.crop_size,
        ]
        goal_crop = img[
            :,
            y_goal : y_goal + self.crop_size,
            x_goal : x_goal + self.crop_size,
        ]

        # 1. Calculate the raw pixel differences
        dx_pixels = float(x_goal - x_ref)
        dy_pixels = float(y_goal - y_ref)

        # This becomes your "action" target
        action = torch.tensor([dx_pixels, dy_pixels], dtype=torch.float32)

        # Keep your original origins around for evaluation metrics if needed
        ref_origin = torch.tensor([x_ref, y_ref], dtype=torch.float32)
        goal_origin = torch.tensor([x_goal, y_goal], dtype=torch.float32)

        if self.return_full_image:
            return ref_crop, action, goal_crop, ref_origin, goal_origin, img
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
