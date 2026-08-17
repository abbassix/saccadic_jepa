# eb_jepa/imagenet_patches/dataset.py
import random
import math
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
        self.transform = transforms.Compose([
            transforms.Resize(config.img_size),  # shortest side = img_size
            transforms.CenterCrop(config.img_size),
            transforms.ToTensor(),
        ])

        self.crop_size = config.crop_size
        self.img_size = config.img_size
        # self.min_distance = config.min_distance
        # self.sigma = config.sigma
        # self.max_distance = config.max_distance
        # self.center_sigma = config.center_sigma
        
        # Extremely weak transform of color and random noise
        # self.goal_crop_transform = transforms.Compose([
        #     # 1. Color disruption (independent)
        #     transforms.RandomApply([
        #         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
        #     ], p=0.8),
        #     transforms.RandomGrayscale(p=0.2),
            
        #     # 2. Detail destruction
        #     transforms.RandomApply([
        #         transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))
        #     ], p=0.5),
            
        #     # 3. Geometric micro-distortions (Center-anchored)
        #     transforms.RandomAffine(
        #         degrees=(-5, 5),
        #         scale=(0.9, 1.1),
        #         shear=(-3, 3),
        #         center=(self.crop_size // 2, self.crop_size // 2) # Keep (dx, dy) center intact!
        #     ),
            
        #     # Convert to tensor / normalize if not already done
        #     # ...
            
        #     # 4. Occlusion (applied post-tensor)
        #     transforms.RandomErasing(p=0.4, scale=(0.05, 0.20), ratio=(0.3, 3.3))
        # ])
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
    
    # def _get_random_crop_bounds(self, width, height, min_distance=None, rng=None):
    #     """
    #     Samples reference and goal crop bounds such that they are separated by 
    #     at least `min_distance` pixels along AT LEAST ONE axis (x or y).

    #     - min_distance = crop_size     --> non-overlapping on at least one axis
    #     - min_distance = crop_size + K --> non-overlapping with K pixels gap on at least one axis
    #     - min_distance < crop_size     --> controlled max overlap
    #     """
    #     max_x = width - self.crop_size
    #     max_y = height - self.crop_size

    #     if max_x < 0 or max_y < 0:
    #         raise ValueError(
    #             f"Resized image ({width}x{height}) is smaller than "
    #             f"crop_size={self.crop_size}"
    #         )

    #     # Default min_distance to crop_size if not specified (strictly non-overlapping)
    #     D = min_distance if min_distance is not None else self.crop_size

    #     # At least one axis must be large enough to accommodate min_distance
    #     can_split_x = max_x >= D
    #     can_split_y = max_y >= D

    #     if not can_split_x and not can_split_y:
    #         raise ValueError(
    #             f"Image ({width}x{height}) is too small to place two crops with "
    #             f"min_distance={D} on either axis. Requires width or height >= {self.crop_size + D}."
    #         )

    #     source = rng if rng is not None else random

    #     # Choose which axis will enforce the min_distance constraint
    #     valid_axes = []
    #     if can_split_x:
    #         valid_axes.append('x')
    #     if can_split_y:
    #         valid_axes.append('y')
            
    #     chosen_axis = source.choice(valid_axes)

    #     if chosen_axis == 'x':
    #         # Enforce distance D on x-axis
    #         x_ref = source.randint(0, max_x - D)
    #         x_goal = source.randint(x_ref + D, max_x)

    #         # Sample y freely across all valid positions
    #         y_ref = source.randint(0, max_y)
    #         y_goal = source.randint(0, max_y)
    #     else:
    #         # Sample x freely across all valid positions
    #         x_ref = source.randint(0, max_x)
    #         x_goal = source.randint(0, max_x)

    #         # Enforce distance D on y-axis
    #         y_ref = source.randint(0, max_y - D)
    #         y_goal = source.randint(y_ref + D, max_y)

    #     # Randomly swap x and y bounds independently with 50% probability 
    #     # so ref isn't always top-left relative to goal
    #     if source.random() < 0.5:
    #         x_ref, x_goal = x_goal, x_ref
    #     if source.random() < 0.5:
    #         y_ref, y_goal = y_goal, y_ref

    #     return x_ref, y_ref, x_goal, y_goal
    
    # def _get_random_crop_bounds(self, width, height, min_distance=None, sigma=None, rng=None):
    #     """
    #     Samples reference and goal crop bounds such that they are separated by 
    #     at least an effective `min_distance` along AT LEAST ONE axis (x or y).
        
    #     The effective min_distance is sampled from N(min_distance, sigma^2).
    #     """
    #     max_x = width - self.crop_size
    #     max_y = height - self.crop_size

    #     if max_x < 0 or max_y < 0:
    #         raise ValueError(
    #             f"Resized image ({width}x{height}) is smaller than "
    #             f"crop_size={self.crop_size}"
    #         )

    #     source = rng if rng is not None else random

    #     # 1. Determine base mean for min_distance
    #     mean_D = min_distance if min_distance is not None else self.crop_size

    #     # Maximum possible separation distance the image can physically support on at least one axis
    #     max_possible_D = max(max_x, max_y)

    #     # 2. Sample effective min_distance from Gaussian distribution
    #     if sigma is not None and sigma > 0:
    #         # Sample D ~ N(mean_D, sigma)
    #         sampled_D = source.gauss(mean_D, sigma)
    #         # Clamp between 0 and the maximum allowable distance for this image
    #         clamped_D = max(0, min(sampled_D, max_possible_D))
    #         D = int(round(clamped_D))
    #     else:
    #         D = int(min(mean_D, max_possible_D))
        
    #     # X. If D is zero, allow completely unconstrained independent sampling on both axes for validation or testing scenarios
    #     if D == 0:
    #         # Completely unconstrained independent sampling on both axes
    #         x_ref = source.randint(0, max_x)
    #         x_goal = source.randint(0, max_x)
    #         y_ref = source.randint(0, max_y)
    #         y_goal = source.randint(0, max_y)
    #         return x_ref, y_ref, x_goal, y_goal

    #     # 3. Check which axes can fit the sampled min_distance
    #     can_split_x = max_x >= D
    #     can_split_y = max_y >= D

    #     if not can_split_x and not can_split_y:
    #         raise ValueError(
    #             f"Image ({width}x{height}) is too small to place two crops with "
    #             f"sampled min_distance={D} (mean={mean_D}, sigma={sigma}) on either axis. "
    #             f"Requires width or height >= {self.crop_size + D}."
    #         )

    #     # 4. Pick an axis that satisfies D
    #     valid_axes = []
    #     if can_split_x:
    #         valid_axes.append('x')
    #     if can_split_y:
    #         valid_axes.append('y')
            
    #     chosen_axis = source.choice(valid_axes)

    #     # 5. Sample crop coordinates
    #     if chosen_axis == 'x':
    #         # Enforce distance D on x-axis
    #         x_ref = source.randint(0, max_x - D)
    #         x_goal = x_ref + D

    #         # Sample y freely
    #         y_ref = source.randint(0, max_y)
    #         y_goal = source.randint(0, max_y)
            
    #         # Randomly swap ONLY the constrained axis (or swap order of ref/goal)
    #         if source.random() < 0.5:
    #             x_ref, x_goal = x_goal, x_ref
    #     else:
    #         # Sample x freely
    #         x_ref = source.randint(0, max_x)
    #         x_goal = source.randint(0, max_x)

    #         # Enforce distance D on y-axis
    #         y_ref = source.randint(0, max_y - D)
    #         y_goal = y_ref + D
            
    #         # Randomly swap ONLY the constrained axis
    #         if source.random() < 0.5:
    #             y_ref, y_goal = y_goal, y_ref

    #     return x_ref, y_ref, x_goal, y_goal
        
    # def _get_random_crop_bounds(
    #     self,
    #     width,
    #     height,
    #     min_dist=64,
    #     max_dist=192,
    #     center_sigma=35.0,
    #     rng=None
    # ):
    #     source = rng if rng is not None else random
        
    #     max_x = width - self.crop_size
    #     max_y = height - self.crop_size

    #     if max_x < 0 or max_y < 0:
    #         raise ValueError(
    #             f"Image dimensions ({width}x{height}) are smaller than crop_size={self.crop_size}"
    #         )

    #     # Center origins along both axes
    #     center_x = max_x / 2.0
    #     center_y = max_y / 2.0

    #     # 1. Sample C1 origin anchored around (center_x, center_y)
    #     while True:
    #         x1 = round(source.gauss(center_x, center_sigma))
    #         y1 = round(source.gauss(center_y, center_sigma))
    #         if 0 <= x1 <= max_x and 0 <= y1 <= max_y:
    #             break

    #     # 2. Sample C2 using distance offset
    #     for _ in range(1000):
    #         r = source.uniform(min_dist, max_dist)
    #         theta = source.uniform(0, 2 * math.pi)
            
    #         x2 = round(x1 + r * math.cos(theta))
    #         y2 = round(y1 + r * math.sin(theta))

    #         # Check bounds for asymmetric width and height
    #         if 0 <= x2 <= max_x and 0 <= y2 <= max_y:
    #             # 50% chance to swap C1 and C2
    #             if source.random() < 0.5:
    #                 x1, y1, x2, y2 = x2, y2, x1, y1
                
    #             return x1, y1, x2, y2

    #     # Fallback to standard center crops if sampling fails
    #     fallback_x = round(center_x)
    #     fallback_y = round(center_y)
    #     return fallback_x, fallback_y, fallback_x, fallback_y

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

        rng = random.Random(idx) if self.config.split == "val" else None
        x_ref, y_ref = self._get_random_crop_bounds(width, height, rng)
        x_goal, y_goal = self._get_random_crop_bounds(width, height, rng)
        # x_ref, y_ref, x_goal, y_goal = self._get_random_crop_bounds(width, height, self.min_distance, self.sigma, rng)
        # x_ref, y_ref, x_goal, y_goal = self._get_random_crop_bounds(width, height, self.min_distance, self.max_distance, self.center_sigma, rng)

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
        
        # Only augment goal crop during training
        # if self.config.split == "train":
        #     goal_crop = self.goal_crop_transform(goal_crop)
        

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
