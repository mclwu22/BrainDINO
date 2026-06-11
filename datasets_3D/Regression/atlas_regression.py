import torch
import numpy as np
import nibabel as nib
import os
import pandas as pd
import warnings
from torch.utils.data import Dataset
from monai.transforms import (
    RandAffined, RandFlipd, RandGaussianNoised,
    RandGaussianSmoothd, RandAdjustContrastd
)

def zscore_normalize(volume, clip_percentile=True):
    """
    Robust z-score normalization for 3D MRI.

    Parameters
    ----------
    volume : np.ndarray
        3D MRI volume, assumed to be float32 or convertible to float32.
        Shape: (D,H,W) or (H,W,D)

    clip_percentile : bool
        If True, clip intensities using 1%–99% percentiles (recommended for MRI).

    Returns
    -------
    out : np.ndarray
        z-score normalized volume, with same shape as input.
        (mean ~0, std ~1 inside brain, outliers suppressed)
    """
    vol = volume.astype(np.float32)

    # ----------------------------------------
    # Step 1: optional percentile clipping
    # ----------------------------------------
    if clip_percentile:
        # Compute percentiles only on non-zero voxels (avoid air/background)
        nonzero = vol[vol > 0]
        if nonzero.size < 10:
            # fallback if almost all values are zero
            nonzero = vol.reshape(-1)

        lo = np.percentile(nonzero, 1)   # 1st percentile
        hi = np.percentile(nonzero, 99)  # 99th percentile

        # Clip intensities to robust range
        vol = np.clip(vol, lo, hi)

    # ----------------------------------------
    # Step 2: compute robust mean and std from non-zero region
    # ----------------------------------------
    nz = vol[vol > 0]

    if nz.size > 0:
        mean = nz.mean()
        std = nz.std()
    else:
        # fallback: use whole volume
        mean = vol.mean()
        std = vol.std()

    # Avoid division by zero
    std = max(std, 1e-6)

    # ----------------------------------------
    # Step 3: z-score normalize
    # ----------------------------------------
    vol = (vol - mean) / std

    return vol.astype(np.float32)


class ATLASRegressionSet(Dataset):
    """
    ATLAS dataset loader for regression (Days Post Stroke prediction)

    Dataset: ATLAS (Anatomical Tracings of Lesions After Stroke)
    Task: Regression - Predict days post stroke from brain MRI
    Target: Continuous value (days)

    CSV Format:
        Days Post Stroke,image_path
        1.0,/path/to/sub-r049s016.nii.gz
        2.0,/path/to/sub-r009s021.nii.gz

    Supports train.csv and valid.csv
    """
    def __init__(self, config, base_dir, train_ratio=1.0, flag='train', teacher=None):
        super().__init__()
        self.config = config
        self.flag = flag
        self.base_dir = base_dir
        self.n_slices = config.n_slices if hasattr(config, 'n_slices') else 128
        self.teacher = teacher

        def _cfg(name, default):
            return getattr(self.config, name, default)

        # Augmentation hyperparameters (override via config)
        self.aug_rotate_range = tuple(_cfg("atlas_aug_rotate_range", _cfg("aug_rotate_range", (0.1, 0.1, 0.1))))
        self.aug_translate_range = tuple(_cfg("atlas_aug_translate_range", _cfg("aug_translate_range", (5, 5, 5))))
        self.aug_scale_range = tuple(_cfg("atlas_aug_scale_range", _cfg("aug_scale_range", (0.1, 0.1, 0.1))))
        self.aug_affine_prob = float(_cfg("atlas_aug_affine_prob", _cfg("aug_affine_prob", 0.5)))
        self.aug_flip_prob = float(_cfg("atlas_aug_flip_prob", _cfg("aug_flip_prob", 0.5)))
        self.aug_smooth_prob = float(_cfg("atlas_aug_smooth_prob", _cfg("aug_smooth_prob", 0.2)))
        self.aug_noise_prob = float(_cfg("atlas_aug_noise_prob", _cfg("aug_noise_prob", 0.2)))
        self.aug_noise_std = float(_cfg("atlas_aug_noise_std", _cfg("aug_noise_std", 0.05)))
        self.aug_contrast_prob = float(_cfg("atlas_aug_contrast_prob", _cfg("aug_contrast_prob", 0.2)))
        self.aug_contrast_gamma = tuple(_cfg("atlas_aug_contrast_gamma", _cfg("aug_contrast_gamma", (0.7, 1.3))))

        self.all_images = []
        self.all_labels = []

        def _resolve_csv_path(cur_flag):
            """Allow configurable CSV names for train/val/test splits."""
            if cur_flag == "train":
                csv_name = getattr(self.config, "atlas_train_csv", "train.csv")
            elif cur_flag in ("valid", "val"):
                csv_name = getattr(self.config, "atlas_valid_csv", "valid.csv")
            elif cur_flag in ("test", "holdout"):
                csv_name = getattr(self.config, "atlas_test_csv", "valid.csv")
            else:
                csv_name = f"{cur_flag}.csv"
            return os.path.join(self.base_dir, csv_name)

        # -------------------------------
        # Load CSV
        # -------------------------------
        if flag == "train":
            csv_path = _resolve_csv_path(flag)
            assert os.path.exists(csv_path), f"Missing: {csv_path}"

            df = pd.read_csv(csv_path)
            assert 'Days Post Stroke' in df.columns and 'image_path' in df.columns, \
                f"CSV must have columns: 'Days Post Stroke' and 'image_path'"

            for _, row in df.iterrows():
                label = float(row['Days Post Stroke'])
                img_path = row['image_path']
                self.all_labels.append(label)
                self.all_images.append(img_path)

            # Apply ratio-based subsampling for training
            if train_ratio < 1.0:
                total = len(self.all_images)
                n_keep = max(1, int(total * train_ratio))

                # random but reproducible
                idx = np.random.RandomState(0).choice(total, n_keep, replace=False)

                self.all_images = [self.all_images[i] for i in idx]
                self.all_labels = [self.all_labels[i] for i in idx]

                print(f"[ATLAS] Train ratio={train_ratio}, using {n_keep}/{total} samples")

        else:   # valid / test / holdout
            csv_path = _resolve_csv_path(flag)
            assert os.path.exists(csv_path), f"Missing: {csv_path}"

            df = pd.read_csv(csv_path)
            assert 'Days Post Stroke' in df.columns and 'image_path' in df.columns, \
                f"CSV must have columns: 'Days Post Stroke' and 'image_path'"

            for _, row in df.iterrows():
                self.all_labels.append(float(row['Days Post Stroke']))
                self.all_images.append(row['image_path'])

        assert len(self.all_images) > 0, "No samples found!"

        print(f"[ATLAS] {flag} total loaded: {len(self.all_images)} samples")
        print(f"[ATLAS] {flag} days range: [{min(self.all_labels):.2f}, {max(self.all_labels):.2f}]")

    # -------------------------------------------------------
    def load_nii(self, path):
        # Some ATLAS files have invalid qfac in header; nibabel auto-fixes it.
        # Suppress repeated warning to keep training logs readable.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*pixdim\[0\] \(qfac\) should be 1 \(default\) or -1; setting qfac to 1.*",
                category=UserWarning,
            )
            nii = nib.load(path)
            vol = nii.get_fdata().astype(np.float32)
        vol = np.transpose(vol, (2,0,1))   # (D,H,W)

        # === z-score normalization ===
        vol = zscore_normalize(vol)

        return vol

    def __getitem__(self, index):
        img_path = self.all_images[index]
        label = self.all_labels[index]

        assert os.path.exists(img_path), f"File missing: {img_path}"

        # ---- load 3D MRI: (D,H,W) ----
        vol = self.load_nii(img_path)
        vol = vol.astype(np.float32)

        # ---- Convert to torch tensor: (D,H,W) → (1,1,D,H,W) ----
        vol_t = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)

        # ---- Data Augmentation on 3D volume (before converting to 2D slices) ----
        use_augmentation = getattr(self.config, 'use_augmentation', False)

        if self.flag == "train" and (self.teacher == "brainiac" or use_augmentation):
            # Apply 3D augmentations while data is still in 3D format (1,1,D,H,W)
            affine_transform = RandAffined(
                keys=["image"],
                rotate_range=self.aug_rotate_range,
                translate_range=self.aug_translate_range,
                scale_range=self.aug_scale_range,
                prob=self.aug_affine_prob,
                padding_mode="border",
                mode="trilinear"  # Use trilinear for 3D
            )
            flip_transform = RandFlipd(keys=["image"], spatial_axis=[2], prob=self.aug_flip_prob)
            smooth_transform = RandGaussianSmoothd(keys=["image"], prob=self.aug_smooth_prob)
            noise_transform = RandGaussianNoised(keys=["image"], prob=self.aug_noise_prob, std=self.aug_noise_std)
            contrast_transform = RandAdjustContrastd(
                keys=["image"], prob=self.aug_contrast_prob, gamma=self.aug_contrast_gamma
            )

            # Squeeze to (C, D, H, W) for MONAI transforms
            vol_3d = vol_t.squeeze(0)  # (1, D, H, W)
            data_dict = {"image": vol_3d}
            data_dict = affine_transform(data_dict)
            data_dict = flip_transform(data_dict)
            data_dict = smooth_transform(data_dict)
            data_dict = noise_transform(data_dict)
            data_dict = contrast_transform(data_dict)
            vol_t = data_dict["image"].unsqueeze(0)  # Back to (1, 1, D, H, W)

        # ---- Now resize based on teacher type ----
        if self.teacher == "brainiac":
            # BrainIAC: Direct resize to (96, 96, 96)
            vol_t = torch.nn.functional.interpolate(
                vol_t,
                size=(96, 96, 96),
                mode="trilinear",
                align_corners=False
            )  # → (1,1,96,96,96)
            # Squeeze to (1, 96, 96, 96)
            vol = vol_t.squeeze(0)  # (1, 96, 96, 96)
        else:
            # Med_DINOv3: Convert to 2D slices (128, 224, 224)
            vol_t = torch.nn.functional.interpolate(
                vol_t,
                size=(128, vol.shape[1], vol.shape[2]),
                mode="trilinear",
                align_corners=False
            )  # → (1,1,128,H,W)

            # ---- squeeze back to (128,H,W) ----
            vol_t = vol_t[0, 0]   # (128,H,W)

            # Determine target H,W size
            if self.teacher:
                target_size = 128
            else:
                target_size = 224

            # ---- Resize H,W using bilinear interpolation ----
            vol_t = torch.nn.functional.interpolate(
                vol_t.unsqueeze(0),  # (1,128,H,W)
                size=(target_size, target_size),
                mode="bilinear",
                align_corners=False
            )  # → (1,128,target_size,target_size)

            vol = vol_t  # (1,128,target_size,target_size)

        # ---- label (for regression) ----
        label = torch.tensor(label, dtype=torch.float32)  # Use float32 for regression

        # ---- image name ----
        image_name = os.path.basename(img_path).replace(".nii.gz", "")

        return vol, label, image_name

    def __len__(self):
        return len(self.all_images)


if __name__ == "__main__":
    """
    Test the dataset loader
    """
    class TestConfig:
        n_slices = 128
        use_augmentation = False

    config = TestConfig()
    base_dir = "/path/to/Med_DINOv3/Datasets/Finetune_regression/ATLAS"

    # Test training set
    print("\n" + "="*50)
    print("Testing TRAINING set...")
    print("="*50)
    train_dataset = ATLASRegressionSet(config, base_dir, flag='train')
    print(f"Total training samples: {len(train_dataset)}")

    # Test one sample
    vol, label, name = train_dataset[0]
    print(f"\nSample 0:")
    print(f"  Volume shape: {vol.shape}")
    print(f"  Label: {label} (type: {label.dtype})")
    print(f"  Name: {name}")
    print(f"  Volume range: [{vol.min():.3f}, {vol.max():.3f}]")

    # Test validation set
    print("\n" + "="*50)
    print("Testing VALIDATION set...")
    print("="*50)
    valid_dataset = ATLASRegressionSet(config, base_dir, flag='valid')
    print(f"Total validation samples: {len(valid_dataset)}")

    vol, label, name = valid_dataset[0]
    print(f"\nSample 0:")
    print(f"  Volume shape: {vol.shape}")
    print(f"  Label: {label} (type: {label.dtype})")
    print(f"  Name: {name}")

    print("\n" + "="*50)
    print("Dataset loader test completed!")
    print("="*50)
