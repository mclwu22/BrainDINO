import torch
import numpy as np
import nibabel as nib
import os
import csv
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

class BraTS_2023_ClassificationSet(Dataset):
    """
    BraTS-2023 multi-modality MRI classification dataset (for modality classification)

    This dataset is designed for 4-class classification where each class corresponds 
    to a specific MRI modality extracted from nnUNet-style raw BraTS data:

        Class 0: T1c   (file suffix: *0000.nii.gz)
        Class 1: T1n   (file suffix: *0001.nii.gz)
        Class 2: FLAIR (file suffix: *0002.nii.gz)
        Class 3: T2w   (file suffix: *0003.nii.gz)

    Expected input CSV files:
        - train_0.csv
        - train_1.csv
        - train_2.csv
        - train_3.csv
    """
    def __init__(self, config, base_dir, flag='train', train_ratio=1.0, teacher=None):
        super().__init__()
        self.config = config
        self.flag = flag
        self.base_dir = base_dir
        self.teacher = teacher  # Store teacher name for resize decision
        self.n_slices = config.n_slices if hasattr(config, 'n_slices') else 128

        self.all_images = []
        self.all_labels = []

        # -------------------------------
        # Load CSV(s)
        # -------------------------------
        if flag == "train":
            # Load all 3 classes for training
            csv_files = [
                (os.path.join(self.base_dir, "train_0.csv"), 0),
                (os.path.join(self.base_dir, "train_1.csv"), 1),
                (os.path.join(self.base_dir, "train_2.csv"), 2),
                (os.path.join(self.base_dir, "train_3.csv"), 3)
            ]

            for csv_path, expected_label in csv_files:
                assert os.path.exists(csv_path), f"Missing: {csv_path}"

                with open(csv_path, "r") as f:
                    reader = csv.reader(f)
                    for row in reader:
                        label = int(row[0])
                        img_path = row[1]
                        # Verify label consistency
                        assert label == expected_label, \
                            f"Label mismatch in {csv_path}: expected {expected_label}, got {label}"
                        self.all_labels.append(label)
                        self.all_images.append(img_path)

                print(f"[ADNI] Loaded {csv_path}: class {expected_label}, "
                      f"{sum(1 for l in self.all_labels if l == expected_label)} samples")

            if train_ratio < 1.0:
                total = len(self.all_images)
                n_keep = max(1, int(total * train_ratio))
                idx = np.random.RandomState(0).choice(total, n_keep, replace=False)
                self.all_images = [self.all_images[i] for i in idx]
                self.all_labels = [self.all_labels[i] for i in idx]
                print(f"[BraTS] Train ratio={train_ratio}, using {n_keep}/{total} samples")

        else:   # valid or test
            csv_path = os.path.join(self.base_dir, f"{flag}.csv")
            assert os.path.exists(csv_path), f"Missing: {csv_path}"

            with open(csv_path, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    self.all_labels.append(int(row[0]))
                    self.all_images.append(row[1])

            # Print class distribution for validation
            if flag == "valid":
                from collections import Counter
                class_counts = Counter(self.all_labels)
                print(f"[ADNI] {flag} class distribution: {dict(class_counts)}")

        assert len(self.all_images) > 0, "No samples found!"
        print(f"[ADNI] {flag} total loaded: {len(self.all_images)} samples")

    # -------------------------------------------------------
    def sample_slices(self, vol, n_slices):
        """Uniformly sample n_slices from the depth dimension"""
        D = vol.shape[0]
        idx = np.linspace(0, D - 1, n_slices).astype(np.int32)
        return vol[idx, :, :]
    import numpy as np


    # -------------------------------------------------------
    def load_nii(self, path):
        nii = nib.load(path)
        vol = nii.get_fdata().astype(np.float32)
        vol = np.transpose(vol, (2,0,1))   # (D,H,W)

        # === ADD z-score normalization here ===
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
                rotate_range=(0.1, 0.1, 0.1),
                translate_range=(5, 5, 5),
                scale_range=(0.1, 0.1, 0.1),
                prob=0.5,
                padding_mode="border",
                mode="trilinear"  # Use trilinear for 3D
            )
            flip_transform = RandFlipd(keys=["image"], spatial_axis=[2], prob=0.5)
            smooth_transform = RandGaussianSmoothd(keys=["image"], prob=0.2)
            noise_transform = RandGaussianNoised(keys=["image"], prob=0.2, std=0.05)
            contrast_transform = RandAdjustContrastd(keys=["image"], prob=0.2, gamma=(0.7, 1.3))

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

            # ---- Resize H,W to 224 using bilinear interpolation ----
            vol_t = torch.nn.functional.interpolate(
                vol_t.unsqueeze(0),  # (1,128,H,W)
                size=(224, 224),
                mode="bilinear",
                align_corners=False
            )  # → (1,128,224,224)

            vol = vol_t  # (1,128,224,224)

        # ---- label (for 3-class classification) ----
        label = torch.tensor(label, dtype=torch.long)  # Use long for CrossEntropyLoss

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

    config = TestConfig()
    base_dir = "/path/to/Med_DINOv3/Datasets/Finetune_classification/ADNI"

    # Test training set
    print("\n" + "="*50)
    print("Testing TRAINING set...")
    print("="*50)
    train_dataset = BraTS_2023_ClassificationSet(config, base_dir, flag='train')
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
    valid_dataset = BraTS_2023_ClassificationSet(config, base_dir, flag='valid')
    print(f"Total validation samples: {len(valid_dataset)}")

    vol, label, name = valid_dataset[0]
    print(f"\nSample 0:")
    print(f"  Volume shape: {vol.shape}")
    print(f"  Label: {label} (type: {label.dtype})")
    print(f"  Name: {name}")

    print("\n" + "="*50)
    print("Dataset loader test completed!")
    print("="*50)
