import torch
import numpy as np
import nibabel as nib
import os
import csv
from torch.utils.data import Dataset
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

class BrainAgeRegressionSet(Dataset):
    """
    ADNI dataset loader for 3-class classification
    Supports train_0.csv + train_1.csv + train_2.csv for training
    Supports valid.csv for validation
    Supports .nii.gz with uniform slice sampling

    Classes:
        0: CN (Cognitive Normal)
        1: MCI (Mild Cognitive Impairment)
        2: AD (Alzheimer's Disease)
    """
    def __init__(self, config, base_dir, flag='train'):
        super().__init__()
        self.config = config
        self.flag = flag
        self.base_dir = base_dir
        self.n_slices = config.n_slices if hasattr(config, 'n_slices') else 128

        self.all_images = []
        self.all_labels = []

        # -------------------------------
        # Load CSV(s)
        # -------------------------------
        if flag == "train":
            csv_path = os.path.join(self.base_dir, "train.csv")
            assert os.path.exists(csv_path), f"Missing: {csv_path}"
            with open(csv_path, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    label = float(row[0])
                    img_path = row[1]
                    # Verify label consistency
                    assert label == expected_label, \
                        f"Label mismatch in {csv_path}: expected {expected_label}, got {label}"
                    self.all_labels.append(label)
                    self.all_images.append(img_path)

            print(f"[Reg] Loaded {csv_path}: class {expected_label}, "
                    f"{sum(1 for l in self.all_labels if l == expected_label)} samples")

        else:   # valid or test
            csv_path = os.path.join(self.base_dir, f"{flag}.csv")
            assert os.path.exists(csv_path), f"Missing: {csv_path}"

            with open(csv_path, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    self.all_labels.append(float(row[0]))
                    self.all_images.append(row[1])

            # Print class distribution for validation
            if flag == "valid":
                from collections import Counter
                class_counts = Counter(self.all_labels)
                print(f"[Reg] {flag} class distribution: {dict(class_counts)}")

        assert len(self.all_images) > 0, "No samples found!"
        print(f"[Reg] {flag} total loaded: {len(self.all_images)} samples")
        print(
            "[DEBUG]",
            min(self.all_labels),
            max(self.all_labels),
            type(self.all_labels[0])
        )

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

        # ---- Resize depth to 128 using trilinear interpolation ----
        # vol: (D,H,W) → torch: (1,1,D,H,W)
        vol_t = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
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

        # ---- Final output format (1,128,224,224) ----
        vol = vol_t   # already correct shape

        # ---- label (for 3-class classification) ----
        label = torch.tensor(label, dtype=torch.float32)  # Use long for CrossEntropyLoss

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
    train_dataset = ADNIClassificationSet(config, base_dir, flag='train')
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
    valid_dataset = ADNIClassificationSet(config, base_dir, flag='valid')
    print(f"Total validation samples: {len(valid_dataset)}")

    vol, label, name = valid_dataset[0]
    print(f"\nSample 0:")
    print(f"  Volume shape: {vol.shape}")
    print(f"  Label: {label} (type: {label.dtype})")
    print(f"  Name: {name}")

    print("\n" + "="*50)
    print("Dataset loader test completed!")
    print("="*50)
