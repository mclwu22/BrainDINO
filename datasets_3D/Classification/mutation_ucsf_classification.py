"""
UCSF-PDGM Mutation Classification Dataset (Multi-Modality)

Binary classification: IDH wildtype (0) vs non-wildtype (1)
Uses 2 modalities: T1c and FLAIR

Format similar to ABIDE (two_csv format):
- train_0.csv: wildtype samples (label 0)
- train_1.csv: non-wildtype samples (label 1)
- valid.csv: validation samples
"""

import torch
import numpy as np
import nibabel as nib
import os
import csv
import pandas as pd
from torch.utils.data import Dataset
import torch.nn.functional as F


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

    # Step 1: optional percentile clipping
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

    # Step 2: compute robust mean and std from non-zero region
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

    # Step 3: z-score normalize
    vol = (vol - mean) / std

    return vol.astype(np.float32)


class MutationUCSFClassificationSet(Dataset):
    """
    UCSF-PDGM Mutation Classification Dataset (Multi-Modality)

    Loads T1c and FLAIR modalities for each patient.
    Binary classification: IDH wildtype (0) vs non-wildtype (1)

    Similar to ABIDE format:
    - train_0.csv: wildtype patients
    - train_1.csv: non-wildtype patients
    - valid.csv: validation patients
    """

    def __init__(self, config, base_dir, flag='train', train_ratio=1.0, teacher=None):
        super().__init__()
        self.config = config
        self.flag = flag
        self.n_slices = config.n_slices
        self.train_ratio = train_ratio

        self.base_dir = base_dir

        # Storage for samples
        self.samples = []

        # Load CSV(s) based on flag
        if flag == "train":
            csv0 = os.path.join(self.base_dir, "train_0.csv")
            csv1 = os.path.join(self.base_dir, "train_1.csv")
            assert os.path.exists(csv0), f"Missing: {csv0}"
            assert os.path.exists(csv1), f"Missing: {csv1}"

            # Load both classes
            self._load_from_csv(csv0)
            self._load_from_csv(csv1)

        else:  # valid or test
            csv_path = os.path.join(self.base_dir, f"{flag}.csv")
            assert os.path.exists(csv_path), f"Missing: {csv_path}"

            self._load_from_csv(csv_path)

        # Apply train_ratio subsampling
        if flag == "train" and train_ratio < 1.0:
            original_size = len(self.samples)
            np.random.seed(42)
            n_samples = int(len(self.samples) * train_ratio)
            indices = np.random.choice(len(self.samples), n_samples, replace=False)
            self.samples = [self.samples[i] for i in sorted(indices)]
            print(f"  ⚠️  Train ratio {train_ratio:.1f}: {original_size} -> {len(self.samples)} samples")

        assert len(self.samples) > 0, "No samples found!"

        # Print statistics
        labels = [s['label'] for s in self.samples]
        print(f"[UCSF-Mutation] {flag} loaded: {len(self.samples)} samples")
        print(f"  Label distribution: {np.bincount(labels)}")
        print(f"  Label 0 (wildtype): {labels.count(0)}")
        print(f"  Label 1 (non-wildtype): {labels.count(1)}")

    def _load_from_csv(self, csv_path):
        """
        Load samples from CSV file.

        CSV format: label, t1c_path

        For each row, we:
        1. Extract patient ID from t1c_path
        2. Construct FLAIR path
        3. Verify both files exist
        4. Add to samples list
        """
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                label = int(row[0])
                t1c_path = row[1]

                # Extract patient ID from t1c path
                # Example: /path/UCSF-PDGM-0004_T1c.nii.gz -> UCSF-PDGM-0004
                basename = os.path.basename(t1c_path)
                patient_id = basename.replace('_T1c.nii.gz', '').replace('_T1CE.nii.gz', '')

                # Construct FLAIR path (same directory as T1c)
                data_dir = os.path.dirname(t1c_path)
                flair_path = os.path.join(data_dir, f"{patient_id}_FLAIR.nii.gz")

                # Verify both modalities exist
                if not os.path.exists(t1c_path):
                    print(f"[WARNING] Missing T1c: {t1c_path}")
                    continue

                if not os.path.exists(flair_path):
                    print(f"[WARNING] Missing FLAIR: {flair_path}")
                    continue

                # Add sample
                self.samples.append({
                    'patient_id': patient_id,
                    'label': label,
                    't1c_path': t1c_path,
                    'flair_path': flair_path
                })

    def load_nii(self, path):
        """
        Load NIfTI file and return normalized volume.

        Returns:
            vol: (D, H, W) normalized volume
        """
        nii = nib.load(path)
        vol = nii.get_fdata().astype(np.float32)

        # Ensure (D, H, W) orientation
        if vol.ndim == 3:
            # Common cases: (H,W,D) or (D,H,W)
            # If third dim is smallest, it's likely depth
            if vol.shape[2] < min(vol.shape[0], vol.shape[1]):
                # (H,W,D) -> (D,H,W)
                vol = np.transpose(vol, (2, 0, 1))

        # Z-score normalization
        vol = zscore_normalize(vol)

        return vol

    def preprocess_volume(self, vol):
        """
        Preprocess volume to target size: (128, 224, 224)

        Args:
            vol: (D, H, W) numpy array

        Returns:
            vol_t: (128, 224, 224) tensor
        """
        # Convert to tensor and add batch/channel dims
        vol_t = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)

        # Resize depth to 128 using trilinear interpolation
        vol_t = F.interpolate(
            vol_t,
            size=(128, vol_t.shape[3], vol_t.shape[4]),
            mode="trilinear",
            align_corners=False
        )

        # Resize spatial to 224x224
        vol_t = F.interpolate(
            vol_t,
            size=(vol_t.shape[2], 224, 224),
            mode="trilinear",
            align_corners=False
        )

        # Remove batch and channel dims: (1, 1, 128, 224, 224) -> (128, 224, 224)
        return vol_t[0, 0]

    def __getitem__(self, index):
        """
        Returns:
            multi_modal_volume: (2, 128, 224, 224) tensor with T1c and FLAIR
            label: 0 (wildtype) or 1 (non-wildtype)
            patient_id: string identifier
        """
        sample = self.samples[index]

        patient_id = sample['patient_id']
        label = sample['label']
        t1c_path = sample['t1c_path']
        flair_path = sample['flair_path']

        # Load and preprocess both modalities
        t1c_vol = self.load_nii(t1c_path)
        flair_vol = self.load_nii(flair_path)

        t1c_preprocessed = self.preprocess_volume(t1c_vol)       # (128, 224, 224)
        flair_preprocessed = self.preprocess_volume(flair_vol)   # (128, 224, 224)

        # Stack modalities: (2, 128, 224, 224)
        multi_modal_volume = torch.stack([t1c_preprocessed, flair_preprocessed], dim=0)

        # Label as tensor
        label = torch.tensor(label, dtype=torch.long)

        return multi_modal_volume, label, patient_id

    def __len__(self):
        return len(self.samples)
