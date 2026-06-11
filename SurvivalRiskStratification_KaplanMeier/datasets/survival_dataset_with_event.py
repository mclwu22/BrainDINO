import torch
import numpy as np
import nibabel as nib
import pandas as pd
import os
from torch.utils.data import Dataset


def zscore_normalize(volume, clip_percentile=True):
    """
    Robust z-score normalization for 3D MRI.
    """
    vol = volume.astype(np.float32)

    if clip_percentile:
        nonzero = vol[vol > 0]
        if nonzero.size < 10:
            nonzero = vol.reshape(-1)

        lo = np.percentile(nonzero, 1)
        hi = np.percentile(nonzero, 99)
        vol = np.clip(vol, lo, hi)

    nz = vol[vol > 0]
    if nz.size > 0:
        mean = nz.mean()
        std = nz.std()
    else:
        mean = vol.mean()
        std = vol.std()

    std = max(std, 1e-6)
    vol = (vol - mean) / std
    return vol.astype(np.float32)


class SurvivalDatasetWithEvent(Dataset):
    """
    UPENN-GBM survival dataset with event handling for risk stratification.

    Loads:
    1. Clinical info (with Survival_Status and survival times)
    2. Image paths
    3. Train/val split from existing CSVs

    Returns:
    - volume: (1, 128, 224, 224) tensor
    - binary_label: 0 (low risk) or 1 (high risk, death ≤ 365 days)
    - survival_time: days from surgery
    - event: 1 (deceased) or 0 (censored)
    - patient_id: string identifier
    """

    def __init__(self, config, flag='train', train_ratio=1.0):
        super().__init__()
        self.config = config
        self.flag = flag
        self.n_slices = config.n_slices
        self.risk_threshold_days = config.risk_threshold_days

        # Load clinical info
        clinical_path = os.path.join(config.data_root, config.clinical_csv)
        self.clinical_df = pd.read_csv(clinical_path)

        # Load image paths
        image_csv_path = os.path.join(config.data_root, config.image_csv)
        self.image_df = pd.read_csv(image_csv_path)

        # Load train/val split
        if flag == 'train':
            csv0 = os.path.join(config.data_root, config.train_csv_0)
            csv1 = os.path.join(config.data_root, config.train_csv_1)
            split_paths = self._load_split_csvs([csv0, csv1])
        elif flag == 'valid':
            csv_path = os.path.join(config.data_root, config.valid_csv)
            split_paths = self._load_split_csvs([csv_path])
        else:
            raise ValueError(f"Unknown flag: {flag}")

        # Process data
        self.samples = self._process_samples(split_paths)

        # Apply train_ratio if training
        if flag == 'train' and train_ratio < 1.0:
            n_samples = int(len(self.samples) * train_ratio)
            self.samples = self.samples[:n_samples]

        print(f"[SurvivalDataset] {flag} loaded: {len(self.samples)} samples")
        print(f"  - High risk (label=1): {sum(s['label'] == 1 for s in self.samples)}")
        print(f"  - Low risk (label=0): {sum(s['label'] == 0 for s in self.samples)}")

    def _load_split_csvs(self, csv_paths):
        """Load file paths from split CSVs (train_0.csv, train_1.csv, valid.csv)"""
        paths = []
        for csv_path in csv_paths:
            df = pd.read_csv(csv_path, header=None)
            paths.extend(df.iloc[:, 1].tolist())  # Column 1 is file path
        return paths

    def _extract_patient_id(self, path_or_id):
        """Extract patient ID prefix (e.g., UPENN-GBM-00001_11 -> UPENN-GBM-00001)"""
        basename = os.path.basename(path_or_id).replace('.nii.gz', '')
        # Extract UPENN-GBM-XXXXX part (before underscore if exists)
        if '_' in basename:
            return basename.split('_')[0]
        return basename

    def _process_samples(self, split_paths):
        """
        Merge clinical info with image paths and generate labels.

        Label generation logic:
        - event=1 & survival_time <= 365 → label=1 (high risk, early death)
        - event=1 & survival_time > 365 → label=0 (low risk, survived > 1 year)
        - event=0 & survival_time > 365 → label=0 (censored, survived > 1 year)
        - event=0 & survival_time <= 365 → DISCARD (censored before 1 year, ambiguous)
        """
        samples = []

        def parse_survival_time(value):
            if pd.isna(value) or value == 'Not Available':
                return None
            return float(value)

        for img_path in split_paths:
            patient_id = self._extract_patient_id(img_path)

            # Find matching image info
            img_row = self.image_df[self.image_df['patient_id'] == patient_id]
            if img_row.empty:
                print(f"[WARNING] No image info for {patient_id}, skipping")
                continue

            # Find matching clinical info (may have suffix like _11, _21)
            clinical_matches = self.clinical_df[
                self.clinical_df['ID'].str.startswith(patient_id)
            ]

            if clinical_matches.empty:
                print(f"[WARNING] No clinical info for {patient_id}, skipping")
                continue

            # Use first match (baseline scan typically has _11 suffix)
            clinical_row = clinical_matches.iloc[0]

            # Parse survival time
            survival_time_raw = clinical_row['Survival_from_surgery_days_UPDATED']
            survival_status = clinical_row['Survival_Status']
            survival_censor = clinical_row['Survival_Censor']

            if survival_status == 'Not Available':
                print(f"[WARNING] Survival status not available for {patient_id}, skipping")
                continue

            # Parse event status
            if survival_status == 'Deceased':
                event = 1
                survival_time = parse_survival_time(survival_time_raw)
                if survival_time is None:
                    survival_time = parse_survival_time(survival_censor)
            elif survival_status == 'Lost to Follow-up':
                event = 0  # censored
                survival_time = parse_survival_time(survival_censor)
            else:
                print(f"[WARNING] Unknown survival status '{survival_status}' for {patient_id}, skipping")
                continue

            if survival_time is None:
                print(f"[WARNING] No survival time for {patient_id}, skipping")
                continue

            # Generate binary label
            if event == 1 and survival_time <= self.risk_threshold_days:
                label = 1  # high risk (early death)
            elif event == 1 and survival_time > self.risk_threshold_days:
                label = 0  # low risk (survived > 1 year)
            elif event == 0 and survival_time > self.risk_threshold_days:
                label = 0  # censored but survived > 1 year
            else:
                # event == 0 and survival_time <= threshold: censored before 1 year, discard
                print(f"[INFO] Discarding {patient_id}: censored at {survival_time} days (< {self.risk_threshold_days})")
                continue

            samples.append({
                'image_path': img_path,
                'patient_id': patient_id,
                'label': label,
                'survival_time': survival_time,
                'event': event
            })

        return samples

    def load_nii(self, path):
        """Load and preprocess .nii.gz file"""
        nii = nib.load(path)
        vol = nii.get_fdata().astype(np.float32)
        vol = np.transpose(vol, (2, 0, 1))  # (D, H, W)
        vol = zscore_normalize(vol)
        return vol

    def __getitem__(self, index):
        sample = self.samples[index]

        img_path = sample['image_path']
        label = sample['label']
        survival_time = sample['survival_time']
        event = sample['event']
        patient_id = sample['patient_id']

        # Load 3D MRI
        vol = self.load_nii(img_path)

        # Resize depth to 128 using trilinear interpolation
        vol_t = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
        vol_t = torch.nn.functional.interpolate(
            vol_t,
            size=(128, vol.shape[1], vol.shape[2]),
            mode='trilinear',
            align_corners=False
        )  # (1, 1, 128, H, W)

        vol_t = vol_t[0, 0]  # (128, H, W)

        # Resize H, W to 224 using bilinear interpolation
        vol_t = torch.nn.functional.interpolate(
            vol_t.unsqueeze(0),  # (1, 128, H, W)
            size=(224, 224),
            mode='bilinear',
            align_corners=False
        )  # (1, 128, 224, 224)

        # Final output
        vol = vol_t  # (1, 128, 224, 224)
        label = torch.tensor(label, dtype=torch.long)
        survival_time = torch.tensor(survival_time, dtype=torch.float32)
        event = torch.tensor(event, dtype=torch.long)

        return vol, label, survival_time, event, patient_id

    def __len__(self):
        return len(self.samples)
