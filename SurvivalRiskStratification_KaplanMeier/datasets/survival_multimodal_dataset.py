"""
Multi-modality survival dataset with event status and survival time.

Combines:
1. Clinical CSV (survival time + event status)
2. Multi-modality image paths (4 modalities per patient)
3. Binary label generation (1-year mortality prediction)

Returns: (multi_modal_volume, label, survival_time, event, patient_id)
- multi_modal_volume: (4, 128, 224, 224) tensor
- label: 0/1 binary classification label
- survival_time: days from surgery
- event: 0 (censored) / 1 (deceased)
- patient_id: patient identifier
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
import torch.nn.functional as F


class SurvivalMultiModalDataset(Dataset):
    """
    Multi-modality UPENN-GBM survival dataset with survival labels.
    """

    def __init__(self, config, base_dir, flag='train', fold=0, train_ratio=1.0):
        super().__init__()
        self.config = config
        self.flag = flag
        self.risk_threshold_days = getattr(config, 'risk_threshold_days', 365)
        self.train_ratio = train_ratio

        # Multi-modality data directory
        self.multimodal_dir = "/path/to/Med_DINOv3/Datasets/Finetune_classification/Survival_multi_modality/gbm_multimodal_dataset"

        # Clinical CSV path
        self.clinical_csv = os.path.join(
            base_dir,
            "UPENN-GBM_clinical_info_v2.1 (1).csv"
        )

        # Load clinical data
        self.clinical_df = pd.read_csv(self.clinical_csv)
        print(f"[SurvivalMultiModal] Loaded clinical data: {len(self.clinical_df)} patients")

        # Load existing single-modality split CSVs
        if flag == "train":
            csv0 = os.path.join(base_dir, "train_0.csv")
            csv1 = os.path.join(base_dir, "train_1.csv")
            split_paths = self._load_split_csvs([csv0, csv1])
        elif flag == "valid":
            csv_path = os.path.join(base_dir, "valid.csv")
            split_paths = self._load_split_csvs([csv_path])
        else:
            raise ValueError(f"Unknown flag: {flag}")

        # Build sample list with clinical labels
        self.samples = self._process_samples(split_paths)

        # Apply train_ratio subsampling for training set
        if flag == "train" and train_ratio < 1.0:
            original_size = len(self.samples)
            np.random.seed(42)  # Fixed seed for reproducibility
            n_samples = int(len(self.samples) * train_ratio)
            indices = np.random.choice(len(self.samples), n_samples, replace=False)
            self.samples = [self.samples[i] for i in sorted(indices)]
            print(f"  ⚠️  Train ratio {train_ratio:.1f}: {original_size} -> {len(self.samples)} samples")

        print(f"[SurvivalMultiModal] {flag} loaded: {len(self.samples)} patients")
        if len(self.samples) > 0:
            labels = [s['label'] for s in self.samples]
            print(f"  Label distribution: {np.bincount(labels)}")

    def _load_split_csvs(self, csv_paths):
        """Load file paths from split CSVs"""
        paths = []
        for csv_path in csv_paths:
            df = pd.read_csv(csv_path, header=None)
            paths.extend(df.iloc[:, 1].tolist())
        return paths

    def _extract_patient_id(self, path):
        """
        Extract patient ID from path.
        Example: /path/UPENN-GBM-00001_11.nii.gz -> UPENN-GBM-00001
        """
        basename = os.path.basename(path).replace('.nii.gz', '')
        if '_' in basename:
            return basename.split('_')[0]
        return basename

    def _process_samples(self, split_paths):
        """
        Build sample list with:
        1. All 4 modality paths
        2. Survival time
        3. Event status
        4. Binary label
        """
        samples = []

        for single_modal_path in split_paths:
            patient_id = self._extract_patient_id(single_modal_path)

            # Get clinical info
            # Clinical CSV has IDs with suffix (UPENN-GBM-00001_11)
            # Try exact match first, then match with suffix
            clinical_row = self.clinical_df[self.clinical_df['ID'] == patient_id]

            if len(clinical_row) == 0:
                # Try matching with _11 suffix
                clinical_row = self.clinical_df[self.clinical_df['ID'] == f"{patient_id}_11"]

            if len(clinical_row) == 0:
                # Try matching as prefix (clinical ID starts with patient_id)
                clinical_row = self.clinical_df[self.clinical_df['ID'].str.startswith(patient_id)]

            if len(clinical_row) == 0:
                # Skip silently (many patients in split don't have clinical data)
                continue

            clinical_row = clinical_row.iloc[0]

            # Parse survival info
            survival_time = clinical_row['Survival_from_surgery_days_UPDATED']
            survival_status = clinical_row['Survival_Status']

            # Skip invalid entries
            if pd.isna(survival_time) or survival_status == "Not Available":
                continue

            # Convert survival time to float
            try:
                survival_time = float(survival_time)
            except (ValueError, TypeError):
                continue

            # Parse event status
            if survival_status == "Deceased":
                event = 1
            elif survival_status == "Lost to Follow-up":
                event = 0
                # Use censored time if available
                if 'Survival_Censor' in clinical_row and not pd.isna(clinical_row['Survival_Censor']):
                    try:
                        survival_time = float(clinical_row['Survival_Censor'])
                    except (ValueError, TypeError):
                        pass  # Keep original survival_time
            else:
                continue

            # Generate binary label for 1-year survival prediction
            if event == 1 and survival_time <= self.risk_threshold_days:
                label = 1  # high risk (early death)
            elif event == 1 and survival_time > self.risk_threshold_days:
                label = 0  # low risk (survived > 1 year)
            elif event == 0 and survival_time > self.risk_threshold_days:
                label = 0  # censored but survived > 1 year
            else:
                # Discard: censored before 1 year (ambiguous)
                continue

            # Construct paths for all 4 modalities
            modality_paths = {
                't1': os.path.join(self.multimodal_dir, f"{patient_id}_T1.nii.gz"),
                't1ce': os.path.join(self.multimodal_dir, f"{patient_id}_T1CE.nii.gz"),
                't2': os.path.join(self.multimodal_dir, f"{patient_id}_T2.nii.gz"),
                'flair': os.path.join(self.multimodal_dir, f"{patient_id}_FLAIR.nii.gz"),
            }

            # Verify all 4 modalities exist
            if not all(os.path.exists(p) for p in modality_paths.values()):
                missing = [k for k, v in modality_paths.items() if not os.path.exists(v)]
                print(f"[WARNING] Skipping {patient_id}: missing modalities {missing}")
                continue

            samples.append({
                'patient_id': patient_id,
                'label': int(label),
                'survival_time': float(survival_time),
                'event': int(event),
                **modality_paths
            })

        return samples

    def load_nii(self, path):
        """Load NIfTI file and return numpy array"""
        img = nib.load(path)
        vol = img.get_fdata()
        return vol.astype(np.float32)

    def preprocess_volume(self, vol):
        """
        Preprocess single modality volume:
        1. Z-score normalization
        2. Resize to (128, 224, 224)
        """
        # Z-score normalization
        mean = vol.mean()
        std = vol.std()
        if std > 0:
            vol = (vol - mean) / std
        else:
            vol = vol - mean

        # Convert to tensor and add channel dim
        vol_t = torch.from_numpy(vol).float().unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)

        # Resize depth to 128
        vol_t = F.interpolate(
            vol_t,
            size=(128, vol_t.shape[3], vol_t.shape[4]),
            mode='trilinear',
            align_corners=False
        )

        # Resize spatial to 224x224
        vol_t = F.interpolate(
            vol_t,
            size=(vol_t.shape[2], 224, 224),
            mode='trilinear',
            align_corners=False
        )

        return vol_t[0, 0]  # (128, 224, 224)

    def __getitem__(self, index):
        sample = self.samples[index]

        patient_id = sample['patient_id']
        label = sample['label']
        survival_time = sample['survival_time']
        event = sample['event']

        # Load all 4 modalities
        modalities = []
        for modality_name in ['t1', 't1ce', 't2', 'flair']:
            path = sample[modality_name]

            vol = self.load_nii(path)
            vol = self.preprocess_volume(vol)  # (128, 224, 224)
            modalities.append(vol)

        # Stack modalities: (4, 128, 224, 224)
        multi_modal_volume = torch.stack(modalities, dim=0)

        return multi_modal_volume, label, survival_time, event, patient_id

    def __len__(self):
        return len(self.samples)
