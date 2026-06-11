import torch
import numpy as np
import nibabel as nib
import pandas as pd
import os
from torch.utils.data import Dataset


def zscore_normalize(volume, clip_percentile=True):
    """Robust z-score normalization for 3D MRI."""
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


class Survival_upenn_MultiModalSet(Dataset):
    """
    Multi-modality UPENN-GBM survival dataset.

    Returns 4 modalities (T1, T1CE, T2, FLAIR) for each patient.
    Each modality is processed independently and stacked as (4, 128, 224, 224).

    Key feature: Infers paths to all 4 modalities from a single T1 path.
    Example: /path/UPENN-GBM-00001.nii.gz →
             /path/UPENN-GBM-00001_T1.nii.gz
             /path/UPENN-GBM-00001_T1CE.nii.gz
             /path/UPENN-GBM-00001_T2.nii.gz
             /path/UPENN-GBM-00001_FLAIR.nii.gz
    """

    def __init__(self, config, base_dir, flag='train', fold=0):
        super().__init__()
        self.config = config
        self.flag = flag

        # Multi-modality data directory
        self.multimodal_dir = "/path/to/Med_DINOv3/Datasets/Finetune_classification/Survival_multi_modality/gbm_multimodal_dataset"

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

        # Build sample list with all 4 modality paths
        self.samples = []
        for single_modal_path in split_paths:
            patient_id = self._extract_patient_id(single_modal_path)

            # Construct paths for all 4 modalities
            modality_paths = {
                't1': os.path.join(self.multimodal_dir, f"{patient_id}_T1.nii.gz"),
                't1ce': os.path.join(self.multimodal_dir, f"{patient_id}_T1CE.nii.gz"),
                't2': os.path.join(self.multimodal_dir, f"{patient_id}_T2.nii.gz"),
                'flair': os.path.join(self.multimodal_dir, f"{patient_id}_FLAIR.nii.gz"),
            }

            # Verify all 4 modalities exist
            if all(os.path.exists(p) for p in modality_paths.values()):
                self.samples.append({
                    'patient_id': patient_id,
                    **modality_paths
                })
            else:
                missing = [k for k, v in modality_paths.items() if not os.path.exists(v)]
                print(f"[WARNING] Skipping {patient_id}: missing {missing}")

        print(f"[MultiModalDataset] {flag} loaded: {len(self.samples)} patients")

    def _load_split_csvs(self, csv_paths):
        """Load file paths from split CSVs"""
        paths = []
        for csv_path in csv_paths:
            df = pd.read_csv(csv_path, header=None)
            paths.extend(df.iloc[:, 1].tolist())
        return paths

    def _extract_patient_id(self, path_or_id):
        """Extract patient ID prefix"""
        basename = os.path.basename(path_or_id).replace(".nii.gz", "")
        if '_' in basename:
            return basename.split('_')[0]
        return basename

    def load_nii(self, path):
        """Load and preprocess .nii.gz file"""
        nii = nib.load(path)
        vol = nii.get_fdata().astype(np.float32)
        vol = np.transpose(vol, (2, 0, 1))  # (D, H, W)
        vol = zscore_normalize(vol)
        return vol

    def preprocess_volume(self, vol):
        """Resize volume to (128, 224, 224)"""
        vol_t = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)

        # Resize depth to 128
        vol_t = torch.nn.functional.interpolate(
            vol_t,
            size=(128, vol.shape[1], vol.shape[2]),
            mode='trilinear',
            align_corners=False
        )

        vol_t = vol_t[0, 0]  # (128, H, W)

        # Resize H, W to 224
        vol_t = torch.nn.functional.interpolate(
            vol_t.unsqueeze(0),
            size=(224, 224),
            mode='bilinear',
            align_corners=False
        )

        return vol_t[0]  # (128, 224, 224)

    def __getitem__(self, index):
        sample = self.samples[index]

        patient_id = sample['patient_id']

        # Load all 4 modalities
        modalities = []
        for modality_name in ['t1', 't1ce', 't2', 'flair']:
            path = sample[modality_name]

            assert os.path.exists(path), f"Missing file: {path}"

            vol = self.load_nii(path)
            vol = self.preprocess_volume(vol)  # (128, 224, 224)
            modalities.append(vol)

        # Stack modalities: (4, 128, 224, 224)
        multi_modal_volume = torch.stack(modalities, dim=0)

        # Get label (placeholder - will be replaced by survival labels)
        # For now, use 0 as placeholder
        label = torch.tensor(0, dtype=torch.long)

        return multi_modal_volume, label, patient_id

    def __len__(self):
        return len(self.samples)
