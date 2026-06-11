import os
import io
import json
import sys
import glob
import random
import numpy as np
import nibabel as nib
import lmdb
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from nibabel.orientations import axcodes2ornt, aff2axcodes, io_orientation, inv_ornt_aff
from typing import Dict, List, Optional

# Add DUNE path for importing teacher preprocessing
DUNE_PATH = "/scr2/yz_wu/foundation/dune"
if DUNE_PATH not in sys.path:
    sys.path.insert(0, DUNE_PATH)

# --------------------------
# Teacher preprocessing configurations (from dune)
# --------------------------
TEACHER_PREPROCESSING_CONFIGS = {
    "bm_mae": {
        "target_size": (128, 128, 128),
        "normalization": "zscore",
        "channels": 1,
        "feature_dim": 768,
    },
    "brainmvp": {
        "target_size": (128, 128, 128),
        "normalization": "zscore",  # No normalization for brainmvp
        "channels": 1,
        "multi_view": True,
        "num_views": 2,
        "feature_dim": 1024,
    },
}

def open_single_lmdb(lmdb_path, key=None):
    env = lmdb.open(lmdb_path, readonly=True, lock=False, subdir=False, readahead=False)
    with env.begin(write=False) as txn:
        index_raw = txn.get(b"__index__")
        keys = [k for k in index_raw.decode().strip().split("\n") if len(k) > 0]

        if key is None:
            key = np.random.choice(keys)

        npy_key = f"{key}.npy".encode()
        json_key = f"{key}.json".encode()

        npy_data = txn.get(npy_key)
        if npy_data is None:
            raise ValueError(f"❌ 在 {lmdb_path} 中找不到 {npy_key}")

        volume = np.load(io.BytesIO(npy_data)).astype(np.float32)
        meta_data = txn.get(json_key)
        meta = json.loads(meta_data.decode()) if meta_data is not None else {}

    env.close()
    return volume, meta

# --------------------------
# 你已有的函数，可以直接 import
# --------------------------
# from utils import select_slices
# 或者这里直接定义个简单版本：
def select_slices(arr, n_slices=8, mode="uniform", scale=1):
    """
    根据指定模式选择 n_slices 张切片。

    参数：
        arr (np.ndarray): shape (D, H, W)
        n_slices (int): 要选的切片数
        mode (str):
            "uniform" → 在非空范围内均匀抽取
            "normal"  → 在中间位置附近随机抽样（高斯分布）
            "all"     → 返回所有非空切片索引
    返回：
        list[int]: 有效切片索引
    """
    D = arr.shape[0]
    mask = (arr != 0).any(axis=(1, 2))
    if not mask.any():
        # fallback: 全部 slice 非空
        first, last = 0, D - 1
    else:
        first, last = mask.argmax(), len(mask) - mask[::-1].argmax() - 1

    if mode == "uniform":
        indices = np.linspace(first, last, num=n_slices, dtype=int)

    elif mode == "normal":
        center = (first + last) // 2
        sampled = np.random.normal(
            loc=center,
            scale=(last - first) *scale,
            size=n_slices
        ).astype(int)
        sampled = np.clip(sampled, first, last)
        indices = np.unique(sampled)  # 去重避免重复
        if len(indices) < n_slices:
            # 不足时均匀补充
            extra = np.linspace(first, last, num=n_slices - len(indices), dtype=int)
            indices = np.sort(np.concatenate([indices, extra]))

    elif mode == "all":
        indices = np.arange(first, last + 1)

    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'uniform', 'normal', or 'all'.")

    return indices.tolist()


# --------------------------
# Teacher preprocessing helper functions (from dune/data/teacher_configs.py)
# --------------------------
def resize_volume(volume: torch.Tensor, target_size: tuple) -> torch.Tensor:
    """Resize volume to target size using trilinear interpolation."""
    if volume.shape[-3:] == target_size:
        return volume

    # Ensure 5D: [B, C, D, H, W]
    needs_batch = volume.dim() == 4
    if needs_batch:
        volume = volume.unsqueeze(0)

    resized = F.interpolate(
        volume,
        size=target_size,
        mode='trilinear',
        align_corners=False
    )

    if needs_batch:
        resized = resized.squeeze(0)

    return resized


def apply_normalization(volume: torch.Tensor, norm_type: str) -> torch.Tensor:
    """Apply normalization based on teacher requirements."""
    if norm_type == "zscore":
        # Z-score normalization (mean=0, std=1)
        mean = volume.mean()
        std = volume.std()
        normalized = (volume - mean) / (std + 1e-8)

    elif norm_type == "minmax":
        # Min-max normalization to [0, 1]
        vmin, vmax = volume.min(), volume.max()
        normalized = (volume - vmin) / (vmax - vmin + 1e-8)

    elif norm_type == "none":
        # No normalization - use raw data
        normalized = volume

    else:
        # Default: no normalization
        normalized = volume

    return normalized


def create_multi_views(volume: torch.Tensor, num_views: int = 2) -> List[torch.Tensor]:
    """
    Create multiple views for contrastive learning (for brainmvp).

    Args:
        volume: [C, D, H, W]
        num_views: number of views to generate

    Returns:
        List of augmented views
    """
    views = [volume]  # Original view

    # Create additional views with light augmentations
    for i in range(num_views - 1):
        view = volume.clone()

        # Light rotation (±5 degrees)
        if torch.rand(1) < 0.5:
            angle = (torch.rand(1) * 10 - 5).item()  # ±5 degrees
            view = light_rotation_3d(view, angle)

        # Light intensity variation (0.9-1.1x)
        if torch.rand(1) < 0.5:
            factor = (torch.rand(1) * 0.2 + 0.9).item()  # 0.9-1.1x
            view = view * factor

        views.append(view)

    return views


def light_rotation_3d(volume: torch.Tensor, angle_deg: float) -> torch.Tensor:
    """Apply light rotation in axial plane for multi-view generation."""
    angle_rad = torch.deg2rad(torch.tensor(angle_deg))
    cos_a, sin_a = torch.cos(angle_rad), torch.sin(angle_rad)

    # 2D rotation in the axial plane (x-y plane)
    rotation_matrix = torch.tensor([
        [cos_a, -sin_a, 0, 0],
        [sin_a, cos_a, 0, 0],
        [0, 0, 1, 0]
    ], dtype=torch.float32).unsqueeze(0)

    # Add batch dimension if needed
    needs_batch = volume.dim() == 4
    if needs_batch:
        volume = volume.unsqueeze(0)

    grid = F.affine_grid(rotation_matrix, volume.shape, align_corners=False)
    rotated = F.grid_sample(volume, grid, align_corners=False, mode='bilinear')

    if needs_batch:
        rotated = rotated.squeeze(0)

    return rotated


# --------------------------
# Dataset 主体
# --------------------------
class BMFVolumeSliceDataset(Dataset):
    """
    Dataset for Stage-2 Brain Foundation Model pretraining.
    Loads 3D MRI (.nii.gz) volumes and extracts a subset of 2D slices.

    Returns:
        {
            'volume': torch.Tensor [1, D, H, W],  # For 3D teacher
            'slices': torch.Tensor [n_slices, 1, H, W],  # For 2D student
            'path': str
        }
    """

    def __init__(self,
                 dataset_roots,
                 n_slices=8,
                 mode="train",
                 transform_2d=None,
                 transform_3d=None,
                 slice_mode="uniform",
                 sample_norm_scale=1,
                 teacher_name="bm_mae",
                 use_teacher_preprocessing=True):
        """
        Args:
            dataset_roots (list[str]): List of folders containing .nii.gz files.
            n_slices (int): Number of slices to sample per volume.
            mode (str): "train" or "val" (affects randomness).
            transform_2d: Augmentation for 2D student input.
            transform_3d: Augmentation for 3D teacher input.
            slice_mode: "uniform" | "normal" | "all"
            sample_norm_scale: Scale for normal sampling mode.
            teacher_name: Teacher model name ("bm_mae" or "brainmvp").
            use_teacher_preprocessing: Whether to apply teacher-specific preprocessing.
        """
        self.dataset_roots = dataset_roots
        self.n_slices = n_slices
        self.mode = mode
        self.transform_2d = transform_2d
        self.transform_3d = transform_3d
        self.slice_mode = slice_mode
        self.scale = sample_norm_scale
        self.teacher_name = teacher_name
        self.use_teacher_preprocessing = use_teacher_preprocessing

        # Load teacher config
        if self.use_teacher_preprocessing:
            if teacher_name not in TEACHER_PREPROCESSING_CONFIGS:
                raise ValueError(f"Unknown teacher: {teacher_name}. Available: {list(TEACHER_PREPROCESSING_CONFIGS.keys())}")
            self.teacher_config = TEACHER_PREPROCESSING_CONFIGS[teacher_name]
            print(f"✅ Using {teacher_name} preprocessing:")
            print(f"   - Target size: {self.teacher_config['target_size']}")
            print(f"   - Normalization: {self.teacher_config['normalization']}")
            if self.teacher_config.get('multi_view'):
                print(f"   - Multi-view: {self.teacher_config['num_views']} views")
        else:
            self.teacher_config = None
            print("⚠️  Teacher preprocessing disabled")

        # gather all nii files
        self.volume_paths = []
        for root in self.dataset_roots:
            if root.endswith(".lmdb"):
                env = lmdb.open(root, readonly=True, lock=False, subdir=False)
                with env.begin(write=False) as txn:
                    index_raw = txn.get(b"__index__")
                    if index_raw is None:
                        raise ValueError(f"❌ LMDB 文件 {root} 缺少 '__index__'")
                    keys = [k for k in index_raw.decode().strip().split("\n") if len(k) > 0]
                    # 每个样本 = (lmdb路径, key)
                    self.volume_paths += [(root, k) for k in keys]
                env.close()
            else:
                self.volume_paths += sorted(glob.glob(os.path.join(root, "*.nii.gz")))
        assert len(self.volume_paths) > 0, "No NIfTI files found in provided roots!"
        print(f"📁 Found {len(self.volume_paths)} volumes")

    def __len__(self):
        return len(self.volume_paths)

    def __getitem__(self, idx):
        entry = self.volume_paths[idx]

        # ------------------------------------------
        # ✅ Step 1. Load volume
        # ------------------------------------------
        if isinstance(entry, tuple):  # 来自 LMDB
            lmdb_path, key = entry
            volume, meta = open_single_lmdb(lmdb_path, key)
            path_display = f"{lmdb_path}:{key}"
            affine = np.eye(4)
            volume_dhw = volume.astype(np.float32)  # [D, H, W]

        elif isinstance(entry, str) and entry.endswith(".nii.gz"):
            path = entry
            nii = nib.load(path)
            volume = nii.get_fdata().astype(np.float32)
            affine = nii.affine
            meta = {}
            path_display = path

            # 做 RAS 对齐
            orig_ornt = io_orientation(affine)
            ras_ornt = axcodes2ornt(('R', 'A', 'S'))
            transform = nib.orientations.ornt_transform(orig_ornt, ras_ornt)
            volume = nib.orientations.apply_orientation(volume, transform)

            # depth方向保证是 S→I（头到脚）
            if affine[2, 2] < 0:
                volume = np.flip(volume, axis=2)

            # 转换成 [D,H,W]
            volume_dhw = volume.transpose(2, 0, 1)

        else:
            raise ValueError(f"❌ Unsupported entry type: {type(entry)}")


        # -------------------------------
        # ✅ Step 3. Normalization (在原始分辨率的完整3D volume上)
        # 重要：必须在取slice之前做normalization，确保slices和volume使用相同的统计量
        # -------------------------------
        if self.use_teacher_preprocessing and self.teacher_config is not None:
            # Convert to tensor for normalization
            volume_tensor = torch.from_numpy(volume_dhw).unsqueeze(0)  # [1, D, H, W]

            # Apply teacher-specific normalization on FULL volume
            volume_normalized = apply_normalization(
                volume_tensor,
                norm_type=self.teacher_config['normalization']
            )  # [1, D, H, W], normalized

            # Convert back to numpy for slice selection
            volume_dhw_normalized = volume_normalized.squeeze(0).numpy()  # [D, H, W]
        else:
            # Old behavior: Simple z-score normalization
            volume_dhw_normalized = (volume_dhw - np.mean(volume_dhw)) / (np.std(volume_dhw) + 1e-8)

        # -------------------------------
        # ✅ Step 4. 数据增强 (TODO: 在已经normalized的volume上进行，除了size相关的)
        # -------------------------------
        # 这里可以添加 MONAI augmentations
        # 注意：不要改变intensity distribution，因为已经normalized了
        # 可以做的augmentation：rotation, flip, elastic deformation等几何变换
        # 目前先保持原样

        # -------------------------------
        # ✅ Step 5. 从上到下选 slices (从已经normalized的volume中取)
        # -------------------------------
        # 调用自定义的 slice 选择函数
        slice_indices = select_slices(
            arr=volume_dhw_normalized,  # [D, H, W], already normalized
            n_slices=self.n_slices,
            mode=self.slice_mode,
            scale=self.scale
        )

        # 从上到下取切片: [n_slices, H, W]
        slices_normalized = volume_dhw_normalized[slice_indices]

        # -------------------------------
        # ✅ Step 6. Resize volume for teacher (slices保持原始分辨率)
        # 重要：只对volume做resize，slices不需要resize
        # -------------------------------
        # Convert slices to tensor [n_slices, 1, H, W] - 保持原始分辨率
        slice_tensor = torch.from_numpy(slices_normalized).unsqueeze(1)  # [n_slices, 1, H, W]

        # Resize full volume to target size for teacher (only volume, not slices)
        if self.use_teacher_preprocessing and self.teacher_config is not None:
            volume_tensor = torch.from_numpy(volume_dhw_normalized).unsqueeze(0)  # [1, D, H, W]
            volume_for_teacher = resize_volume(
                volume_tensor,
                target_size=self.teacher_config['target_size']  # (128, 128, 128)
            )  # [1, 128, 128, 128]
        else:
            # Old behavior
            volume_for_teacher = torch.from_numpy(volume_dhw_normalized).unsqueeze(0)  # [1, D, H, W]

        # # -------------------------------
        # # Multi-view generation (COMMENTED OUT)
        # # -------------------------------
        # # Create multi-views if needed (for brainmvp)
        # multi_views = None
        # if self.teacher_config.get('multi_view') and self.mode == "train":
        #     multi_views = create_multi_views(
        #         volume_for_teacher,
        #         num_views=self.teacher_config.get('num_views', 2)
        #     )  # List of [1, 128, 128, 128] tensors

        result = {
            "volume": volume_for_teacher,  # [1, 128, 128, 128] - for teacher
            "slices": slice_tensor,        # [n_slices, 1, 128, 128] - for student
            "path": path_display,
            "slice_indices": slice_indices
        }

        # # Add multi-views if available (COMMENTED OUT)
        # if multi_views is not None:
        #     result["multi_views"] = multi_views  # List of [1, 128, 128, 128] tensors

        return result
# --------------------------
# 使用示例
# --------------------------
if __name__ == "__main__":
    roots = [
        "/scr2/yz_wu/data/Foundation/Preprocessed/ICBM/ICBM_sin_head",
        "/scr2/yz_wu/data/Foundation/Preprocessed/ID1000/ID1000_sin_head",
    ]

    # Test 1: BM-MAE preprocessing (zscore normalization)
    print("\n" + "="*70)
    print("Test 1: BM-MAE preprocessing (zscore normalization)")
    print("="*70)
    dataset_bmmae = BMFVolumeSliceDataset(
        dataset_roots=roots,
        n_slices=8,
        slice_mode="uniform",
        teacher_name="bm_mae",
        use_teacher_preprocessing=True
    )
    sample = dataset_bmmae[0]
    print(f"✅ Volume shape: {sample['volume'].shape}")  # [1, 128, 128, 128]
    print(f"✅ Slices shape: {sample['slices'].shape}")  # [8, 1, 128, 128]
    print(f"✅ Volume range: [{sample['volume'].min():.3f}, {sample['volume'].max():.3f}]")
    print(f"✅ Multi-views: {'Yes' if 'multi_views' in sample else 'No'}")

    # Test 2: BrainMVP preprocessing (no normalization + multi-view)
    print("\n" + "="*70)
    print("Test 2: BrainMVP preprocessing (no normalization + multi-view)")
    print("="*70)
    dataset_brainmvp = BMFVolumeSliceDataset(
        dataset_roots=roots,
        n_slices=8,
        slice_mode="normal",
        teacher_name="brainmvp",
        use_teacher_preprocessing=True,
        mode="train"  # Enable multi-view in training mode
    )
    sample = dataset_brainmvp[0]
    print(f"✅ Volume shape: {sample['volume'].shape}")
    print(f"✅ Slices shape: {sample['slices'].shape}")
    print(f"✅ Volume range: [{sample['volume'].min():.3f}, {sample['volume'].max():.3f}]")
    if 'multi_views' in sample:
        print(f"✅ Multi-views: {len(sample['multi_views'])} views")
        for i, view in enumerate(sample['multi_views']):
            print(f"   View {i}: {view.shape}")

    # Test 3: Without teacher preprocessing (old behavior)
    print("\n" + "="*70)
    print("Test 3: Without teacher preprocessing (old behavior)")
    print("="*70)
    dataset_old = BMFVolumeSliceDataset(
        dataset_roots=roots,
        n_slices=8,
        slice_mode="uniform",
        use_teacher_preprocessing=False
    )
    sample = dataset_old[0]
    print(f"✅ Volume shape: {sample['volume'].shape}")
    print(f"✅ Slices shape: {sample['slices'].shape}")
    print(f"✅ Volume range: [{sample['volume'].min():.3f}, {sample['volume'].max():.3f}]")
