# visualize_mri.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch.nn.functional as F

# skimage optional
try:
    from skimage.morphology import remove_small_objects, remove_small_holes, binary_closing, disk
    from skimage.segmentation import clear_border
    from skimage.filters import threshold_otsu
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False


# ------------------------------------------------------------
# 🔹 通用工具函数
# ------------------------------------------------------------
def normalize_tensor(sample: torch.Tensor) -> torch.Tensor:
    return (sample - torch.min(sample)) / (torch.max(sample) - torch.min(sample) + 1e-8)

def _to_hw_numpy(img_chw: torch.Tensor) -> np.ndarray:
    if img_chw.ndim == 3:
        c, h, w = img_chw.shape
        if c == 1:
            img = img_chw[0]
        else:
            img = img_chw.mean(dim=0)
    elif img_chw.ndim == 2:
        img = img_chw
    else:
        raise ValueError(f"Unexpected sample shape: {img_chw.shape}")
    img = img.clamp(0, 1)
    return img.numpy()

# ------------------------------------------------------------
# 🔹 PCA visualization
# ------------------------------------------------------------
def visualize_pca(sample, model, input_size, device="cuda"):
    norm_sample = normalize_tensor(sample)
    # 确保使用模型所在的设备
    model_device = next(model.parameters()).device
    with torch.no_grad():
        features = model(sample.unsqueeze(0).to(model_device), is_training=True)['x_norm_patchtokens']

    pca = PCA(n_components=3, whiten=True)
    pca_features = pca.fit_transform(features.squeeze(0).cpu().numpy())

    norm_pca_feats = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())
    print(f"norm_pca_feats shape: {norm_pca_feats.shape}")
    norm_pca_feats = norm_pca_feats.reshape(input_size // model.patch_size, input_size // model.patch_size, -1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(np.transpose(norm_sample.cpu().numpy(), (1, 2, 0)), cmap="gray")
    axes[0].set_title("MRI image")
    axes[0].axis("off")
    axes[1].imshow(norm_pca_feats)
    axes[1].set_title("PCA Features")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# 🔹 CLS vs patch similarity
# ------------------------------------------------------------
def visualize_cls_patch(sample, model, input_size, device="cuda"):
    norm_sample = normalize_tensor(sample)
    num_patches = input_size // model.patch_size
    # 确保使用模型所在的设备
    model_device = next(model.parameters()).device

    with torch.no_grad():
        out = model(sample.unsqueeze(0).to(model_device), is_training=True)
        cls = F.normalize(out['x_norm_clstoken'], dim=-1)
        patches = F.normalize(out['x_norm_patchtokens'], dim=-1)
        sim = torch.matmul(patches, cls.unsqueeze(-1)).squeeze(-1)

    sim_map = sim.reshape(num_patches, num_patches).cpu().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(np.transpose(norm_sample.cpu().numpy(), (1,2,0)), cmap="gray")
    axes[0].set_title("MRI image")
    axes[0].axis("off")
    im = axes[1].imshow(sim_map, cmap="viridis", vmin=-1, vmax=1)
    axes[1].set_title("CLS vs Patch Similarity")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# 🔹 Reference patch similarity
# ------------------------------------------------------------
def visualize_ref_patch(sample, model, input_size, location="center", device="cuda"):
    norm_sample = normalize_tensor(sample)
    ps = model.patch_size
    num_patches = input_size // ps
    # 确保使用模型所在的设备
    model_device = next(model.parameters()).device

    if location == "center":
        idx = (num_patches // 2) * num_patches + (num_patches // 2)
        row, col = num_patches // 2, num_patches // 2
    else:
        idx = int(location)
        row, col = idx // num_patches, idx % num_patches

    with torch.no_grad():
        out = model(sample.unsqueeze(0).to(model_device), is_training=True)
        patch_token = F.normalize(out['x_norm_patchtokens'], dim=-1).squeeze(0)
        ref = patch_token[idx:idx+1]
        sim = torch.matmul(patch_token, ref.T).squeeze(-1)
        sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)
        sim_map = sim.reshape(num_patches, num_patches).cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(np.transpose(norm_sample.cpu().numpy(), (1, 2, 0)), cmap="gray")
    axes[0].set_title("MRI image")
    axes[0].axis("off")
    axes[1].imshow(sim_map, cmap="viridis")
    axes[1].plot(col, row, 'rs', markersize=8, markerfacecolor='red')
    axes[1].set_title("Reference Patch Similarity")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import torch

# ------------------------------------------------------------
# 🔹 Reference patch similarity with mouse click
# ------------------------------------------------------------
def visualize_ref_patch_clickable(sample, model, input_size, device="cuda"):
    norm_sample = normalize_tensor(sample)
    ps = model.patch_size
    num_patches = input_size // ps
    model_device = next(model.parameters()).device

    # --------------------------------------------------------
    # 内部函数：根据点击位置计算 similarity map
    # --------------------------------------------------------
    def compute_similarity(row, col):
        with torch.no_grad():
            out = model(sample.unsqueeze(0).to(model_device), is_training=True)
            patch_token = F.normalize(out['x_norm_patchtokens'], dim=-1).squeeze(0)
            idx = row * num_patches + col
            ref = patch_token[idx:idx+1]
            sim = torch.matmul(patch_token, ref.T).squeeze(-1)
            sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)
            sim_map = sim.reshape(num_patches, num_patches).cpu().numpy()
        return sim_map

    # --------------------------------------------------------
    # 初始化图像
    # --------------------------------------------------------
    sim_map = compute_similarity(num_patches // 2, num_patches // 2)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # 左图：MRI
    img_ax = axes[0]
    img_ax.imshow(np.transpose(norm_sample.cpu().numpy(), (1, 2, 0)), cmap="gray")
    img_ax.set_title("MRI image (click to select reference patch)")
    img_ax.axis("off")

    # 右图：Similarity map
    sim_ax = axes[1]
    im = sim_ax.imshow(sim_map, cmap="viridis")
    ref_dot, = sim_ax.plot(num_patches // 2, num_patches // 2, 'rs', markersize=8, markerfacecolor='red')
    sim_ax.set_title("Reference Patch Similarity")
    sim_ax.axis("off")

    # --------------------------------------------------------
    # 点击事件处理
    # --------------------------------------------------------
    def on_click(event):
        if event.inaxes != img_ax:
            return
        x, y = int(event.xdata), int(event.ydata)

        # 将像素坐标转换为 patch grid 坐标
        col = min(x // ps, num_patches - 1)
        row = min(y // ps, num_patches - 1)

        print(f"📍 选中位置: 像素=({x},{y}) → patch=({row},{col})")

        sim_map = compute_similarity(row, col)
        im.set_data(sim_map)
        ref_dot.set_data([col], [row])
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.tight_layout()
    plt.show()
