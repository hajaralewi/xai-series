"""Explain the brain tumour model with LRP and Grad-CAM.

Run:
    python xai_lrp_gradcam.py                       # sample images from the test set
    python xai_lrp_gradcam.py --per-class 4 --wrong 6
    python xai_lrp_gradcam.py --image path/to/mri.jpg   # explain one custom image

Outputs (in outputs/):
    xai_grid.png            original | LRP | Grad-CAM for all selected samples
    samples/*.png           one figure per sample
    deletion_test.png       faithfulness check (LRP vs Grad-CAM vs random)
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from captum.attr import LRP, LayerAttribution, LayerGradCam
from PIL import Image
from torch.utils.data import DataLoader

from common import (BATCH_SIZE, IMG_SIZE, MODEL_PATH, NUM_WORKERS, OUT_DIR, BrainCNN,
                    denormalize, get_datasets, get_device, get_transforms, set_seed)


# ----------------------------------------------------------------------------
# XAI methods
# ----------------------------------------------------------------------------
def lrp_map(lrp: LRP, x: torch.Tensor, target: int) -> np.ndarray:
    """Pixel-level relevance, shape (H, W). Positive = evidence FOR the class."""
    attr = lrp.attribute(x, target=target)
    return attr[0, 0].detach().cpu().numpy()


def gradcam_map(gradcam: LayerGradCam, x: torch.Tensor, target: int) -> np.ndarray:
    """Coarse localisation map scaled to [0, 1], shape (H, W)."""
    attr = gradcam.attribute(x, target=target, relu_attributions=True)
    attr = LayerAttribution.interpolate(attr, (IMG_SIZE, IMG_SIZE), interpolate_mode="bilinear")
    cam = attr[0, 0].detach().cpu().numpy()
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-12)


# ----------------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------------
def show_lrp(ax, img, lrp_attr):
    vmax = np.percentile(np.abs(lrp_attr), 99.5) + 1e-12
    ax.imshow(img, cmap="gray")
    ax.imshow(lrp_attr, cmap="seismic", vmin=-vmax, vmax=vmax, alpha=0.6)
    ax.set_title("LRP (red = for, blue = against)", fontsize=9)


def show_cam(ax, img, cam):
    ax.imshow(img, cmap="gray")
    ax.imshow(cam, cmap="jet", alpha=0.45)
    ax.set_title("Grad-CAM", fontsize=9)


def plot_row(axes, img, lrp_attr, cam, title):
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title(title, fontsize=9)
    show_lrp(axes[1], img, lrp_attr)
    show_cam(axes[2], img, cam)
    for a in axes:
        a.axis("off")


# ----------------------------------------------------------------------------
# Faithfulness: deletion test
# ----------------------------------------------------------------------------
@torch.no_grad()
def deletion_curve(model, x, target, heat, fractions):
    """Remove the top-k% most important pixels (set to black) and track the
    probability of the predicted class. A faster drop = more faithful explanation."""
    order = np.argsort(heat.ravel())[::-1]
    probs = []
    for f in fractions:
        k = int(f * order.size)
        mask = np.ones(order.size, dtype=np.float32)
        mask[order[:k]] = 0.0
        mask = torch.tensor(mask.reshape(IMG_SIZE, IMG_SIZE), device=x.device)
        x_masked = x * mask + (-1.0) * (1 - mask)  # -1 = black after Normalize
        probs.append(F.softmax(model(x_masked), dim=1)[0, target].item())
    return probs


# ----------------------------------------------------------------------------
# Sample selection
# ----------------------------------------------------------------------------
def select_samples(model, dataset, device, per_class, n_wrong, seed):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    preds = []
    with torch.no_grad():
        for x, _ in loader:
            preds += model(x.to(device)).argmax(1).cpu().tolist()
    labels = [y for _, y in dataset.samples]
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(dataset))

    chosen, per_cls_count, wrong_count = [], {c: 0 for c in range(len(dataset.classes))}, 0
    for i in order:
        if preds[i] == labels[i] and per_cls_count[labels[i]] < per_class:
            chosen.append(int(i)); per_cls_count[labels[i]] += 1
        elif preds[i] != labels[i] and wrong_count < n_wrong:
            chosen.append(int(i)); wrong_count += 1
    chosen.sort(key=lambda i: (labels[i], preds[i] != labels[i]))
    return chosen


# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=3, help="correct samples per class")
    parser.add_argument("--wrong", type=int, default=4, help="misclassified samples to include")
    parser.add_argument("--image", type=str, default=None, help="explain a single image file")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    (OUT_DIR / "samples").mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(MODEL_PATH, map_location=device)
    classes = ckpt["classes"]
    model = BrainCNN(n_classes=len(classes)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    lrp = LRP(model)
    gradcam = LayerGradCam(model, model.gradcam_layer())

    # ---------------- build list of (tensor, true_label_name) ----------------
    if args.image:
        tfm = get_transforms(train=False)
        x = tfm(Image.open(args.image).convert("RGB"))
        items = [(x, "unknown", args.image)]
    else:
        _, _, test_ds = get_datasets()
        idxs = select_samples(model, test_ds, device, args.per_class, args.wrong, args.seed)
        items = [(test_ds[i][0], classes[test_ds[i][1]], test_ds.samples[i][0]) for i in idxs]
    print(f"Explaining {len(items)} image(s)...")

    fractions = [0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    curves = {"LRP": [], "Grad-CAM": [], "Random": []}

    fig, axes = plt.subplots(len(items), 3, figsize=(10, 3.2 * len(items)), squeeze=False)
    for r, (x, true_name, path) in enumerate(items):
        x = x.unsqueeze(0).to(device)
        with torch.no_grad():
            probs = F.softmax(model(x), dim=1)[0]
        pred = int(probs.argmax())
        title = f"True: {true_name}\nPred: {classes[pred]} ({probs[pred]:.1%})"

        attr_lrp = lrp_map(lrp, x, pred)
        cam = gradcam_map(gradcam, x, pred)
        img = denormalize(x)[0, 0].cpu().numpy()

        plot_row(axes[r], img, attr_lrp, cam, title)

        # single-sample figure
        f1, a1 = plt.subplots(1, 3, figsize=(10, 3.4))
        plot_row(a1, img, attr_lrp, cam, title)
        f1.tight_layout()
        f1.savefig(OUT_DIR / "samples" / f"{r:02d}_true-{true_name}_pred-{classes[pred]}.png", dpi=130)
        plt.close(f1)

        # deletion test (LRP: only positive relevance counts as evidence)
        curves["LRP"].append(deletion_curve(model, x, pred, np.maximum(attr_lrp, 0), fractions))
        curves["Grad-CAM"].append(deletion_curve(model, x, pred, cam, fractions))
        curves["Random"].append(deletion_curve(model, x, pred, np.random.rand(IMG_SIZE, IMG_SIZE), fractions))

    fig.tight_layout()
    fig.savefig(OUT_DIR / "xai_grid.png", dpi=130)
    plt.close(fig)

    # ---------------- deletion plot ----------------
    plt.figure(figsize=(6, 4))
    for name, c in curves.items():
        plt.plot([f * 100 for f in fractions], np.mean(c, axis=0), marker="o", label=name)
    plt.xlabel("% of most-relevant pixels removed")
    plt.ylabel("Mean probability of predicted class")
    plt.title("Deletion test (lower / faster drop = more faithful)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(OUT_DIR / "deletion_test.png", dpi=150)
    plt.close()

    print("\nMean predicted-class probability after deleting top pixels:")
    print("removed%  " + "  ".join(f"{n:>9}" for n in curves))
    for j, f in enumerate(fractions):
        print(f"{f*100:7.0f}%  " + "  ".join(f"{np.mean(curves[n], axis=0)[j]:9.3f}" for n in curves))
    print("\nSaved figures to", OUT_DIR)


if __name__ == "__main__":
    main()
