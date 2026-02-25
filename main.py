import random
import glob
from pathlib import Path

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO

# ──────────────────[ CONFIG ]──────────────────
DATASET_PATH  = "dataset"
YAML_PATH     = "dataset.yaml"
OUTPUT_DIR    = Path("../runs/astrotect")
MODEL_NAME    = "yolov8n.pt"

EPOCHS        = 50
IMG_SIZE      = 640
BATCH_SIZE    = 32
WORKERS       = 4
DEVICE        = 0 if torch.cuda.is_available() else "cpu"
PATIENCE = 8

VIS_SAMPLES   = 8
VIS_CONF      = 0.25

CLASS_NAMES   = {0: "Comet", 1: "Galaxy", 2: "Globular Cluster", 3: "Nebula"}
CLASS_COLORS  = {0: "#FF6B6B", 1: "#4ECDC4", 2: "#FFE66D", 3: "#A29BFE"}

# ──────────────────[ PREPROCESSING ]──────────────────
def preprocess_image(img_bgr: np.ndarray) -> np.ndarray:
    denoised = cv2.GaussianBlur(img_bgr, (3, 3), 0)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return enhanced

# ──────────────────[ HELPER METHODS ]──────────────────
def load_gt_boxes(label_path: Path, img_w: int, img_h: int) -> list:
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])
            x1 = int((cx - bw / 2) * img_w)
            y1 = int((cy - bh / 2) * img_h)
            x2 = int((cx + bw / 2) * img_w)
            y2 = int((cy + bh / 2) * img_h)
            boxes.append((cls, x1, y1, x2, y2))
    return boxes

def find_best_weights() -> str:
    candidates = sorted(glob.glob("runs/astrotect*/weights/best.pt"))
    if not candidates:
        candidates = sorted(glob.glob("runs/**/weights/best.pt", recursive=True))
    if not candidates:
        raise FileNotFoundError("Could not find best.pt")
    return candidates[-1]

# ──────────────────[ VISUALIZATION ]──────────────────
def visualize_predictions(model, val_img_dir: Path, val_label_dir: Path) -> None:
    img_files = list(val_img_dir.glob("*.jpg")) + list(val_img_dir.glob("*.png"))
    if not img_files:
        return

    samples = random.sample(img_files, min(VIS_SAMPLES, len(img_files)))
    cols = 4
    rows = (len(samples) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    axes = np.array(axes).flatten()
    fig.patch.set_facecolor("#0d0d0d")

    for ax in axes:
        ax.axis("off")
        ax.set_facecolor("#0d0d0d")

    for i, img_path in enumerate(samples):
        ax = axes[i]

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        img_bgr = preprocess_image(img_bgr)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        ax.imshow(img_rgb)

        label_path = val_label_dir / (img_path.stem + ".txt")
        for (cls, x1, y1, x2, y2) in load_gt_boxes(label_path, w, h):
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=1.5, edgecolor="#00FF88",
                facecolor="none", linestyle="--"
            )
            ax.add_patch(rect)
            ax.text(x1, y1 - 4, f"GT: {CLASS_NAMES.get(cls, cls)}",
                    color="#00FF88", fontsize=6, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="#0d0d0d", alpha=0.6))

        results = model.predict(str(img_path), conf=VIS_CONF, verbose=False)[0]
        for box in results.boxes:
            cls_id     = int(box.cls[0])
            confidence = float(box.conf[0])
            bx1, by1, bx2, by2 = map(int, box.xyxy[0].tolist())
            color = CLASS_COLORS.get(cls_id, "#FFFFFF")

            rect = patches.Rectangle(
                (bx1, by1), bx2 - bx1, by2 - by1,
                linewidth=2, edgecolor=color,
                facecolor=color + "33"
            )
            ax.add_patch(rect)
            ax.text(bx1, by2 + 10, f"{CLASS_NAMES.get(cls_id, cls_id)} {confidence:.0%}",
                    color=color, fontsize=7, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="#0d0d0d", alpha=0.75))

    legend_elements = [
        patches.Patch(facecolor="none", edgecolor="#00FF88", linestyle="--", label="Ground Truth"),
    ] + [
        patches.Patch(facecolor=c + "55", edgecolor=c, label=n)
        for c, n in zip(CLASS_COLORS.values(), CLASS_NAMES.values())
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=5,
               facecolor="#1a1a1a", edgecolor="white",
               labelcolor="white", fontsize=9, bbox_to_anchor=(0.5, 0.01))

    plt.suptitle("Astrotect — Predictions vs Ground Truth", color="white",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    out_path = OUTPUT_DIR / "predictions_grid.png"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()

# ──────────────────[ MAIN ]──────────────────
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = YOLO(MODEL_NAME, verbose=False)

    model.train(
        data=str(YAML_PATH),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        workers=WORKERS,
        device=DEVICE,
        project=str(OUTPUT_DIR.parent),
        name=OUTPUT_DIR.name,
        exist_ok=True,
        hsv_h=0.015,
        hsv_s=0.3,
        hsv_v=0.3,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        plots=True,
        verbose=False,
        patience=PATIENCE,
    )

    best_weights = find_best_weights()
    model = YOLO(best_weights)

    metrics = model.val(data=str(YAML_PATH), device=DEVICE, verbose=False)
    print(f"mAP50: {metrics.box.map50:.4f}, mAP50-95: {metrics.box.map:.4f}")

    visualize_predictions(
        model,
        val_img_dir=DATASET_PATH / "valid" / "images",
        val_label_dir=DATASET_PATH / "valid" / "labels"
    )

if __name__ == "__main__":
    main()