"""
Astrotect — YOLOv8 Nano Object Detection
-----------------------------------------
Classes: 0=comet  1=galaxy  2=globular_cluster  3=nebula

Run:
    pip install ultralytics
    python main.py
"""

import os
import random
import shutil
from pathlib import Path

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO

from dataset_reader import DatasetReader

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_PATH   =  Path(__file__).parent.parent / "dataset"
YAML_PATH      = Path(__file__).parent.parent / "dataset.yaml"
MODEL_NAME     = "yolov8n.pt"          # nano — fastest, still very accurate
EPOCHS         = 10
IMG_SIZE       = 640                   # keep native res, YOLO handles it efficiently
BATCH_SIZE     = 32                    # bump to 32 if you have VRAM headroom
WORKERS        = 4
DEVICE         = 0 if torch.cuda.is_available() else "cpu"
VIS_SAMPLES    = 8                     # how many val images to visualize after training
VIS_CONF       = 0.25                  # confidence threshold for drawing boxes
OUTPUT_DIR     = Path("../runs/astrotect")

CLASS_NAMES    = {0: "Comet", 1: "Galaxy", 2: "Globular Cluster", 3: "Nebula"}
CLASS_COLORS   = {0: "#FF6B6B", 1: "#4ECDC4", 2: "#FFE66D", 3: "#A29BFE"}


# ── Visualization helpers ─────────────────────────────────────────────────────

def load_gt_boxes(label_path: Path, img_w: int, img_h: int):
    """Parse YOLO-format label file → list of (class_id, x1, y1, x2, y2)."""
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


def hex_to_rgb(hex_color: str):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def visualize_predictions(model, val_img_dir: Path, val_label_dir: Path,
                           n_samples: int = VIS_SAMPLES, conf: float = VIS_CONF):
    """
    For n random val images, draw:
      - GREEN dashed boxes  = ground truth
      - COLORED solid boxes = model predictions with confidence score
    Saves a single figure grid to OUTPUT_DIR/predictions_grid.png
    """
    img_files = list(val_img_dir.glob("*.jpg")) + list(val_img_dir.glob("*.png"))
    if not img_files:
        print("[WARN] No validation images found for visualization.")
        return

    samples = random.sample(img_files, min(n_samples, len(img_files)))
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

        # Load image
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        # ── Ground truth boxes (green dashed) ────────────────────────────────
        label_path = val_label_dir / (img_path.stem + ".txt")
        gt_boxes = load_gt_boxes(label_path, w, h)

        ax.imshow(img_rgb)

        for (cls, x1, y1, x2, y2) in gt_boxes:
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=1.5, edgecolor="#00FF88",
                facecolor="none", linestyle="--"
            )
            ax.add_patch(rect)
            ax.text(x1, y1 - 4, f"GT:{CLASS_NAMES.get(cls, cls)}",
                    color="#00FF88", fontsize=6, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="#0d0d0d", alpha=0.6))

        # ── Model predictions (colored solid) ────────────────────────────────
        results = model.predict(str(img_path), conf=conf, verbose=False)[0]
        for box in results.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            bx1, by1, bx2, by2 = map(int, box.xyxy[0].tolist())
            color = CLASS_COLORS.get(cls_id, "#FFFFFF")

            rect = patches.Rectangle(
                (bx1, by1), bx2 - bx1, by2 - by1,
                linewidth=2, edgecolor=color,
                facecolor=color + "33"   # 20% fill
            )
            ax.add_patch(rect)
            label_txt = f"{CLASS_NAMES.get(cls_id, cls_id)} {confidence:.0%}"
            ax.text(bx1, by2 + 10, label_txt,
                    color=color, fontsize=7, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="#0d0d0d", alpha=0.75))

    # Legend
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
    print(f"\n[INFO] Prediction grid saved → {out_path}")
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  ASTROTECT — YOLOv8 Nano Training")
    print(f"  Device : {'GPU (CUDA)' if DEVICE == 0 else 'CPU'}")
    print(f"  Epochs : {EPOCHS}   Batch: {BATCH_SIZE}   ImgSize: {IMG_SIZE}")
    print("=" * 55)

    # ── 1. Dataset analysis ───────────────────────────────────────────────────
    print("\n[STEP 1] Analyzing dataset...")
    analyzer = DatasetReader(DATASET_PATH)
    analyzer.show_distribution()

    # ── 2. Load YOLOv8 nano ───────────────────────────────────────────────────
    print("\n[STEP 2] Loading YOLOv8 nano (downloads ~6MB if not cached)...")
    model = YOLO(MODEL_NAME, verbose=False)

    # ── 3. Train ──────────────────────────────────────────────────────────────
    print("\n[STEP 3] Training...")
    results = model.train(
        data=str(YAML_PATH),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        workers=WORKERS,
        device=DEVICE,
        project=str(OUTPUT_DIR.parent),
        name=OUTPUT_DIR.name,
        exist_ok=True,
        #resume=True,
        # Augmentation — good defaults for astro data
        hsv_h=0.015,
        hsv_s=0.3,
        hsv_v=0.3,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        # Logging
        plots=True,
        verbose=False,
    )

    print("\n[STEP 4] Training complete. Loading best weights...")
    # Find the latest best.pt from runs
    import glob
    weight_candidates = sorted(glob.glob("runs/astrotect*/weights/best.pt"))
    if not weight_candidates:
        weight_candidates = sorted(glob.glob("runs/**/weights/best.pt", recursive=True))
    if not weight_candidates:
        raise FileNotFoundError("Could not find best.pt — check your runs/ folder")
    best_weights = weight_candidates[-1]  # take the most recent
    print(f"[INFO] Loading weights from: {best_weights}")
    model = YOLO(best_weights)

    # ── 4. Validation metrics ─────────────────────────────────────────────────
    print("\n[STEP 5] Running validation...")
    metrics = model.val(data=str(YAML_PATH), device=DEVICE, verbose=True)
    print(f"\n  mAP50      : {metrics.box.map50:.4f}")
    print(f"  mAP50-95   : {metrics.box.map:.4f}")

    # ── 5. Visualize predictions ──────────────────────────────────────────────
    print("\n[STEP 6] Generating prediction visualization...")
    val_img_dir   = DATASET_PATH / "valid" / "images"
    val_label_dir = DATASET_PATH / "valid" / "labels"
    visualize_predictions(model, val_img_dir, val_label_dir)

    print("\n✓ Done! Results saved to:", OUTPUT_DIR)
    print("  • predictions_grid.png  — visual predictions vs ground truth")
    print("  • weights/best.pt       — best model weights")
    print("  • results.png           — training curves (loss / mAP)")


if __name__ == "__main__":
    main()