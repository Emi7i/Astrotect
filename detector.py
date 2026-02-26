import random
import glob
from pathlib import Path

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO
from PIL import Image

from image_preprocessor import ImagePreprocessor
from dataset import AstroDataset
from utils import compute_iou, append_csv

DATASET_PATH = Path("dataset")
YAML_PATH    = "dataset.yaml"
OUTPUT_DIR   = Path("models/detector")
MODEL_NAME   = "yolov8n.pt"

EPOCHS     = 150
IMG_SIZE   = 640
BATCH_SIZE = 32
WORKERS    = 4
DEVICE     = 0 if torch.cuda.is_available() else "cpu"
PATIENCE   = 10

VIS_SAMPLES = 10
VIS_CONF    = 0.25

CLASS_NAMES  = {0: "Comet", 1: "Galaxy", 2: "Globular Cluster", 3: "Nebula"}
CLASS_COLORS = {0: "#FF6B6B", 1: "#4ECDC4", 2: "#FFE66D", 3: "#A29BFE"}


class Detector:
    def __init__(self):
        self.path_to_weights = ""
        self.preprocessor = ImagePreprocessor()
        self.val_dataset = AstroDataset(
            str(DATASET_PATH / "valid" / "images"),
            str(DATASET_PATH / "valid" / "labels")
        )

    def train(self):
        if not DATASET_PATH.exists():
            raise FileNotFoundError(f"Dataset path {DATASET_PATH} does not exist.")
        if not Path(YAML_PATH).exists():
            raise FileNotFoundError(f"YAML file {YAML_PATH} does not exist.")

        print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

        model = YOLO(MODEL_NAME, verbose=False)
        model.add_callback("on_fit_epoch_end", self._save_epoch_metrics)
        model.train(
            data=YAML_PATH, epochs=EPOCHS, imgsz=IMG_SIZE,
            batch=BATCH_SIZE, workers=WORKERS, device=DEVICE,
            freeze=10, lr0=0.001, weight_decay=0.0005,
            project=str(OUTPUT_DIR.parent), name=OUTPUT_DIR.name, exist_ok=True,
            hsv_h=0.015, hsv_s=0.3, hsv_v=0.3,
            flipud=0.5, fliplr=0.5, mosaic=1.0,
            plots=True, verbose=False, patience=PATIENCE,
        )

        self.path_to_weights = self.find_best_weights()
        dest = Path("models/detection_model.pt")
        dest.parent.mkdir(exist_ok=True)
        dest.write_bytes(Path(self.path_to_weights).read_bytes())
        print(f"Model weights copied to {dest}")
        return self.path_to_weights

    def _save_epoch_metrics(self, trainer) -> None:
        m = trainer.metrics
        append_csv(
            'results_detector_epochs.csv',
            ['Epoch', 'box_loss', 'cls_loss', 'dfl_loss', 'Precision', 'Recall', 'mAP50', 'mAP50-95'],
            [
                trainer.epoch + 1,
                f"{m.get('train/box_loss', 0):.4f}",
                f"{m.get('train/cls_loss', 0):.4f}",
                f"{m.get('train/dfl_loss', 0):.4f}",
                f"{m.get('metrics/precision(B)', 0):.4f}",
                f"{m.get('metrics/recall(B)', 0):.4f}",
                f"{m.get('metrics/mAP50(B)', 0):.4f}",
                f"{m.get('metrics/mAP50-95(B)', 0):.4f}",
            ]
        )

    def validate(self, split: str = 'val') -> dict:
        if not self.path_to_weights:
            self.path_to_weights = self.find_best_weights()
        model = YOLO(self.path_to_weights)

        metrics = model.val(data=YAML_PATH, device=DEVICE, verbose=False, split=split)
        avg_iou = self._compute_average_iou(model)

        print(f"mAP50: {metrics.box.map50:.4f} | mAP50-95: {metrics.box.map:.4f} | IoU: {avg_iou:.4f}")

        img_dir   = DATASET_PATH / ("test" if split == "test" else "valid") / "images"
        label_dir = DATASET_PATH / ("test" if split == "test" else "valid") / "labels"
        self._visualize(model, img_dir, label_dir, split)

        return {'precision': float(np.mean(metrics.box.p)), 'recall': float(np.mean(metrics.box.r)),
                'map50': float(metrics.box.map50), 'map50_95': float(metrics.box.map), 'iou': avg_iou}

    def test(self) -> dict:
        return self.validate(split='test')

    def _compute_average_iou(self, model) -> float:
        total, count = 0.0, 0
        for img_path, _, box in self.val_dataset.samples:
            img = np.array(Image.open(str(img_path)))
            h, w = img.shape[:2]
            results = model.predict(img, conf=0.25, verbose=False)[0]
            if results.boxes:
                pred = list(map(int, results.boxes[0].xyxy[0].tolist()))
                gt = [int((box[0]-box[2]/2)*w), int((box[1]-box[3]/2)*h),
                      int((box[0]+box[2]/2)*w), int((box[1]+box[3]/2)*h)]
                total += compute_iou(pred, gt)
                count += 1
        return total / count if count > 0 else 0.0

    def find_best_weights(self) -> str:
        pt_path = Path("models/detection_model.pt")
        if pt_path.exists():
            return str(pt_path)
        candidates = sorted(glob.glob("**/weights/best.pt", recursive=True))
        if not candidates:
            raise FileNotFoundError("Could not find detection model.")
        return candidates[-1]

    def _load_gt_boxes(self, label_path: Path, w: int, h: int) -> list:
        if not label_path.exists():
            return []
        boxes = []
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                boxes.append((cls,
                    int((cx-bw/2)*w), int((cy-bh/2)*h),
                    int((cx+bw/2)*w), int((cy+bh/2)*h)))
        return boxes

    def _visualize(self, model, img_dir: Path, label_dir: Path, split: str) -> None:
        img_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        if not img_files:
            print("[WARN] No images found, skipping visualization.")
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
            img_bgr = self.preprocessor.do_magic(img_bgr, show=False)
            results = model.predict(img_bgr, conf=VIS_CONF, verbose=False)[0]
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]
            ax.imshow(img_rgb)

            for cls, x1, y1, x2, y2 in self._load_gt_boxes(label_dir / (img_path.stem + ".txt"), w, h):
                ax.add_patch(patches.Rectangle((x1, y1), x2-x1, y2-y1,
                    linewidth=1.5, edgecolor="#00FF88", facecolor="none", linestyle="--"))
                ax.text(x1, y1-4, f"GT: {CLASS_NAMES.get(cls, cls)}",
                    color="#00FF88", fontsize=6, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="#0d0d0d", alpha=0.6))

            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                bx1, by1, bx2, by2 = map(int, box.xyxy[0].tolist())
                color = CLASS_COLORS.get(cls_id, "#FFFFFF")
                ax.add_patch(patches.Rectangle((bx1, by1), bx2-bx1, by2-by1,
                    linewidth=2, edgecolor=color, facecolor=color+"33"))
                ax.text(bx1, by2+10, f"{CLASS_NAMES.get(cls_id, cls_id)} {conf:.0%}",
                    color=color, fontsize=7, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="#0d0d0d", alpha=0.75))

        legend = [patches.Patch(facecolor="none", edgecolor="#00FF88", linestyle="--", label="Ground Truth")] + \
                 [patches.Patch(facecolor=c+"55", edgecolor=c, label=n)
                  for c, n in zip(CLASS_COLORS.values(), CLASS_NAMES.values())]
        fig.legend(handles=legend, loc="lower center", ncol=5,
            facecolor="#1a1a1a", edgecolor="white", labelcolor="white",
            fontsize=9, bbox_to_anchor=(0.5, 0.01))
        plt.suptitle("Astrotect — Predictions vs Ground Truth",
            color="white", fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout(rect=[0, 0.05, 1, 1])

        out_path = OUTPUT_DIR / f"predictions_{split}.png"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.show()
        print(f"[INFO] Saved to {out_path}")