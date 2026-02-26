import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def yolo_box_to_xyxy(box: list, w: int, h: int) -> tuple:
    x_c, y_c, bw, bh = box
    x1 = max(0, int((x_c - bw / 2) * w))
    y1 = max(0, int((y_c - bh / 2) * h))
    x2 = min(w, int((x_c + bw / 2) * w))
    y2 = min(h, int((y_c + bh / 2) * h))
    return x1, y1, x2, y2


def compute_iou(pred: list, target: list) -> float:
    px1, py1, px2, py2 = pred
    tx1, ty1, tx2, ty2 = target
    ix1, iy1 = max(px1, tx1), max(py1, ty1)
    ix2, iy2 = min(px2, tx2), min(py2, ty2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (px2-px1)*(py2-py1) + (tx2-tx1)*(ty2-ty1) - inter
    return inter / union if union > 0 else 0.0


def compute_mean_iou(samples, preprocessor, detector_model) -> float:
    total, count = 0.0, 0
    for img_path, _, box in samples:
        img = Image.open(str(img_path))
        w, h = img.size
        img_bgr = cv2.imread(str(img_path))
        img_bgr = preprocessor.do_magic(img_bgr, show=False)
        results = detector_model.predict(img_bgr, conf=0.25, verbose=False)[0]
        if results.boxes:
            pred = list(map(int, results.boxes[0].xyxy[0].tolist()))
            gt   = list(yolo_box_to_xyxy(box, w, h))
            total += compute_iou(pred, gt)
            count += 1
    return total / count if count > 0 else 0.0


def append_csv(csv_file: str, header: list, row: list) -> None:
    path = Path(csv_file)
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)