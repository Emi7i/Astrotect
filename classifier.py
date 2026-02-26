import random
from pathlib import Path

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
from sklearn.metrics import average_precision_score, confusion_matrix
from ultralytics import YOLO

from detector import Detector
from image_preprocessor import ImagePreprocessor
from dataset import AstroDataset
from utils import compute_iou, compute_mean_iou, append_csv, yolo_box_to_xyxy

CROP_SIZE    = (224, 224)
BATCH_SIZE   = 32
NUM_EPOCHS   = 30
LR           = 0.0001
NUM_WORKERS  = 6
DROPOUT_RATE = 0.5
PATIENCE     = 4
OUTPUT_DIR   = Path("models")

CLASSES     = ['comet', 'galaxy', 'nebula', 'globular_cluster']
NUM_CLASSES = len(CLASSES)

EPOCH_CSV_HEADER  = ['Epoch', 'Train Loss', 'Train Acc (%)', 'Val Loss', 'Val Acc (%)',
                     'mAP (%)'] + [f'{c} AP' for c in CLASSES]
RESULT_CSV_HEADER = ['Loss', 'Acc (%)', 'mAP (%)'] + [f'{c} AP' for c in CLASSES]


class ClassificationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone[7].parameters():
            param.requires_grad = True
        self.flatten    = nn.Flatten()
        self.dropout    = nn.Dropout(DROPOUT_RATE)
        self.fc1        = nn.Linear(512, 256)
        self.fc2        = nn.Linear(256, NUM_CLASSES)
        self.activation = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.activation(self.fc1(x))
        return self.fc2(x)


def compute_metrics(loader, model, loss_fn, device) -> dict:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    class_correct = [0] * NUM_CLASSES
    class_total   = [0] * NUM_CLASSES
    all_preds, all_labels = [], []

    with torch.no_grad():
        for crops, labels in loader:
            crops, labels = crops.to(device), labels.to(device)
            preds      = model(crops)
            total_loss += loss_fn(preds, labels).item()
            predicted  = preds.argmax(dim=1)
            correct   += (predicted == labels).sum().item()
            total     += labels.size(0)
            for i in range(len(labels)):
                class_total[labels[i]] += 1
                if predicted[i] == labels[i]:
                    class_correct[labels[i]] += 1
            all_preds.append(torch.softmax(preds, dim=1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    predicted_labels = np.argmax(all_preds, axis=1)
    
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, predicted_labels)
    
    ap_per_class = [
        average_precision_score((all_labels == i).astype(int), all_preds[:, i])
        for i in range(NUM_CLASSES)
    ]
    return {
        'loss': total_loss / len(loader),
        'acc': correct / total * 100,
        'mAP': np.mean(ap_per_class) * 100,
        'ap_per_class': ap_per_class,
        'class_correct': class_correct,
        'class_total': class_total,
        'confusion_matrix': cm,
    }


def print_metrics(prefix: str, m: dict, iou: float = None) -> None:
    iou_str = f", IoU: {iou:.4f}" if iou is not None else ""
    print(f"{prefix} | Loss: {m['loss']:.4f}, Acc: {m['acc']:.1f}%, mAP: {m['mAP']:.1f}%{iou_str}")
    for i, cls in enumerate(CLASSES):
        if m['class_total'][i] > 0:
            print(f"  {cls}: {m['class_correct'][i]/m['class_total'][i]*100:.1f}%"
                  f" ({m['class_correct'][i]}/{m['class_total'][i]})")


def save_sample_image(crop, pred_idx, class_idx, confidence, out_path) -> None:
    correct = '✓' if pred_idx == class_idx else '✗'
    plt.figure(figsize=(5, 5))
    plt.imshow(crop)
    plt.title(f'ResNet18 | {correct}\nPredicted: {CLASSES[pred_idx]} ({confidence:.1f}%)\nGT: {CLASSES[class_idx]}')
    plt.axis('off')
    plt.savefig(str(out_path), bbox_inches='tight')
    plt.close()


class Classifier:
    def __init__(self, use_detector_boxes=False):
        self.use_detector_boxes = use_detector_boxes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.transform = transforms.Compose([
            transforms.Resize(CROP_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.train_dataset = AstroDataset('dataset/train/images', 'dataset/train/labels', self.transform)
        self.valid_dataset = AstroDataset('dataset/valid/images', 'dataset/valid/labels', self.transform)
        print(f"Train: {len(self.train_dataset)} | Val: {len(self.valid_dataset)}")

        self.train_loader = DataLoader(self.train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                       num_workers=NUM_WORKERS, pin_memory=True)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                       num_workers=NUM_WORKERS, pin_memory=True)

        self.model     = ClassificationModel().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.loss_fn   = nn.CrossEntropyLoss(label_smoothing=0.1)

        self.preprocessor = ImagePreprocessor()
        self.detector = self._load_detector() if use_detector_boxes else None

    def _load_detector(self) -> Detector:
        d = Detector()
        d.path_to_weights = d.find_best_weights()
        d.model = YOLO(d.path_to_weights)
        return d

    def _load_model_weights(self) -> None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        model_path = OUTPUT_DIR / 'classification_model.pth'
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}. Train first.")
        self.model.load_state_dict(torch.load(str(model_path)))

    def _get_iou(self, samples) -> float | None:
        if not self.use_detector_boxes or self.detector is None:
            return None
        return compute_mean_iou(samples, self.preprocessor, self.detector.model)

    def _get_crop(self, img_path, box) -> Image.Image:
        img = Image.open(str(img_path))
        w, h = img.size
        if self.use_detector_boxes and self.detector is not None:
            img_bgr = cv2.imread(str(img_path))
            img_bgr = self.preprocessor.do_magic(img_bgr, show=False)
            results = self.detector.model.predict(img_bgr, conf=0.25, verbose=False)[0]
            if results.boxes:
                x1, y1, x2, y2 = map(int, results.boxes[0].xyxy[0].tolist())
                return img.crop((x1, y1, x2, y2))
        x1, y1, x2, y2 = yolo_box_to_xyxy(box, w, h)
        return img.crop((x1, y1, x2, y2))

    def _infer(self, crop: Image.Image) -> tuple[int, float]:
        tensor = self.transform(crop).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(tensor).squeeze()
        pred_idx   = output.argmax().item()
        confidence = torch.softmax(output, dim=0)[pred_idx].item() * 100
        return pred_idx, confidence

    def _save_samples(self, samples, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, (img_path, class_idx, box) in enumerate(samples):
            crop = self._get_crop(img_path, box)
            pred_idx, confidence = self._infer(crop)
            save_sample_image(crop, pred_idx, class_idx, confidence, out_dir / f'sample_{i}_{img_path.stem}.png')

    def _show_samples(self, samples, max_samples: int = 10) -> None:
        samples = random.sample(samples, min(max_samples, len(samples)))
        cols, rows = 5, 2
        fig, axes = plt.subplots(rows, cols, figsize=(15, 6))
        axes = axes.flatten()

        for i, (img_path, class_idx, box) in enumerate(samples):
            crop = self._get_crop(img_path, box)
            pred_idx, confidence = self._infer(crop)
            correct = '✓' if pred_idx == class_idx else '✗'
            axes[i].imshow(crop)
            axes[i].set_title(
                f"True: {CLASSES[class_idx]}\nPred: {CLASSES[pred_idx]} {correct}\nConf: {confidence:.1f}%",
                fontsize=9
            )
            axes[i].axis('off')

        for i in range(len(samples), len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

    def _show_confusion_matrix(self, cm) -> None:
        """Display confusion matrix as a heatmap"""
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.xticks(np.arange(len(CLASSES)), CLASSES, rotation=45)
        plt.yticks(np.arange(len(CLASSES)), CLASSES)
        plt.tight_layout()
        plt.show()

    def _run_eval(self, loader, samples, csv_file: str) -> dict:
        self._load_model_weights()
        self.model.eval()
        m   = compute_metrics(loader, self.model, self.loss_fn, self.device)
        iou = self._get_iou(samples)
        print_metrics("Results", m, iou=iou)
        row    = [f"{m['loss']:.4f}", f"{m['acc']:.1f}", f"{m['mAP']:.1f}"] + [f"{ap:.4f}" for ap in m['ap_per_class']]
        header = RESULT_CSV_HEADER + (['IoU'] if iou is not None else [])
        if iou is not None:
            row.append(f"{iou:.4f}")
        append_csv(csv_file, header, row)
        return m

    def train(self) -> None:
        print("──────────────────[ Training ]──────────────────")
        best_val_acc, patience_counter = 0.0, 0

        for epoch in range(NUM_EPOCHS):
            self.model.train()
            train_loss, correct, total = 0.0, 0, 0

            for crops, labels in self.train_loader:
                crops, labels = crops.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                preds = self.model(crops)
                loss  = self.loss_fn(preds, labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                correct    += (preds.argmax(dim=1) == labels).sum().item()
                total      += labels.size(0)

            train_loss /= len(self.train_loader)
            train_acc   = correct / total * 100
            val_m       = compute_metrics(self.valid_loader, self.model, self.loss_fn, self.device)

            print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
            print(f"  Train | Loss: {train_loss:.4f}, Acc: {train_acc:.1f}%")
            print_metrics("  Val  ", val_m)

            append_csv('results_classifier_epochs.csv', EPOCH_CSV_HEADER,
                [epoch+1, f"{train_loss:.4f}", f"{train_acc:.1f}",
                 f"{val_m['loss']:.4f}", f"{val_m['acc']:.1f}", f"{val_m['mAP']:.1f}"]
                + [f"{ap:.4f}" for ap in val_m['ap_per_class']])

            if val_m['acc'] > best_val_acc:
                best_val_acc, patience_counter = val_m['acc'], 0
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                torch.save(self.model.state_dict(), str(OUTPUT_DIR / 'classification_model.pth'))
                print(f"Best Val Acc: {best_val_acc:.1f}% — saved")
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        print("──────────────────[ Done ]──────────────────")

    def validate(self) -> None:
        print("──────────────────[ Validation ]──────────────────")
        metrics = self._run_eval(self.valid_loader, self.valid_dataset.samples, 'results_classifier.csv')
        self._show_samples(self.valid_dataset.samples)
        self._show_confusion_matrix(metrics['confusion_matrix'])
        self._save_samples(random.sample(self.valid_dataset.samples, 10),
                           OUTPUT_DIR / "classifier_validation")

    def test(self, img_dir='dataset/test/images', label_dir='dataset/test/labels') -> None:
        print("──────────────────[ Test ]──────────────────")
        test_dataset = AstroDataset(img_dir, label_dir, self.transform)
        test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=NUM_WORKERS, pin_memory=True)
        print(f"Test samples: {len(test_dataset)}")
        test_metrics = self._run_eval(test_loader, test_dataset.samples, 'results_classifier_test.csv')
        self._show_samples(test_dataset.samples)
        self._show_confusion_matrix(test_metrics['confusion_matrix'])
        self._save_samples(random.sample(test_dataset.samples, min(10, len(test_dataset))),
                           OUTPUT_DIR / "classifier_test")