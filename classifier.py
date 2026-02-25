import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# ──────────────────[ CONFIG ]──────────────────
# Training config
CROP_SIZE = (224, 224)          # ResNet expects 224x224
IMAGE_SIZE = (640, 640)         # original image size for cropping
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
NUM_WORKERS = 6
DROPOUT_RATE = 0.3
EARLY_STOPPING_PATIENCE = 5

# Classes
CLASSES = ['comet', 'galaxy', 'nebula', 'globular_cluster']
NUM_CLASSES = len(CLASSES)

# ──────────────────[ MODEL ]──────────────────
class ClassificationModel(nn.Module):
    def __init__(self) -> None:
        super(ClassificationModel, self).__init__()

        # Pretrained ResNet18 backbone
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        # Freeze backbone — only train classification head
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, NUM_CLASSES)  # 4 classes
        self.activation = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)                         # raw logits, CrossEntropyLoss handles softmax
        return x

# ──────────────────[ DATASET ]──────────────────
class AstroClassifyDataset(Dataset):
    def __init__(self, image_dir: str, label_dir: str, transform=None) -> None:
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.transform = transform
        self.samples = []  # list of (img_path, class_idx, box)

        for img_path in list(self.image_dir.glob('*.jpg')) + list(self.image_dir.glob('*.png')):
            label_path = self.label_dir / (img_path.stem + '.txt')
            if label_path.exists():
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                if lines:
                    parts = lines[0].strip().split()
                    if len(parts) >= 5:
                        class_idx = int(parts[0])
                        box = [float(p) for p in parts[1:5]]  # x_center, y_center, w, h
                        self.samples.append((img_path, class_idx, box))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, class_idx, box = self.samples[idx]
        img = Image.open(str(img_path)).convert('RGB')
        w, h = img.size

        # Crop using known bounding box
        x_c, y_c, bw, bh = box
        x1 = int((x_c - bw / 2) * w)
        y1 = int((y_c - bh / 2) * h)
        x2 = int((x_c + bw / 2) * w)
        y2 = int((y_c + bh / 2) * h)

        # Clamp to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        crop = img.crop((x1, y1, x2, y2))

        if self.transform:
            crop = self.transform(crop)

        return crop, class_idx

# ──────────────────[ MAIN ]──────────────────
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize(CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ResNet normalization
    ])

    train_dataset = AstroClassifyDataset('dataset/train/images', 'dataset/train/labels', transform)
    valid_dataset = AstroClassifyDataset('dataset/valid/images', 'dataset/valid/labels', transform)

    print(f"Train samples: {len(train_dataset)} | Val samples: {len(valid_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = ClassificationModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()  # handles softmax internally

    print("──────────────────[ Training started! ]──────────────────")
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}: ....")
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for crops, labels in train_loader:
            crops, labels = crops.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(crops)
            loss = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_correct += (preds.argmax(dim=1) == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total * 100

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        class_correct = [0] * NUM_CLASSES
        class_total = [0] * NUM_CLASSES

        with torch.no_grad():
            for crops, labels in valid_loader:
                crops, labels = crops.to(device), labels.to(device)
                preds = model(crops)
                loss = loss_fn(preds, labels)
                val_loss += loss.item()
                predicted = preds.argmax(dim=1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

                # Per-class accuracy
                for i in range(len(labels)):
                    class_total[labels[i]] += 1
                    if predicted[i] == labels[i]:
                        class_correct[labels[i]] += 1

        val_loss /= len(valid_loader)
        val_acc = val_correct / val_total * 100

        print(f'Epoch {epoch + 1}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.1f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.1f}%')

        # Per-class breakdown
        for i, cls in enumerate(CLASSES):
            if class_total[i] > 0:
                acc = class_correct[i] / class_total[i] * 100
                print(f'  {cls}: {acc:.1f}% ({class_correct[i]}/{class_total[i]})')

        # ──────────────────[ EARLY STOPPING ]──────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'classification_model.pth')
            print(f"New best Val Acc: {best_val_acc:.1f}% - model saved")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch + 1} — no improvement for {EARLY_STOPPING_PATIENCE} epochs")
                break

    print("──────────────────[ Training Finished! ]──────────────────")

    # ──────────────────[ VISUALIZATION ]──────────────────
    model.load_state_dict(torch.load('classification_model.pth'))
    model.eval()
    print("Visualizing predictions on validation samples...")

    with torch.no_grad():
        for i in range(5):
            img_path, class_idx, box = valid_dataset.samples[i]
            orig_img = Image.open(str(img_path)).convert('RGB')
            w, h = orig_img.size

            # Crop
            x_c, y_c, bw, bh = box
            x1 = int((x_c - bw / 2) * w)
            y1 = int((y_c - bh / 2) * h)
            x2 = int((x_c + bw / 2) * w)
            y2 = int((y_c + bh / 2) * h)
            crop = orig_img.crop((max(0, x1), max(0, y1), min(w, x2), min(h, y2)))

            # Predict
            input_tensor = transform(crop).unsqueeze(0).to(device)
            output = model(input_tensor).squeeze()
            pred_idx = output.argmax().item()
            confidence = torch.softmax(output, dim=0)[pred_idx].item() * 100

            correct = '✓' if pred_idx == class_idx else '✗'
            plt.figure(figsize=(5, 5))
            plt.imshow(crop)
            plt.title(
                f'ResNet18 Classifier | {correct}\n'
                f'Predicted: {CLASSES[pred_idx]} ({confidence:.1f}%)\n'
                f'Ground Truth: {CLASSES[class_idx]}'
            )
            plt.axis('off')
            plt.savefig(f'classification_{i}_{img_path.stem}.png', bbox_inches='tight')
            plt.show()