from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image

from utils import yolo_box_to_xyxy


class AstroDataset(Dataset):
    def __init__(self, image_dir: str, label_dir: str, transform=None) -> None:
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.transform = transform
        self.samples = []

        for img_path in list(self.image_dir.glob('*.jpg')) + list(self.image_dir.glob('*.png')):
            label_path = self.label_dir / (img_path.stem + '.txt')
            if label_path.exists():
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                if lines:
                    parts = lines[0].strip().split()
                    if len(parts) >= 5:
                        class_idx = int(parts[0])
                        box = [float(p) for p in parts[1:5]]
                        self.samples.append((img_path, class_idx, box))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, class_idx, box = self.samples[idx]
        img = Image.open(str(img_path)).convert('RGB')
        w, h = img.size
        x1, y1, x2, y2 = yolo_box_to_xyxy(box, w, h)
        crop = img.crop((x1, y1, x2, y2))
        if self.transform:
            crop = self.transform(crop)
        return crop, class_idx