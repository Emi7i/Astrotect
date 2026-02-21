"""
Astrotect — Image Preprocessor
Takes a single image, removes background and normalizes brightness.
"""
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


class ImagePreprocessor:

    def do_magic(self, img: np.ndarray, show: bool = False) -> np.ndarray:
        """
        Args:
            img:  single BGR numpy array
            show: if True, display original vs processed
        Returns:
            Processed BGR numpy array
        """
        # 1. Background removal - subtract median sky glow per channel
        result = np.zeros_like(img, dtype=np.float32)
        for c in range(3):
            channel = img[:, :, c].astype(np.float32)
            result[:, :, c] = np.clip(channel - np.median(channel), 0, 255)

        # 2. Normalize brightness
        if result.max() > 0:
            result = result / result.max() * 255

        result = result.astype(np.uint8)

        if show:
            fig, axes = plt.subplots(1, 2, figsize=(14, 7))
            axes[0].imshow(cv2.cvtColor(img,    cv2.COLOR_BGR2RGB)); axes[0].set_title("Original");  axes[0].axis("off")
            axes[1].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)); axes[1].set_title("Processed"); axes[1].axis("off")
            plt.tight_layout(); plt.show()

        return result

    def process_image(self, img_or_path, show: bool = False) -> Image.Image:
        """PIL RGB in, PIL RGB out. If img_or_path is str, loads the image from path."""
        if isinstance(img_or_path, str):
            img = Image.open(img_or_path).convert("RGB")
        elif isinstance(img_or_path, Image.Image):
            img = img_or_path.convert("RGB")
        else:
            raise TypeError("img_or_path must be str or PIL.Image")
        arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        arr = self.do_magic(arr, show=show)
        return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))