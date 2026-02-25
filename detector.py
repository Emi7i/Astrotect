"""
 Astrotect - Detector
 Detect objects of interest in images.
"""
import numpy as np
from pathlib import Path

class Detector:
    def __init__(self, image: np.ndarray) -> None:
        self.image = image
        self.class_map = {0: "Comets", 1: "Galaxies", 2: "Nebulae", 3: "Globular Clusters"}
        self.path_to_weights = ""

    def train(self):
