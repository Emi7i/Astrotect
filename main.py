from detector import Detector
from classifier import Classifier

# Configuration
USE_DETECTOR_BOXES = True  # Set to True to use predicted boxes from detector for classifier validation

def train_detector() -> None:
    detector = Detector()
    detector.train()

def train_classifier() -> None:
    classifier = Classifier()
    classifier.train()

def validate() -> None:
    # Validate detector
    print("──────────────────[ VALIDATING DETECTOR ]──────────────────")
    detector = Detector()
    detector.validate()

    # Validate classifier
    print("──────────────────[ VALIDATING CLASSIFIER ]──────────────────")
    classifier = Classifier(use_detector_boxes=USE_DETECTOR_BOXES)
    classifier.validate()

def test() -> None:
    # Test detector
    print("──────────────────[ TESTING DETECTOR ]──────────────────")
    detector = Detector()
    detector.test()

    # Test classifier
    print("──────────────────[ TESTING CLASSIFIER ]──────────────────")
    classifier = Classifier(use_detector_boxes=USE_DETECTOR_BOXES)
    classifier.test()

if __name__ == "__main__":
    # Example usage
    # train_detector()
    # train_classifier()
    # validate()
    test()
    print("──────────────────[ DONE ]──────────────────")