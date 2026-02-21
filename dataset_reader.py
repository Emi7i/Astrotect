import matplotlib.pyplot as plt
from pathlib import Path

class DatasetReader:
    def __init__(self, dataset_path: str) -> None:
        self.dataset_path = Path(dataset_path).resolve()
        self.files = ["train", "valid", "test"]
        self.class_map = {0: "Comets", 1: "Galaxies", 2: "Nebulae", 3: "Globular Clusters"}

        self.class_counter = {}  
        self.percentages = {}

    def calculate_distribution(self) -> None:
        for i, current_file in enumerate(self.files):
            label_path = self.dataset_path / current_file / "labels"
            self.class_counter[current_file] = [0, 0, 0, 0]
            overall_class_counter = 0

            for file_path in label_path.glob("*.txt"):
                with open(file_path, "r") as file:
                    for line in file:
                        try:
                            class_id = int(line.split()[0])
                            if 0 <= class_id <= 3:
                                self.class_counter[current_file][class_id] += 1
                                overall_class_counter += 1
                        except (ValueError, IndexError):
                            continue

            self.percentages[current_file] = (
                [(count * 100 / overall_class_counter) for count in self.class_counter[current_file]]
                if overall_class_counter > 0 else [0, 0, 0, 0]
            )
            self.print_results(current_file, overall_class_counter)

    def print_results(self, current_file: str, overall_class_counter: int) -> None:
        print(f"------- DISTRIBUTION FOR {current_file.upper()} -------")
        classes = ["Comets", "Galaxies", "Nebulae", "Globular star clusters"]
        for i, name in enumerate(classes):
            count = self.class_counter[current_file][i]
            percentage = (count * 100 / overall_class_counter) if overall_class_counter > 0 else 0
            print(f"{name}: {count} ({percentage:.2f}%)")
        print()

    def visualise_distribution(self) -> None:
        # Derive totals per split directly from class_counter
        sizes = [sum(self.class_counter.get(s, [0, 0, 0, 0])) for s in self.files]
        
        # Warn if data isn't loaded
        if sum(sizes) == 0:
            print("[WARN] No data found !")
            return

        plt.figure(figsize=(8, 6))
        plt.pie(sizes, labels=self.files, colors=['blue', 'purple', 'pink'], autopct='%1.2f%%')
        plt.axis('equal')
        plt.title('Dataset Distribution: Train/Valid/Test Split')
        plt.show()

    def show_distribution(self) -> None:
        self.calculate_distribution()
        self.visualise_distribution()


if __name__ == "__main__":
    analyzer = DatasetReader("dataset")
    analyzer.show_distribution()