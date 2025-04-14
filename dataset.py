import os
import csv
from torch.utils.data import Dataset
from PIL import Image

class CSVDataset(Dataset):
    def __init__(self, csv_path, transform=None, label_filter=None):
        self.csv_path = csv_path
        self.transform = transform
        self.label_filter = label_filter
        self.samples = []

        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip the header row if present
            for row in reader:
                fname, label, path = row
                if self.label_filter is None or label == self.label_filter:
                    self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_path, label