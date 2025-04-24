import csv
from collections import Counter

# Define the labels you're interested in
labels_of_interest = {"ADI", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"}

def count_labels(csv_path):
    label_counter = Counter()

    with open(csv_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            label = row['label']
            if label in labels_of_interest:
                label_counter[label] += 1

    # Print the counts for each label
    for label in sorted(labels_of_interest):
        print(f"{label}: {label_counter[label]}")

# Example usage
csv_file_path = '/home/tim/ICL-VL/val_split.csv'
count_labels(csv_file_path)