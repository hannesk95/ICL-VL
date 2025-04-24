#!/usr/bin/env python
import os
import csv
import json

def generate_binary_labels_from_csv(csv_path, output_path):
    """
    Reads a CSV file with headers:
      fname,label,path

    For binary classification, this function maps:
      - "TUM" -> "Tumor"
      - Any other label -> "No Tumor"

    It then generates a JSON file mapping each filename to its binary label.
    """
    labels = {}
    
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            fname = row["fname"].strip()
            orig_label = row["label"].strip()  # e.g., "TUM", "ADI", "STR", etc.
            
            # Map only "TUM" to "Tumor"; all other values to "No Tumor"
            if orig_label == "TUM":
                binary_label = "Tumor"
            else:
                binary_label = "No Tumor"
            
            labels[fname] = binary_label

    with open(output_path, "w") as f:
        json.dump(labels, f, indent=2)

    print(f"[INFO] Generated binary labels for {len(labels)} images from '{csv_path}'.")
    print(f"[INFO] Saved labels to: '{output_path}'")


if __name__ == "__main__":
    # Hard-coded paths for the new dataset CSV and the output JSON file.
    csv_path = "data/CRC100K/binary/CRC100K_test_dataset.csv"
    output_path = "data/CRC100K/binary/labels.json"
    
    generate_binary_labels_from_csv(csv_path, output_path)