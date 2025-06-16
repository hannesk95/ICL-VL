"""
Script: generate_labels_json.py

Description:
This script reads metadata from 'UCSF-PDGM-metadata_v2.csv' and creates a JSON file
named 'labels.json'. Each entry maps an image filename (based on the patient ID + '.png') 
to a class label:
- "class 1" for Glioblastoma
- "class 2" for Astrocytoma or Oligodendroglioma

Example output:
{
    "UCSF-PDGM-004.png": "class 1",
    "UCSF-PDGM-021.png": "class 2"
}
"""

import pandas as pd
import json

# Load the CSV file
df = pd.read_csv("data/glioma/UCSF-PDGM-metadata_v2.csv")

# Function to map diagnosis to class
def map_to_class(diagnosis):
    if pd.isna(diagnosis):
        return None
    if "Glioblastoma" in diagnosis:
        return "class1"
    elif "Astrocytoma" in diagnosis or "Oligodendroglioma" in diagnosis:
        return "class2"
    return None  # Skip unknown types

# Build the labels dictionary
labels = {}
for _, row in df.iterrows():
    label_class = map_to_class(row["Final pathologic diagnosis (WHO 2021)"])
    if label_class:
        image_filename = f"{label_class}_{row['ID']}.png"
        labels[image_filename] = label_class

# Write to JSON
with open("labels.json", "w") as f:
    json.dump(labels, f, indent=2)

print("labels.json created successfully.")