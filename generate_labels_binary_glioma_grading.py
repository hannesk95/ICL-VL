import pandas as pd
import json

# Load the CSV file
df = pd.read_csv("data/glioma/UCSF-PDGM-metadata_v2.csv")

# Function to map grade to class
def map_grade_to_class(grade):
    if pd.isna(grade):
        return None
    if grade == 2:
        return "class1"
    elif grade == 4:
        return "class2"
    return None  # Skip other grades

# Separate dictionaries for each class
class1_labels = {}
class2_labels = {}

for _, row in df.iterrows():
    label_class = map_grade_to_class(row["WHO CNS Grade"])
    if label_class:
        image_filename = f"{label_class}_{row['ID']}.png"
        if label_class == "class1":
            class1_labels[image_filename] = label_class
        elif label_class == "class2":
            class2_labels[image_filename] = label_class

# Combine with class1 entries first
sorted_labels = {**class1_labels, **class2_labels}

# Write to JSON
with open("labels.json", "w") as f:
    json.dump(sorted_labels, f, indent=2)

print("labels.json created successfully.")