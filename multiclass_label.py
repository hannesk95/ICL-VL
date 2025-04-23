import pandas as pd
import json

# Load your dataset
df = pd.read_csv("/u/home/obt/ICL-VL/data/CRC100K/multiclass/CRC100K_test_dataset.csv")  # Replace with your actual CSV filename

# Mapping from short labels to full names
label_map = {
    "ADI": "Adipose",
    "BACK": "Background",
    "DEB": "Debris",
    "LYM": "Lymphocytes",
    "MUC": "Mucus",
    "MUS": "Smooth Muscle",
    "NORM": "Normal Colon Mucosa",
    "STR": "Cancer-Associated Stroma",
    "TUM": "Colorectal Adenocarcinoma Epithelium"
}

# Map labels to readable form
df['readable_label'] = df['label'].map(label_map)

# Create dictionary {filename: readable_label}
label_dict = dict(zip(df['fname'], df['readable_label']))

# Save to JSON
with open("readable_labels.json", "w") as f:
    json.dump(label_dict, f, indent=2)

print("JSON file created with readable labels.")