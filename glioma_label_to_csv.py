import pandas as pd
import json
import re

# Load the original CSV
csv_df = pd.read_csv('same_rgb_images.csv')

# Load the JSON labels
with open('data/glioma/binary/t1/labels.json', 'r') as f:
    label_dict = json.load(f)

new_rows = []
for _, row in csv_df.iterrows():
    original_fname = row['fname']
    path = row['path'].replace('/home/tim/ICL-VL/', '')  # Normalize path

    # Extract the patient ID using regex
    match = re.match(r'(UCSF-PDGM-0*(\d+))', original_fname)
    if match:
        numeric_id = int(match.group(2))  # e.g., '0004' -> 4
        json_key = f"UCSF-PDGM-{numeric_id:03}.png"  # e.g., 4 -> UCSF-PDGM-004.png

        label = label_dict.get(json_key)
        if label:
            label_clean = label.replace(' ', '')  # e.g., "class 1" -> "class1"
            new_fname = f"{label_clean}_{json_key}"
            new_rows.append({
                'fname': new_fname,
                'label': label_clean,
                'path': path
            })

# Create DataFrame
output_df = pd.DataFrame(new_rows)

# Sort so class1 comes first, then class2
output_df = output_df.sort_values(by='label', ascending=True).reset_index(drop=True)

# Save to CSV
output_df.to_csv('output.csv', index=False)