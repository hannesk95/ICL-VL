import pandas as pd
import json
import re

# Load files
csv_path = 'same_rgb_images.csv'
json_path = 'data/glioma/binary/t1/labels.json'

df = pd.read_csv(csv_path)
with open(json_path, 'r') as f:
    label_dict = json.load(f)

# Build map from normalized base ID
reverse_map = {}
for json_fname, label in label_dict.items():
    match = re.match(r'(?:class\d+_)?UCSF-PDGM-(\d+)', json_fname)
    if match:
        number = int(match.group(1))  # normalize
        base_id = f'UCSF-PDGM-{number}'
        reverse_map[base_id] = (label, json_fname)

# Match and build new rows
new_rows = []
for _, row in df.iterrows():
    match = re.search(r'UCSF-PDGM-(\d+)', row['fname'])
    if match:
        number = int(match.group(1))
        base_id = f'UCSF-PDGM-{number}'
        if base_id in reverse_map:
            label, json_fname = reverse_map[base_id]
            new_rows.append({
                'fname': json_fname,
                'label': label,
                'path': row['path']
            })

# Convert to DataFrame
output_df = pd.DataFrame(new_rows)

# Sort by 'label'
output_df = output_df.sort_values(by='label')

# Save result
output_df.to_csv('output.csv', index=False)

print(f"{len(new_rows)} rows matched, sorted by label, and written to output.csv")