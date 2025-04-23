import os
import csv

# Define the target directory and output CSV file
image_dir = 'data/NCT-CRC-HE-100K/NCT-CRC-HE-100K-png'
output_csv = 'image_data.csv'

# Define valid label abbreviations
valid_labels = {
    'ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM'
}

# Prepare list to hold rows
rows = []

# Scan the directory for image files
for fname in os.listdir(image_dir):
    if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        prefix = fname.split('-')[0]  # Get the prefix before the first hyphen
        label = prefix if prefix in valid_labels else 'Unknown'
        rel_path = os.path.join(image_dir, fname)
        rows.append([fname, label, rel_path])

# Sort rows by label
rows.sort(key=lambda x: x[1])

# Insert header at the top
rows.insert(0, ['fname', 'label', 'path'])

# Write to CSV file
with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(rows)

print(f"CSV file '{output_csv}' has been created with {len(rows) - 1} image entries, sorted by label.")
