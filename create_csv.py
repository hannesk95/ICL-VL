import os
import csv
import re

# Define the directory to scan
base_dir = "data/tumor/all"
output_csv = "tumor_all_dataset.csv"

def natural_key(filename):
    """
    Extracts a key for natural sorting from the filename like P1, N2, etc.
    """
    match = re.match(r'([PN])(\d+)', filename)
    if match:
        prefix, number = match.groups()
        return (prefix, int(number))
    return (filename, 0)

# Collect data
data = []
for fname in os.listdir(base_dir):
    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        if fname.startswith('P'):
            label = 'positive'
        elif fname.startswith('N'):
            label = 'negative'
        else:
            continue  # Skip files that don't start with P or N
        
        full_path = os.path.join(base_dir, fname)
        data.append((fname, label, full_path))

# Sort data using natural order
data.sort(key=lambda x: natural_key(x[0]))

# Write to CSV
with open(output_csv, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['fname', 'label', 'path'])
    writer.writerows(data)

print(f"CSV file created: {output_csv} with {len(data)} sorted entries.")