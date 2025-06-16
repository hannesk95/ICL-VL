import pandas as pd

# Load the output CSV
df = pd.read_csv('data/glioma/binary/t1/glioma_dataset_all.csv')

# Check class distribution
class_counts = df['label'].value_counts()
min_count = class_counts.min()

# Sample min_count examples from each class
balanced_df = df.groupby('label').sample(n=min_count, random_state=42)

# Sort so class1 appears before class2
balanced_df = balanced_df.sort_values(by='label', ascending=True).reset_index(drop=True)

# Save to a new balanced CSV
balanced_df.to_csv('data/glioma/binary//t1/glioma_balanced_all.csv', index=False)

# Optional: Print final counts
print("Balanced class distribution:")
print(balanced_df['label'].value_counts())