import pandas as pd

input_csv = '/home/tim/ICL-VL/data/CRC100K/multiclass/CRC100K_all_data.csv'  # Change this to your actual input file
output_csv = 'balanced_output.csv'
target_count = 8763
labels = ["ADI", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
random_seed = 42

def balance_labels(input_csv, output_csv, target_count, labels):
    df = pd.read_csv(input_csv)

    balanced_dfs = []
    for label in labels:
        label_df = df[df['label'] == label]
        if len(label_df) >= target_count:
            label_df = label_df.sample(n=target_count, random_state=random_seed)
        # If less than target_count, keep as-is (optional: could upsample if needed)
        balanced_dfs.append(label_df)

    balanced_df = pd.concat(balanced_dfs)
    balanced_df.to_csv(output_csv, index=False)
    print(f"Balanced CSV saved to: {output_csv}")

# Run the function
balance_labels(input_csv, output_csv, target_count, labels)