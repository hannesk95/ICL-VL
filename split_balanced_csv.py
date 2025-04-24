import pandas as pd
from sklearn.model_selection import train_test_split

# Parameters
input_csv = '/home/tim/ICL-VL/data/CRC100K/multiclass/CRC100K_balanced_data.csv'
train_csv = 'train_split.csv'
val_csv = 'val_split.csv'
test_size = 0.2
random_seed = 42

def split_csv_by_label(input_csv, train_csv, val_csv, test_size=0.2):
    df = pd.read_csv(input_csv)
    
    # Stratified split to maintain label distribution
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df['label'],
        random_state=random_seed
    )
    
    # Save the splits
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    print(f"Training set saved to: {train_csv}")
    print(f"Validation set saved to: {val_csv}")

# Run the function
split_csv_by_label(input_csv, train_csv, val_csv, test_size)