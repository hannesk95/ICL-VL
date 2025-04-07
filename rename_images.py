import os

def rename_images_sequentially(folder_path, prefix):
    # Get all .jpg files
    images = [f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')]
    
    # Sort for consistency
    images.sort()

    # Temporarily rename to avoid conflicts
    temp_names = []
    for idx, filename in enumerate(images):
        temp_name = f"temp_{idx}.jpg"
        os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, temp_name))
        temp_names.append(temp_name)

    # Rename to final names sequentially
    for idx, temp_name in enumerate(temp_names, start=1):
        new_name = f"{prefix}{idx}.jpg"
        os.rename(os.path.join(folder_path, temp_name), os.path.join(folder_path, new_name))

if __name__ == "__main__":
    rename_images_sequentially("data/positive", "P")
    rename_images_sequentially("data/negative", "N")