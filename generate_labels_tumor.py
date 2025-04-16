import os
import json

def generate_labels_from_prefix(test_dir, output_path):
    labels = {}
    for fname in os.listdir(test_dir):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        prefix = fname[0].upper()
        if prefix == "P":
            label = "Tumor"
        elif prefix == "N":
            label = "No Tumor"
        else:
            label = "Unknown"  # fallback if naming doesn't match

        labels[fname] = label

    with open(output_path, "w") as f:
        json.dump(labels, f, indent=2)

    print(f"[INFO] Generated labels for {len(labels)} test images.")
    print(f"[INFO] Saved to: {output_path}")

if __name__ == "__main__":
    test_dir = "data/tumor/test"
    output_path = os.path.join(test_dir, "labels.json")
    generate_labels_from_prefix(test_dir, output_path)