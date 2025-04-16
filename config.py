# config.py
import yaml

def load_config(path="configs/tumor/one_shot.yaml"):
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    # Flatten any ${project}/${mode}/ style variables in save_path
    save_path = config["data"]["save_path"]
    save_path = save_path.replace("${project}", config["project"]).replace("${mode}", config["mode"])
    config["data"]["save_path"] = save_path

    # Set classification settings if not present
    if "classification" not in config:
        config["classification"] = {}
    config["classification"].setdefault("type", "binary")
    config["classification"].setdefault("labels", ["Tumor", "No Tumor"])

    # Add a key for ground-truth labels (path), if not provided, it can be None.
    config["data"].setdefault("labels_path", None)
    return config