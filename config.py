import yaml

def load_config(path="configs/tumor/one_shot.yaml"):
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    save_path = config["data"]["save_path"]
    save_path = (
        save_path
        .replace("${project}", config["project"])
        .replace("${mode}", config["mode"])
        .replace("${classification_type}", config["classification_type"])
    )

    # Add this if modality is defined
    if "${modality}" in save_path:
        modality = config.get("modality", "t1")  # Default to t1 if not set
        save_path = save_path.replace("${modality}", modality)

    config["data"]["save_path"] = save_path

    if "classification" not in config:
        config["classification"] = {}
    config["classification"]["type"] = config.get("classification_type", "binary")
    config["classification"].setdefault("labels", ["Tumor", "No Tumor"])

    config["data"].setdefault("labels_path", None)
    config.setdefault("sampling", {})
    config["sampling"].setdefault("strategy", "random")   # "random" | "knn"
    config["sampling"].setdefault("knn_csv", None)
    config["sampling"].setdefault("anchors", {})
    config["sampling"].setdefault("num_neighbors", 2)

    return config