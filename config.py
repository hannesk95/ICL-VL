import yaml

def load_config(path="configs/tumor/one_shot.yaml"):
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    # Safely retrieve required fields
    project = config.get("project", "unknown_project")
    modality = config.get("modality", "t1")
    model_backend = config.get("model", {}).get("backend", "unknown_model")
    classification_type = config.get("classification_type", "binary")
    sampling_strategy = config.get("sampling", {}).get("strategy", "random")

    # Auto-generate mode from num_shots
    num_shots = config.get("data", {}).get("num_shots", 1)
    mode = f"{num_shots}_shot"
    config["mode"] = mode  # overwrite or set mode

    # Resolve save_path
    save_path = config["data"]["save_path"]
    save_path = (
        save_path
        .replace("${project}", project)
        .replace("${modality}", modality)
        .replace("${model}", model_backend)
        .replace("${classification_type}", classification_type)
        .replace("${sampling_strategy}", sampling_strategy)
        .replace("${mode}", mode)
    )

    config["data"]["save_path"] = save_path

    # Classification defaults
    if "classification" not in config:
        config["classification"] = {}
    config["classification"]["type"] = classification_type
    config["classification"].setdefault("labels", ["Tumor", "No Tumor"])

    config["data"].setdefault("labels_path", None)
    config.setdefault("sampling", {})
    config["sampling"].setdefault("strategy", "random")
    config["sampling"].setdefault("knn_csv", None)
    config["sampling"].setdefault("anchors", {})
    config["sampling"].setdefault("num_neighbors", 2)

    return config