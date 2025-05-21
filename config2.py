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
    config["data"]["save_path"] = save_path

    if "classification" not in config:
        config["classification"] = {}
    config["classification"]["type"] = config.get("classification_type", "binary")
    config["classification"].setdefault("labels", ["Tumor", "No Tumor"])

    config["data"].setdefault("labels_path", None)
    
    config.setdefault("retrieval", {"enable": False})

    return config