import yaml

def load_config(path: str = "configs/tumor/one_shot.yaml") -> dict:
    """Load YAML and resolve template variables."""
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    # ── resolve ${placeholders} in save_path ───────────────────────────────
    save_path = (
        config["data"]["save_path"]
        .replace("${project}",            config["project"])
        .replace("${mode}",               config["mode"])
        .replace("${classification_type}", config["classification_type"])
    )
    config["data"]["save_path"] = save_path

    # ── classification defaults ────────────────────────────────────────────
    if "classification" not in config:
        config["classification"] = {}
    config["classification"]["type"]   = config.get("classification_type", "binary")
    config["classification"].setdefault("labels", ["Tumor", "No Tumor"])

    # ── optional evaluation labels ----------------------------------------
    config["data"].setdefault("labels_path", None)

    # ── NEW: few-shot sampling knobs --------------------------------------
    config["data"].setdefault("sampling_strategy", "random")   # "random" | "knn"
    config["data"].setdefault("knn_csv_path", None)           # path to neighbour CSV
    config["data"].setdefault("knn_top_k", None)              # int, defaults to num_shots

    return config