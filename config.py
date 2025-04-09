import yaml

def load_config(path="configs/tumor/zero_shot.yaml"):
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    # Flatten any ${project}/${mode}/ style variables
    save_path = config["data"]["save_path"]
    save_path = save_path.replace("${project}", config["project"]).replace("${mode}", config["mode"])
    config["data"]["save_path"] = save_path

    return config