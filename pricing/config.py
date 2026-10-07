from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = ROOT / "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # make every path absolute so scripts work from any working directory
    cfg["data"]["raw_dir"] = ROOT / cfg["data"]["raw_dir"]
    for key, value in cfg["paths"].items():
        cfg["paths"][key] = ROOT / value
        cfg["paths"][key].mkdir(parents=True, exist_ok=True)
    return cfg
