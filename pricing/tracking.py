"""Very small experiment log.

Every pipeline step appends one JSON line (params + metrics + git commit) to
outputs/runs.jsonl. If MLflow is installed and MLFLOW_TRACKING_URI is set,
the same run is also sent to MLflow. I kept MLflow optional so the project
runs without a tracking server.
"""
import json
import logging
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _flatten(d: dict, prefix: str = "") -> dict:
    flat = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            flat.update(_flatten(v, prefix=f"{key}."))
        else:
            flat[key] = v
    return flat


def log_run(step: str, params: dict, metrics: dict, out_dir: Path,
            artifacts: list[Path] | None = None) -> None:
    record = {
        "step": step,
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "params": _flatten(params),
        "metrics": metrics,
    }
    with open(Path(out_dir) / "runs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")

    if os.environ.get("MLFLOW_TRACKING_URI"):
        try:
            import mlflow
        except ImportError:
            log.warning("MLFLOW_TRACKING_URI is set but mlflow is not installed, skipping")
            return
        mlflow.set_experiment("dynamic-pricing")
        with mlflow.start_run(run_name=step):
            mlflow.log_params({k: str(v)[:250] for k, v in record["params"].items()})
            mlflow.log_metrics({k: float(v) for k, v in _flatten(metrics).items()
                                if isinstance(v, (int, float))})
            for path in artifacts or []:
                mlflow.log_artifact(str(path))
