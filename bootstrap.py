#!/usr/bin/env python
"""
bootstrap.py – resampling wrapper for the VLM pipeline
=====================================================

• Mirrors the normal results/… directory structure under results_boot/…
• Prints config summary (backend, sampling, modality, K shots, prompt, N runs)
• Runs main.py N times with different seeds, timing each replica
• Saves per-run CSV and JSON summary

Run with zero flags:
    $ python bootstrap.py
Optional overrides:
    $ python bootstrap.py --config my.yaml --runs 50 --base-seed 7
"""
from __future__ import annotations
import subprocess, json, statistics, sys, re, argparse, csv, datetime, os, time
from pathlib import Path
from typing import Dict, List, Any

from config import load_config

# ───────────────────────── USER-EDITABLE CONSTANTS ─────────────────────────
CONFIG_PATH = "configs/glioma/binary/t2/three_shot.yaml"
NUM_RUNS    = 3
BASE_SEED   = 5
METRIC_KEYS = ["accuracy", "mcc", "auc"]    # fields inside *_metrics.json
# ────────────────────────────────────────────────────────────────────────────

_METRICS_RE = re.compile(r"\b\S+_metrics\.json\b")   # locate *_metrics.json path


# ╭──────────────────────────────────────────────────────────────────╮
# │ Helper functions                                                │
# ╰──────────────────────────────────────────────────────────────────╯
def _locate_metrics_path(text: str) -> Path:
    m = _METRICS_RE.search(text)
    if not m:
        raise RuntimeError("Could not find *_metrics.json in main.py output.")
    return Path(m.group(0)).expanduser().resolve()


def _boot_dir_from_cfg(cfg: Dict[str, Any]) -> Path:
    """Replace the first 'results' in save_path with 'results_boot'."""
    save_path = Path(cfg["data"]["save_path"]).expanduser()
    parts = list(save_path.parts)
    try:
        parts[parts.index("results")] = "results_boot"
    except ValueError:
        parts = ["results_boot"] + parts
    return Path(*parts)


def _print_cfg_summary(cfg: Dict[str, Any], n_runs: int) -> None:
    backend  = cfg.get("model", {}).get("backend", "unknown")
    sampling = cfg.get("sampling", {}).get("strategy", "unknown")
    modality = cfg.get("modality", "unknown")
    shots_k  = cfg.get("data", {}).get("num_shots", "n/a")
    prompt   = cfg.get("user_args", {}).get("prompt_path", "n/a")
    print("──────────────── Config summary ────────────────")
    print(f"Model backend     : {backend}")
    print(f"Sampling strategy : {sampling}")
    print(f"Modality          : {modality}")
    print(f"Few-shot (K)      : {shots_k}")
    print(f"Prompt file       : {prompt}")
    print(f"Bootstrap runs (N): {n_runs}")
    print("────────────────────────────────────────────────")


# bootstrap.py  ─────────────────────────────────────────────────────────
def _run_once(seed: int, cfg_path: str) -> dict[str, float | None]:
    """Run main.py once and return metrics + elapsed time (prints child errors)."""
    print(f"[BOOT] seed={seed}")
    t0 = time.time()

    try:
        # ← unchanged call
        cp = subprocess.run(
            [sys.executable, "main.py", "--config", cfg_path, "--seed", str(seed)],
            text=True,
            capture_output=True,
            check=True,          # raises CalledProcessError on non-zero exit
        )
    except subprocess.CalledProcessError as e:
        # NEW: show the traceback from the failing replica for easy debugging
        print("\n──────── stderr from main.py ────────")
        print(e.stderr or "<no stderr>")
        print("─────────────────────────────────────\n")
        raise                    # re-raise so bootstrap stops (or remove if you
                                 # prefer to continue with the next seed)

    elapsed = time.time() - t0

    mp = _locate_metrics_path(cp.stdout + cp.stderr)
    with open(mp, "r") as f:
        j = json.load(f)

    res = {"seed": seed, "time_sec": elapsed} | {k: j.get(k) for k in METRIC_KEYS}
    mtxt = ", ".join(
        f"{k}={res[k]:.4f}" if res[k] is not None else f"{k}=n/a"
        for k in METRIC_KEYS
    )
    print(f"  ↪ {mtxt}   |   {elapsed:.1f}s")
    return res
# ────────────────────────────────────────────────────────────────────────


def _aggregate(runs: List[Dict[str, float | None]]) -> Dict[str, Dict[str, float]]:
    agg: Dict[str, Dict[str, float]] = {}
    for k in METRIC_KEYS:
        vals = [d[k] for d in runs if d[k] is not None]
        if vals:
            agg[k] = {
                "mean": statistics.mean(vals),
                "std":  statistics.stdev(vals)    if len(vals) > 1 else 0.0,
                "var":  statistics.variance(vals) if len(vals) > 1 else 0.0,
            }
    return agg


def _print_summary(agg: Dict[str, Dict[str, float]], n_runs: int) -> None:
    print("\n──────── Bootstrap summary ────────")
    print(f"runs: {n_runs}")
    for k, s in agg.items():
        print(f"{k}: mean={s['mean']:.4f}  std={s['std']:.4f}  var={s['var']:.6f}")


def _save_outputs(
    runs: List[Dict[str, float | None]],
    agg: Dict[str, Dict[str, float]],
    boot_dir: Path,
) -> None:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path  = boot_dir / f"bootstrap_{ts}.csv"
    json_path = boot_dir / f"bootstrap_{ts}_summary.json"
    boot_dir.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "time_sec"] + METRIC_KEYS)
        writer.writeheader()
        writer.writerows(runs)

    with open(json_path, "w") as f:
        json.dump({"runs": len(runs), "aggregates": agg}, f, indent=2)

    print("\nSaved bootstrap results:")
    print(f"  • Per-run CSV : {csv_path}")
    print(f"  • Summary JSON: {json_path}")


# ╭──────────────────────────────────────────────────────────────────╮
# │ Main                                                             │
# ╰──────────────────────────────────────────────────────────────────╯
def main() -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config")
    p.add_argument("--runs", type=int)
    p.add_argument("--base-seed", type=int)
    args, _ = p.parse_known_args()

    cfg_path = args.config     or CONFIG_PATH
    n_runs   = args.runs       or NUM_RUNS
    seed0    = args.base_seed  or BASE_SEED

    cfg      = load_config(cfg_path)
    boot_dir = _boot_dir_from_cfg(cfg)

    _print_cfg_summary(cfg, n_runs)

    runs = [_run_once(seed0 + i, cfg_path) for i in range(n_runs)]
    agg  = _aggregate(runs)
    _print_summary(agg, n_runs)
    _save_outputs(runs, agg, boot_dir)


if __name__ == "__main__":
    main()