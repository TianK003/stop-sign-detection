"""Sweep chain_eqe across k values for either the 256 or 192 rescaled-obj champion.

Per k value:
  0) (auto) If best_model_obj_k{k}.keras doesn't exist in the model's saved_models
     dir, generate it via `python scripts/rescale_obj_logits.py --k {k}`. The base
     `best_model.keras` must be present for this fallback to work; if it isn't, the
     k value is skipped with status=FAIL.
  1) Overwrite the target chain_eqe YAML so model.model_path points at
     best_model_obj_k{k}.keras. All other fields are copied verbatim from the
     canonical k=2 template, intentionally dropping any bogus yolo_anchors / wrong
     source training run that the in-place target might carry.
  2) Run `python stm32ai_main.py --config-name <target stem>` from object_detection/.
     Op-mode chain_eqe: eval float → quantize → eval int8.
  3) Find the new Hydra output dir, parse stm32ai_main.log for `float_model_map:`
     and `quantized_model_map:`, append a row to scripts/<model>_k_sweep_results.csv.

Resumable: rows already at status=OK in the per-model CSV are skipped on re-run.
Use --force to re-evaluate. Failures (subprocess crash, missing log, parse failure)
are recorded with status=FAIL or PARSE_FAIL and the sweep continues.

Run from repo root:
    python scripts/sweep_chain_eqe_k.py --model 256              # default k=3..10
    python scripts/sweep_chain_eqe_k.py --model 192
    python scripts/sweep_chain_eqe_k.py --model 256 --k-values 3 5 8
    python scripts/sweep_chain_eqe_k.py --model 192 --k-min 6 --k-max 10

Wall-clock estimate: ~5 min per k on CPU. Full k=3..10 sweep is ~40 min per model;
192 adds ~10 s per missing kN.keras for the auto-rescale step.
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OD_DIR = REPO_ROOT / "object_detection"
EXPERIMENTS_DIR = OD_DIR / "tf" / "src" / "experiments_outputs"
RESCALE_SCRIPT = REPO_ROOT / "scripts" / "rescale_obj_logits.py"

# Per-model configuration. Each entry locates the canonical k=2 template (used
# verbatim except for model_path), the target YAML that gets rewritten per k,
# the training run whose saved_models/ holds the .keras files, and the output CSV.
MODELS = {
    "256": {
        "template": OD_DIR / "my_chain_eqe_obj_rescaled_k2_256_default.yaml",
        "target": OD_DIR / "my_chain_eqe_obj_rescaled_256.yaml",
        "training_dir": "2026_05_08_20_48_06",
        "results_csv": REPO_ROOT / "scripts" / "256_k_sweep_results.csv",
    },
    "192": {
        "template": OD_DIR / "my_chain_eqe_obj_rescaled_k2_192v1.yaml",
        "target": OD_DIR / "my_chain_eqe_obj_rescaled_192v1.yaml",
        "training_dir": "2026_05_07_10_02_38",
        "results_csv": REPO_ROOT / "scripts" / "192_k_sweep_results.csv",
    },
}

FLOAT_MAP_RE = re.compile(r"float_model_map:\s*([\d.]+)")
INT8_MAP_RE = re.compile(r"quantized_model_map:\s*([\d.]+)")
MODEL_PATH_RE = re.compile(r"^(\s*model_path:\s*).+$", re.MULTILINE)


def saved_models_dir(model: str) -> Path:
    return EXPERIMENTS_DIR / MODELS[model]["training_dir"] / "saved_models"


def kn_keras(model: str, k: int) -> Path:
    return saved_models_dir(model) / f"best_model_obj_k{k}.keras"


def ensure_kn_keras(model: str, k: int) -> Path:
    """Return the path to best_model_obj_k{k}.keras, generating it if missing.

    Generation invokes scripts/rescale_obj_logits.py against the run's base
    best_model.keras. If the base is also missing, raises SystemExit — there's
    no way to fabricate the rescaled weights without it.
    """
    target = kn_keras(model, k)
    if target.exists():
        return target

    base = saved_models_dir(model) / "best_model.keras"
    if not base.exists():
        raise SystemExit(
            f"Cannot generate {target.name}: base {base} is also missing. "
            f"You will need to re-train the {model} model or copy the base .keras in "
            f"from another location."
        )
    print(f"  generating {target.name} via rescale_obj_logits.py (k={k}) ...")
    cmd = [sys.executable, str(RESCALE_SCRIPT),
           "--in", str(base), "--out", str(target), "--k", str(k)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not target.exists():
        raise SystemExit(
            f"rescale_obj_logits.py failed (exit {proc.returncode}). "
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return target


def write_config_for_k(model: str, k: int, keras_path: Path) -> Path:
    """Overwrite MODELS[model]['target'] with model_path swapped to keras_path.

    All other YAML fields come verbatim from MODELS[model]['template'] — the
    canonical k=2 config that produced the deployed model. This intentionally
    drops any drifted contents from the in-place target.
    """
    template = MODELS[model]["template"]
    target = MODELS[model]["target"]
    if not template.exists():
        raise SystemExit(f"Template config not found: {template}")

    text = template.read_text(encoding="utf-8")
    rel_keras = (
        f"./tf/src/experiments_outputs/{MODELS[model]['training_dir']}"
        f"/saved_models/{keras_path.name}"
    )
    new_text, n = MODEL_PATH_RE.subn(rf"\g<1>{rel_keras}", text, count=1)
    if n != 1:
        raise SystemExit(
            f"Did not find a unique model_path: line in {template} (got {n} matches). "
            f"Refusing to rewrite to avoid corrupting the config."
        )
    target.write_text(new_text, encoding="utf-8")
    return target


def newest_experiments_dir_after(t0: datetime) -> Path | None:
    if not EXPERIMENTS_DIR.exists():
        return None
    candidates = []
    for sub in EXPERIMENTS_DIR.iterdir():
        if not sub.is_dir():
            continue
        try:
            ts = datetime.strptime(sub.name, "%Y_%m_%d_%H_%M_%S")
        except ValueError:
            continue
        if ts >= t0:
            candidates.append((ts, sub))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def parse_log_for_maps(log_path: Path) -> tuple[float | None, float | None]:
    if not log_path.exists():
        return None, None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    fm = FLOAT_MAP_RE.search(text)
    qm = INT8_MAP_RE.search(text)
    return (float(fm.group(1)) if fm else None,
            float(qm.group(1)) if qm else None)


def load_done_ks(csv_path: Path) -> set[int]:
    if not csv_path.exists():
        return set()
    done: set[int] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") == "OK":
                try:
                    done.add(int(row["k"]))
                except (KeyError, ValueError):
                    continue
    return done


def append_row(csv_path: Path, row: dict) -> None:
    new_file = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["k", "run_dir", "float_map", "int8_map",
                                          "status", "error", "completed_at"])
        if new_file:
            w.writeheader()
        w.writerow(row)


def run_one_k(model: str, k: int) -> dict:
    print(f"\n{'=' * 78}\n  model = {model}   k = {k}\n{'=' * 78}")
    try:
        keras_path = ensure_kn_keras(model, k)
        target_cfg = write_config_for_k(model, k, keras_path)
    except SystemExit as e:
        return {"k": k, "run_dir": "", "float_map": "", "int8_map": "",
                "status": "FAIL", "error": f"setup: {e}",
                "completed_at": datetime.now().isoformat(timespec="seconds")}

    print(f"  rescaled model: {keras_path.relative_to(REPO_ROOT)}")
    print(f"  config:         {target_cfg.relative_to(REPO_ROOT)}")
    print(f"  invoking stm32ai_main.py (this takes ~5 min on CPU) ...")

    t0 = datetime.now()
    config_name = target_cfg.stem
    try:
        proc = subprocess.run(
            [sys.executable, "stm32ai_main.py", "--config-name", config_name],
            cwd=str(OD_DIR),
            check=False,
        )
    except Exception as e:
        return {"k": k, "run_dir": "", "float_map": "", "int8_map": "",
                "status": "FAIL", "error": f"subprocess: {e}",
                "completed_at": datetime.now().isoformat(timespec="seconds")}

    if proc.returncode != 0:
        return {"k": k, "run_dir": "", "float_map": "", "int8_map": "",
                "status": "FAIL", "error": f"exit code {proc.returncode}",
                "completed_at": datetime.now().isoformat(timespec="seconds")}

    run_dir = newest_experiments_dir_after(t0)
    if run_dir is None:
        return {"k": k, "run_dir": "", "float_map": "", "int8_map": "",
                "status": "PARSE_FAIL", "error": "no new experiments_outputs/ dir found",
                "completed_at": datetime.now().isoformat(timespec="seconds")}

    log_path = run_dir / "stm32ai_main.log"
    float_map, int8_map = parse_log_for_maps(log_path)
    if float_map is None or int8_map is None:
        return {"k": k, "run_dir": str(run_dir.relative_to(REPO_ROOT)), "float_map": "",
                "int8_map": "", "status": "PARSE_FAIL",
                "error": f"could not parse mAPs from {log_path.name}",
                "completed_at": datetime.now().isoformat(timespec="seconds")}

    print(f"  -> float mAP: {float_map:.1f}    int8 mAP: {int8_map:.1f}")
    print(f"  -> hydra dir: {run_dir.relative_to(REPO_ROOT)}")
    return {"k": k, "run_dir": str(run_dir.relative_to(REPO_ROOT)),
            "float_map": f"{float_map:.2f}", "int8_map": f"{int8_map:.2f}",
            "status": "OK", "error": "",
            "completed_at": datetime.now().isoformat(timespec="seconds")}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=sorted(MODELS.keys()), required=True,
                   help="which champion family to sweep (256 or 192)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--k-values", nargs="+", type=int,
                   help="explicit list of k values (overrides --k-min/--k-max)")
    p.add_argument("--k-min", type=int, default=3, help="inclusive (default 3)")
    p.add_argument("--k-max", type=int, default=10, help="inclusive (default 10)")
    p.add_argument("--force", action="store_true",
                   help="re-run k values that already have status=OK in the CSV")
    args = p.parse_args()

    model = args.model
    cfg = MODELS[model]
    ks = args.k_values if args.k_values else list(range(args.k_min, args.k_max + 1))
    done = set() if args.force else load_done_ks(cfg["results_csv"])
    to_run = [k for k in ks if k not in done]
    skip = [k for k in ks if k in done]

    print(f"Sweep plan: model={model}  k ∈ {ks}")
    if skip:
        print(f"  already done (status=OK in {cfg['results_csv'].relative_to(REPO_ROOT)}): {skip}")
    print(f"  will run: {to_run}")
    if not to_run:
        print("Nothing to do. Use --force to re-run completed k values.")
        return

    print(f"\nResults CSV: {cfg['results_csv'].relative_to(REPO_ROOT)}")
    print(f"Template:    {cfg['template'].relative_to(REPO_ROOT)}")
    print(f"Target cfg:  {cfg['target'].relative_to(REPO_ROOT)}\n")

    if cfg["target"].exists():
        backup = cfg["target"].with_suffix(".yaml.before_sweep")
        if not backup.exists():
            shutil.copy2(cfg["target"], backup)
            print(f"Backed up existing {cfg['target'].name} → {backup.name}\n")

    for k in to_run:
        row = run_one_k(model, k)
        append_row(cfg["results_csv"], row)

    print(f"\n{'=' * 78}\n  SWEEP COMPLETE\n{'=' * 78}")
    print(f"  results: {cfg['results_csv'].relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
