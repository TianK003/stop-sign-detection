"""Generate chain_eqe YAML configs for the Pareto sweep over k = {0, 2, 5}.

Given a trained model's run directory (the timestamped output of a chain_tqe run
under object_detection/tf/src/experiments_outputs/), this writes one chain_eqe
config per k value, all pointing at the same base model (k=0 -> best_model.keras,
k=N -> best_model_obj_kN.keras).

Output naming is normalized to my_chain_eqe_k{K}_{name}.yaml so the
run_pareto_sweep.ps1 orchestration script can reference them with a consistent
pattern. The .keras files for k>0 must be created separately via
rescale_obj_logits.py BEFORE running chain_eqe -- this script prints the exact
rescale commands you need to run.

Examples
--------

After your 256_default_val_map training finishes:

    python scripts/generate_chain_eqe_configs.py \\
        --name 256_default_val_map \\
        --timestamp 2026_05_09_15_30_00 \\
        --resolution 256 \\
        --quant-split 0.10

This writes:
    object_detection/my_chain_eqe_k0_256_default_val_map.yaml
    object_detection/my_chain_eqe_k2_256_default_val_map.yaml
    object_detection/my_chain_eqe_k5_256_default_val_map.yaml

And prints the rescale_obj_logits.py commands plus a PowerShell snippet you can
paste into the $Models array in scripts/run_pareto_sweep.ps1.

For a non-standard model path, pass --model-path instead of --timestamp:

    python scripts/generate_chain_eqe_configs.py \\
        --name some_custom \\
        --model-path object_detection/MODELS/some_model.keras \\
        --resolution 192
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OBJ_DET_DIR = REPO_ROOT / "object_detection"

CONFIG_TEMPLATE = """\
general:
  project_name: stop_sign_st_yololcv1_{name}_k{k}
  display_figures: false
  gpu_memory_limit: 6
  num_threads_tflite: 8
  global_seed: 127

operation_mode: chain_eqe

model:
  model_path: {model_path}
  model_type: st_yololcv1
  input_shape: ({res},{res},3)

dataset:
  format: coco
  dataset_name: coco
  class_names: [stop_sign]
  test_images_path: ./datasets/coco_stop_sign/val/data
  test_annotations_path: ./datasets/coco_stop_sign/val/labels.json
  test_path: ./datasets/coco_stop_sign/train/tfs_labels/val
  quantization_path: ./datasets/coco_stop_sign/train/data
  quantization_split: {quant_split}

preprocessing:
  rescaling:
    scale: 1/255
    offset: 0
  resizing:
    aspect_ratio: fit
    interpolation: nearest
  color_mode: rgb

postprocessing:
  confidence_thresh: 0.001
  NMS_thresh: 0.5
  IoU_eval_thresh: 0.5
  plot_metrics: true
  max_detection_boxes: 10

quantization:
  quantizer: TFlite_converter
  quantization_type: PTQ
  quantization_input_type: uint8
  quantization_output_type: float
  export_dir: quantized_models

mlflow:
  uri: ./tf/src/experiments_outputs/mlruns

hydra:
  run:
    dir: ./tf/src/experiments_outputs/${{now:%Y_%m_%d_%H_%M_%S}}
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--name",
        required=True,
        help="Short identifier for the model variant (e.g. '192v1', '256_default_val_map'). "
             "Used in project_name and output filenames. Must match [a-zA-Z0-9_].",
    )
    src_group = p.add_mutually_exclusive_group(required=True)
    src_group.add_argument(
        "--timestamp",
        help="Hydra run-dir timestamp like '2026_05_09_15_30_00'. The script assumes the "
             "model lives at object_detection/tf/src/experiments_outputs/<timestamp>/saved_models/best_model.keras",
    )
    src_group.add_argument(
        "--model-path",
        help="Explicit relative path to best_model.keras (relative to object_detection/). "
             "Use when the model isn't under the standard tf/src/experiments_outputs tree.",
    )
    p.add_argument(
        "--resolution",
        type=int,
        choices=[192, 224, 256],
        required=True,
        help="Model input resolution. 192 / 224 / 256.",
    )
    p.add_argument(
        "--quant-split",
        type=float,
        default=0.05,
        help="quantization_split fraction (default 0.05 — matches 192v1 and 256_default).",
    )
    p.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[0, 2, 5],
        help="k values to generate configs for (default: 0 2 5).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=OBJ_DET_DIR,
        help="Where to write the configs (default: object_detection/).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing config files (default: skip and warn).",
    )
    return p.parse_args()


def validate_name(name: str) -> None:
    if not re.match(r"^[A-Za-z0-9_]+$", name):
        sys.exit(f"--name must match [A-Za-z0-9_]+; got '{name}'")


def resolve_base_keras_path(args: argparse.Namespace) -> tuple[str, Path]:
    """Return (rel_path_for_yaml, abs_path_for_existence_check).

    The YAML config's model_path is relative to object_detection/, while the
    existence check needs an absolute path.
    """
    if args.timestamp:
        rel = f"./tf/src/experiments_outputs/{args.timestamp}/saved_models/best_model.keras"
    else:
        rel = args.model_path
        if not rel.startswith("./"):
            rel = "./" + rel.lstrip("/")
    abs_path = (OBJ_DET_DIR / rel.lstrip("./")).resolve()

    if not rel.endswith("best_model.keras"):
        print(
            f"[WARN] base model path doesn't end with 'best_model.keras': {rel}\n"
            f"[WARN]   the rescaled k=N paths assume that filename and will be wrong",
            file=sys.stderr,
        )
    return rel, abs_path


def keras_path_for_k(base_rel: str, k: int) -> str:
    if k == 0:
        return base_rel
    return base_rel.replace("best_model.keras", f"best_model_obj_k{k}.keras")


def main() -> None:
    args = parse_args()
    validate_name(args.name)

    base_rel, base_abs = resolve_base_keras_path(args)
    if not base_abs.exists():
        print(f"[WARN] base model not found at {base_abs}", file=sys.stderr)
        print("[WARN] continuing — configs will be written but chain_eqe will fail until the model exists",
              file=sys.stderr)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    skipped: list[Path] = []

    for k in args.k:
        out_path = args.output_dir / f"my_chain_eqe_k{k}_{args.name}.yaml"
        if out_path.exists() and not args.force:
            print(f"[SKIP] {out_path} already exists (use --force to overwrite)")
            skipped.append(out_path)
            continue

        keras_rel = keras_path_for_k(base_rel, k)
        content = CONFIG_TEMPLATE.format(
            name=args.name,
            k=k,
            model_path=keras_rel,
            res=args.resolution,
            quant_split=args.quant_split,
        )
        out_path.write_text(content, encoding="utf-8")
        print(f"[WROTE] {out_path}")
        written.append(out_path)

    # 1) Print rescale commands needed before chain_eqe (for k > 0)
    rescale_ks = [k for k in args.k if k > 0]
    if rescale_ks:
        print()
        print("=" * 72)
        print("Run these BEFORE chain_eqe (creates the rescaled .keras files):")
        print("=" * 72)
        rescale_script_rel = "scripts/rescale_obj_logits.py"
        for k in rescale_ks:
            in_path = base_rel.lstrip("./")  # relative to object_detection/
            in_path_repo = f"object_detection/{in_path}"
            out_path_repo = in_path_repo.replace("best_model.keras", f"best_model_obj_k{k}.keras")
            print(f"python {rescale_script_rel} `")
            print(f"  --in  {in_path_repo} `")
            print(f"  --out {out_path_repo} `")
            print(f"  --k {k}")
            print()

    # 2) Print snippet for run_pareto_sweep.ps1
    print("=" * 72)
    print("Snippet for scripts/run_pareto_sweep.ps1 ($Models array):")
    print("=" * 72)
    base_for_snippet = base_rel.lstrip("./")
    print("[PSCustomObject]@{")
    print(f'    Name        = "{args.name}"')
    print(f'    BaseKeras   = Join-Path $RepoRoot "object_detection/{base_for_snippet}"')
    for k in sorted(args.k):
        print(f'    ConfigK{k}    = "my_chain_eqe_k{k}_{args.name}"')
    print("},")
    print()

    # 3) Summary
    print("=" * 72)
    print(f"Summary: wrote {len(written)} configs, skipped {len(skipped)}")
    if written:
        print(f"  Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
