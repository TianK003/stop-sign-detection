"""Post-hoc rescale of YOLO objectness-logit weights to recover INT8 PTQ.

Diagnostic finding (LOGBOOK 2026-05-08): the 256-input model's objectness
logits saturated at ~119 in raw-logit space. Sigmoid(119) and sigmoid(8) are both
indistinguishable from 1.0, so the model never benefited from being that
extreme — but the obj outliers monopolize the per-tensor activation
quantization scale and crush (tx, ty) precision.

This script scales the final BatchNorm gamma/beta on the 5 obj channels by
1/k. Algebraically:

    out_new = (gamma/k)*(x - mu)/sqrt(var + eps) + beta/k = out/k

so the obj logits shrink k-fold while everything else is untouched. Float
predictions are functionally identical (sigmoid still saturates), but the
post-fold per-tensor scale should tighten, restoring (tx, ty) resolution.

Channel layout for ST Yolo LC v1 with 1 class:
  30 channels = 5 anchors x 6 fields (anchor-major)
  field order: tx=0, ty=1, tw=2, th=3, obj=4, cls=5
  obj channels = [4, 10, 16, 22, 28]

Usage:
  python scripts/rescale_obj_logits.py \\
      --in  object_detection/tf/src/experiments_outputs/2026_05_07_12_28_35/saved_models/best_model.keras \\
      --out object_detection/tf/src/experiments_outputs/2026_05_07_12_28_35/saved_models/best_model_obj_k5.keras \\
      --k 5

Then re-run PTQ + eval against the rescaled model:
  python stm32ai_main.py --config-name my_chain_eqe_obj_rescaled_256
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

NUM_ANCHORS = 5
FIELDS_PER_ANCHOR = 6
OBJ_FIELD = 4
OBJ_CHANNELS = [a * FIELDS_PER_ANCHOR + OBJ_FIELD for a in range(NUM_ANCHORS)]  # [4, 10, 16, 22, 28]


def find_final_bn(model: tf.keras.Model) -> tf.keras.layers.BatchNormalization:
    """Return the last BatchNormalization layer in the model.

    For ST Yolo LC v1 this is `batch_normalization_12` and operates on the
    (H, W, 30) raw output. We assert the channel count matches the expected
    anchor-major layout to avoid silently scaling the wrong layer.
    """
    bns = [l for l in model.layers if isinstance(l, tf.keras.layers.BatchNormalization)]
    if not bns:
        raise SystemExit("No BatchNormalization layers found in model.")
    final = bns[-1]
    expected_channels = NUM_ANCHORS * FIELDS_PER_ANCHOR
    last_dim = final.gamma.shape[-1]
    if last_dim != expected_channels:
        raise SystemExit(
            f"Last BN '{final.name}' has {last_dim} channels; expected {expected_channels} "
            "(5 anchors x 6 fields). Aborting — model layout doesn't match the assumed "
            "ST Yolo LC v1 anchor-major output."
        )
    return final


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--in", dest="src", required=True, help="path to source best_model.keras")
    p.add_argument("--out", dest="dst", required=True, help="path to write rescaled model")
    p.add_argument("--k", type=float, default=5.0, help="divide obj gamma/beta by this factor (default 5)")
    p.add_argument("--dry-run", action="store_true", help="report what would change without writing")
    args = p.parse_args()

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    if not src.exists():
        raise SystemExit(f"Source model not found: {src}")
    if args.k <= 0:
        raise SystemExit("k must be positive")

    print(f"Loading {src}")
    model = tf.keras.models.load_model(src, compile=False)

    bn = find_final_bn(model)
    print(f"Final BN layer: '{bn.name}'   channels={bn.gamma.shape[-1]}")

    gamma = bn.gamma.numpy().copy()
    beta = bn.beta.numpy().copy()

    print(f"\nFull layer stats:")
    print(f"  gamma: min={gamma.min():.4f}   max={gamma.max():.4f}   mean={gamma.mean():.4f}")
    print(f"  beta:  min={beta.min():.4f}   max={beta.max():.4f}   mean={beta.mean():.4f}")

    print(f"\nObj channels {OBJ_CHANNELS} BEFORE rescale:")
    print(f"  {'ch':>4}  {'gamma':>10}  {'beta':>10}")
    for c in OBJ_CHANNELS:
        print(f"  {c:>4}  {gamma[c]:10.4f}  {beta[c]:10.4f}")

    new_gamma = gamma.copy()
    new_beta = beta.copy()
    for c in OBJ_CHANNELS:
        new_gamma[c] /= args.k
        new_beta[c] /= args.k

    print(f"\nObj channels {OBJ_CHANNELS} AFTER rescale (k={args.k}):")
    print(f"  {'ch':>4}  {'gamma':>10}  {'beta':>10}")
    for c in OBJ_CHANNELS:
        print(f"  {c:>4}  {new_gamma[c]:10.4f}  {new_beta[c]:10.4f}")

    print(f"\nNon-obj channels are unchanged.")
    non_obj = [c for c in range(NUM_ANCHORS * FIELDS_PER_ANCHOR) if c not in OBJ_CHANNELS]
    assert np.array_equal(new_gamma[non_obj], gamma[non_obj])
    assert np.array_equal(new_beta[non_obj], beta[non_obj])

    if args.dry_run:
        print("\n[dry-run] no file written.")
        return

    bn.gamma.assign(new_gamma)
    bn.beta.assign(new_beta)

    dst.parent.mkdir(parents=True, exist_ok=True)
    model.save(dst)
    print(f"\nWrote rescaled model -> {dst}")
    print(
        "\nNext: run\n"
        "  python stm32ai_main.py --config-name my_chain_eqe_obj_rescaled_256\n"
        "from object_detection/. The chain_eqe op-mode will:\n"
        "  1) evaluate the rescaled float model (should still be ~36.9% mAP — sanity check)\n"
        "  2) re-quantize via TFLite_converter\n"
        "  3) evaluate the new int8 model\n"
        "If int8 mAP recovers above ~25%, the obj-saturation diagnosis is confirmed.\n"
    )


if __name__ == "__main__":
    main()
