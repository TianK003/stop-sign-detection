"""Compare raw 16x16x30 (or 12x12x30) outputs of float vs INT8 ST Yolo LC v1 on the same images.

The model emits raw logits that postprocessing later interprets as
  feats[..., 0:2] -> sigmoid -> (tx, ty)        cell-relative center
  feats[..., 2:4] -> exp     -> (tw, th)        anchor-multiplier
  feats[..., 4:5] -> sigmoid -> objectness
  feats[..., 5: ] -> softmax -> class probs

Quantization noise on tw/th gets exponentiated, so the same noise on those
two channels is far more damaging than on tx/ty/obj/cls. This script breaks
errors out by anchor and field type so you can see exactly where INT8 hurts.

Usage:
  python scripts/compare_float_int8.py \\
      --float object_detection/tf/src/experiments_outputs/2026_05_07_12_28_35/saved_models/best_model.keras \\
      --int8  object_detection/tf/src/experiments_outputs/2026_05_07_12_28_35/quantized_models/quantized_model.tflite \\
      --images object_detection/datasets/coco_stop_sign/val/data \\
      --size 256 --n 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


FIELD_NAMES = ["tx", "ty", "tw", "th", "obj", "cls"]


def load_image(path: Path, size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((size, size), Image.NEAREST)
    return np.asarray(img, dtype=np.float32) / 255.0


def run_float(model: tf.keras.Model, x: np.ndarray) -> np.ndarray:
    return model(x[np.newaxis, ...], training=False).numpy()[0]


def run_int8(interp: tf.lite.Interpreter, x: np.ndarray) -> np.ndarray:
    in_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]

    scale, zp = in_d["quantization"]
    x_q = (x / scale + zp).astype(in_d["dtype"])
    info = np.iinfo(in_d["dtype"])
    x_q = np.clip(x_q, info.min, info.max)

    interp.set_tensor(in_d["index"], x_q[np.newaxis, ...])
    interp.invoke()
    return interp.get_tensor(out_d["index"])[0]


def summarize(errs_per_image: np.ndarray, num_anchors: int) -> None:
    """errs_per_image shape (N, H, W, num_anchors, 6)"""
    abs_err = np.abs(errs_per_image)

    # Mean abs error per (anchor, field) across all images and spatial positions
    mean_per_af = abs_err.mean(axis=(0, 1, 2))  # (num_anchors, 6)
    max_per_af = abs_err.max(axis=(0, 1, 2))    # (num_anchors, 6)

    print("\n" + "=" * 78)
    print("Mean absolute error per (anchor, field) — averaged over images & cells")
    print("=" * 78)
    header = f"{'anchor':>7}  " + "  ".join(f"{n:>9}" for n in FIELD_NAMES)
    print(header)
    print("-" * len(header))
    for a in range(num_anchors):
        row = "  ".join(f"{mean_per_af[a, f]:9.4f}" for f in range(6))
        print(f"{a:>7}  {row}")

    print("\nMax absolute error per (anchor, field):")
    print(header)
    print("-" * len(header))
    for a in range(num_anchors):
        row = "  ".join(f"{max_per_af[a, f]:9.4f}" for f in range(6))
        print(f"{a:>7}  {row}")

    # Marginalize: mean error by FIELD (averaged over anchors)
    print("\n" + "=" * 78)
    print("Mean abs error by FIELD type (anchor-averaged)")
    print("=" * 78)
    for f, name in enumerate(FIELD_NAMES):
        m = mean_per_af[:, f].mean()
        mx = max_per_af[:, f].max()
        print(f"  {name:>4}: mean={m:8.4f}   max={mx:8.4f}")

    # Specific check: tw/th vs tx/ty
    txty_mean = mean_per_af[:, 0:2].mean()
    twth_mean = mean_per_af[:, 2:4].mean()
    obj_mean = mean_per_af[:, 4].mean()
    print(f"\n(tx,ty) mean abs err: {txty_mean:.4f}")
    print(f"(tw,th) mean abs err: {twth_mean:.4f}")
    print(f"objectness mean abs err: {obj_mean:.4f}")
    if twth_mean > 1.5 * txty_mean:
        print(
            "\n>>> (tw,th) error is significantly larger than (tx,ty) error.\n"
            "    Box-size logits get exp()'d in postprocessing, so this kind of error\n"
            "    blows up multiplicatively into wrong-sized boxes."
        )
    if obj_mean > 1.0:
        print(
            "\n>>> Objectness error is large in raw-logit space.\n"
            "    Sigmoid is steepest near 0, so a 1.0-magnitude error there can\n"
            "    flip a box from confident-object to background or vice versa."
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--float", dest="float_path", required=True, help="path to best_model.keras")
    p.add_argument("--int8", dest="int8_path", required=True, help="path to quantized_model.tflite")
    p.add_argument("--images", required=True, help="directory of val/test images")
    p.add_argument("--n", type=int, default=20, help="number of images to compare (default 20)")
    p.add_argument("--size", type=int, default=256, help="model input H=W (default 256)")
    p.add_argument("--num-anchors", type=int, default=5)
    args = p.parse_args()

    float_path = Path(args.float_path)
    int8_path = Path(args.int8_path)
    img_dir = Path(args.images)

    print(f"Loading float model: {float_path}")
    float_model = tf.keras.models.load_model(float_path, compile=False)

    print(f"Loading INT8 model: {int8_path}")
    interp = tf.lite.Interpreter(model_path=str(int8_path))
    interp.allocate_tensors()

    in_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    print(f"  input  scale={in_d['quantization'][0]:.6f}  zp={in_d['quantization'][1]}  dtype={in_d['dtype']}")
    print(f"  output dtype={out_d['dtype']}  shape={out_d['shape']}")

    image_paths = sorted(
        [p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )[: args.n]
    if not image_paths:
        raise SystemExit(f"No images found in {img_dir}")

    print(f"\nComparing {len(image_paths)} images at {args.size}x{args.size}")

    errs = []  # list of (H, W, num_anchors, 6)
    floats = []
    int8s = []

    for ip in image_paths:
        x = load_image(ip, args.size)
        out_f = run_float(float_model, x)        # (H, W, 30)
        out_q = run_int8(interp, x)              # (H, W, 30)

        H, W, C = out_f.shape
        assert C == args.num_anchors * 6, f"unexpected channel count {C}"
        out_f5 = out_f.reshape(H, W, args.num_anchors, 6)
        out_q5 = out_q.reshape(H, W, args.num_anchors, 6)

        errs.append(out_f5 - out_q5)
        floats.append(out_f5)
        int8s.append(out_q5)

    errs_np = np.stack(errs)        # (N, H, W, num_anchors, 6)
    floats_np = np.stack(floats)
    int8s_np = np.stack(int8s)

    print(f"\nFloat output stats — overall:")
    print(f"  min={floats_np.min():.3f}  max={floats_np.max():.3f}  std={floats_np.std():.3f}")
    print(f"INT8 (dequantized) output stats — overall:")
    print(f"  min={int8s_np.min():.3f}  max={int8s_np.max():.3f}  std={int8s_np.std():.3f}")

    # Per-field range in float — useful for checking if calibration captured the range
    print(f"\nFloat output range per field (anchor-averaged):")
    f5 = floats_np.reshape(-1, args.num_anchors, 6)
    for f, name in enumerate(FIELD_NAMES):
        vals = f5[:, :, f]
        print(f"  {name:>4}: min={vals.min():8.3f}  max={vals.max():8.3f}  std={vals.std():.3f}")

    summarize(errs_np, args.num_anchors)


if __name__ == "__main__":
    main()
