"""
Confidence-threshold sweep for the deployed INT8 models on the SLO field set.

For each (model, test-set) combination, runs inference once at a very low
confidence threshold (captures all post-NMS candidates), then sweeps the
score threshold from 0.05 to 0.95 in 0.05 steps. For each threshold:
   - filters detections by score >= T
   - matches to GT via IoU >= 0.5 (greedy, highest-IoU-first)
   - counts TP, FP, FN
   - computes precision, recall, F1
Outputs:
   - one CSV per (model, set): threshold,TP,FP,FN,precision,recall,F1
   - three plots per (model, set): P/R-vs-T, F1-vs-T, PR curve
   - one combined comparison plot: F1-vs-T for all (model, set) combos
   - one summary CSV: best F1 threshold per (model, set), AP from trapezoidal PR

Why cache raw model outputs once instead of re-running inference per threshold:
NMS is deterministic given input scores — running NMS at confidence_thresh=0.001
captures every detection that would survive at any threshold, since suppressed
candidates would be suppressed at any higher threshold too. Filtering post-NMS by
score >= T is mathematically equivalent to running NMS at threshold T, provided
max_detection_boxes is set generously (we use 50 vs the eval's 10 to avoid clipping).

Run from repo root:
    python scripts/threshold_sweep.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from object_detection.tf.src.postprocessing import get_nmsed_detections

# ─── Configuration ─────────────────────────────────────────────────────────
DATASET_DIR = REPO_ROOT / "object_detection" / "datasets" / "SLO_stop_sign_field_test"
IMAGES_DIR = DATASET_DIR / "data"
FULL_LABELS = DATASET_DIR / "labels.json"
NEARMID_LABELS = DATASET_DIR / "labels_near_mid.json"

DEFAULT_ANCHORS = [
    0.076023, 0.258508, 0.163031, 0.413531, 0.234769, 0.702585,
    0.427054, 0.715892, 0.748154, 0.857092,
]

MODELS = [
    {
        "name": "256_default_k2",
        "tflite": REPO_ROOT / "object_detection/tf/src/experiments_outputs/2026_05_09_11_35_01/quantized_models/quantized_model.tflite",
        "input_size": 256,
    },
    {
        "name": "192v1_k0",
        "tflite": REPO_ROOT / "object_detection/tf/src/experiments_outputs/2026_05_07_10_02_38/quantized_models/quantized_model.tflite",
        "input_size": 192,
    },
]

TEST_SETS = [
    {"name": "full",     "labels_path": FULL_LABELS},
    {"name": "nearmid",  "labels_path": NEARMID_LABELS},
]

# Sweep parameters
SWEEP_LOW = 0.001           # confidence at which we run NMS (captures all candidates)
SWEEP_STEP = 0.05           # threshold step for the sweep
THRESHOLDS = np.arange(0.05, 1.00, SWEEP_STEP)
NMS_THRESH = 0.5            # NMS IoU threshold (same as eval)
IOU_MATCH_THRESH = 0.5      # match threshold (same as eval / standard mAP@0.5)
MAX_DETECTIONS = 10         # matches eval config + deployment (deployment_config.yaml: max_detection_boxes: 10)
                            # — higher values let too many low-score noise detections survive NMS,
                            # inflating FP counts at low sweep thresholds.

OUT_DIR = REPO_ROOT / "object_detection" / "tf" / "src" / "experiments_outputs" / "threshold_sweep"


# ─── Inference + matching plumbing ─────────────────────────────────────────
def build_cfg(confidence_thresh: float):
    """SimpleNamespace cfg that satisfies get_nmsed_detections (cf. postprocess.py:627).

    OmegaConf rejects np.ndarray values, so we hand-build with the ready-to-use
    reshaped anchor array.
    """
    return SimpleNamespace(
        model=SimpleNamespace(model_type="st_yololcv1", framework="tf"),
        dataset=SimpleNamespace(class_names=["stop_sign"]),
        postprocessing=SimpleNamespace(
            confidence_thresh=confidence_thresh,
            NMS_thresh=NMS_THRESH,
            max_detection_boxes=MAX_DETECTIONS,
            yolo_anchors=np.array(DEFAULT_ANCHORS, dtype=np.float32).reshape(-1, 2),
            network_stride=16,
        ),
    )


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def infer_one(interpreter, input_details, output_details, img: Image.Image, input_size: int, cfg):
    """Run inference + NMS once. Returns (pred_boxes_norm_xyxy, pred_scores)."""
    resized = img.resize((input_size, input_size), Image.NEAREST).convert("RGB")
    arr = (np.asarray(resized, dtype=np.float32) / 255.0)[None, ...]
    input_dtype = input_details["dtype"]
    if np.issubdtype(input_dtype, np.floating):
        feed = arr.astype(input_dtype)
    else:
        scale, zero_point = input_details["quantization"]
        feed = (arr / scale + zero_point).astype(input_dtype)
    interpreter.set_tensor(input_details["index"], feed)
    interpreter.invoke()
    raw = interpreter.get_tensor(output_details["index"])
    boxes, scores, _ = get_nmsed_detections(cfg, raw, (input_size, input_size))
    return boxes.numpy()[0], scores.numpy()[0]


def count_tp_fp_fn(gt_norm: list, pred_boxes: np.ndarray, pred_scores: np.ndarray,
                   threshold: float) -> tuple[int, int, int]:
    """Greedy IoU matching of GTs to predictions at the given score threshold.

    Same matching logic as analyze_field_attributes.py — each GT picks its
    highest-IoU unmatched prediction. Any prediction not claimed by a GT is FP.
    """
    keep = pred_scores >= threshold
    preds = pred_boxes[keep]
    matched_pred = set()
    tp = fn = 0
    for gt in gt_norm:
        best_iou = 0.0
        best_pi = -1
        gt_arr = np.asarray(gt)
        for pi in range(len(preds)):
            if pi in matched_pred:
                continue
            iou = iou_xyxy(gt_arr, preds[pi])
            if iou > best_iou:
                best_iou = iou
                best_pi = pi
        if best_iou >= IOU_MATCH_THRESH and best_pi >= 0:
            matched_pred.add(best_pi)
            tp += 1
        else:
            fn += 1
    fp = len(preds) - len(matched_pred)
    return tp, fp, fn


def compute_pr_metrics(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Returns (precision, recall, F1) for a single confusion count."""
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def trapezoidal_ap(rows: list[dict]) -> float:
    """Approximate AP as area under the PR curve (trapezoidal rule).

    Sorts rows by recall ascending, then sums (Δrecall × average precision).
    Not exactly equivalent to COCO's 101-point interpolated AP, but matches it
    closely for smooth curves. Mainly useful here as a sanity check vs the
    modelzoo's reported AP.
    """
    pts = sorted([(r["recall"], r["precision"]) for r in rows], key=lambda x: x[0])
    # Include the (recall=0, precision=last) anchor so the area starts at 0
    if pts and pts[0][0] > 0:
        pts = [(0.0, pts[0][1])] + pts
    ap = 0.0
    for i in range(1, len(pts)):
        dr = pts[i][0] - pts[i - 1][0]
        avg_p = (pts[i][1] + pts[i - 1][1]) / 2
        ap += dr * avg_p
    return ap


# ─── Plotting ──────────────────────────────────────────────────────────────
def plot_pr_vs_threshold(rows: list[dict], title: str, out_path: Path):
    """Precision and Recall both as functions of confidence threshold."""
    ts = [r["threshold"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ts, [r["precision"] for r in rows], marker="o", label="Precision", color="#1f77b4")
    ax.plot(ts, [r["recall"] for r in rows], marker="s", label="Recall", color="#d62728")
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Metric value")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_f1_vs_threshold(rows: list[dict], title: str, out_path: Path):
    """F1 vs confidence threshold with the max F1 point highlighted."""
    ts = [r["threshold"] for r in rows]
    f1s = [r["f1"] for r in rows]
    best_idx = int(np.argmax(f1s))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ts, f1s, marker="o", color="#2ca02c")
    ax.scatter([ts[best_idx]], [f1s[best_idx]], s=180, facecolor="none",
               edgecolor="#d62728", linewidth=2.5,
               label=f"max F1 = {f1s[best_idx]:.3f} at T = {ts[best_idx]:.2f}", zorder=5)
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("F1 score")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(0.5, max(f1s) * 1.1))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_pr_curve(rows: list[dict], title: str, out_path: Path):
    """Classic Precision-Recall curve with points colored by threshold."""
    recalls = [r["recall"] for r in rows]
    precisions = [r["precision"] for r in rows]
    thresholds = [r["threshold"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(recalls, precisions, c=thresholds, cmap="viridis", s=60, edgecolor="k", linewidth=0.5)
    ax.plot(recalls, precisions, color="#cccccc", linewidth=1, zorder=0)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Confidence threshold")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_f1_comparison(all_results: dict, out_path: Path):
    """F1 vs threshold for all (model, set) combos overlaid — comparison view."""
    fig, ax = plt.subplots(figsize=(10, 6))
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, (key, rows) in enumerate(all_results.items()):
        ts = [r["threshold"] for r in rows]
        f1s = [r["f1"] for r in rows]
        ax.plot(ts, f1s, marker="o", markersize=4, label=key, color=palette[i % len(palette)])
        best_idx = int(np.argmax(f1s))
        ax.scatter([ts[best_idx]], [f1s[best_idx]], s=140, facecolor="none",
                   edgecolor=palette[i % len(palette)], linewidth=2, zorder=5)
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("F1 score")
    ax.set_title("F1 vs confidence threshold — all (model, test-set) combinations")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", title="Max F1 points circled")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ─── Per-(model, set) sweep ────────────────────────────────────────────────
def sweep_one(model_info: dict, test_set: dict, cached_preds: dict | None = None) -> tuple[list[dict], dict]:
    """Run one full sweep for a (model, test-set) combination.

    `cached_preds` lets us reuse inference results across test-set variants
    (full vs near+mid use the SAME predictions, only the GT subset changes).
    Returns (rows, predictions_cache_for_reuse).
    """
    label = f"{model_info['name']} / {test_set['name']}"
    print(f"\n{'='*78}")
    print(f"Sweep: {label}")
    print(f"{'='*78}")

    with open(test_set["labels_path"], "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    images_by_id = {im["id"]: im for im in gt_data["images"]}
    anns_by_image: dict = {}
    for a in gt_data["annotations"]:
        anns_by_image.setdefault(a["image_id"], []).append(a)
    total_gt = sum(len(v) for v in anns_by_image.values())
    print(f"  images: {len(images_by_id)}  GT boxes: {total_gt}")

    if cached_preds is None:
        # Run inference once per image at very low threshold (captures all candidates)
        cfg_low = build_cfg(SWEEP_LOW)
        interpreter = tf.lite.Interpreter(model_path=str(model_info["tflite"]))
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]

        cached_preds = {}
        for image_id, image_meta in images_by_id.items():
            img_path = IMAGES_DIR / image_meta["file_name"]
            if not img_path.exists():
                continue
            img = Image.open(img_path)
            W, H = img.size
            pred_boxes, pred_scores = infer_one(interpreter, input_details, output_details,
                                                 img, model_info["input_size"], cfg_low)
            cached_preds[image_id] = (W, H, pred_boxes, pred_scores)
        print(f"  inference cached: {len(cached_preds)} images")

    # Build per-image GT (normalized xyxy) using the test-set's annotation subset
    per_image_gt: dict = {}
    for image_id, image_meta in images_by_id.items():
        if image_id not in cached_preds:
            continue
        W, H, _, _ = cached_preds[image_id]
        gts = anns_by_image.get(image_id, [])
        per_image_gt[image_id] = [
            [g["bbox"][0] / W, g["bbox"][1] / H,
             (g["bbox"][0] + g["bbox"][2]) / W,
             (g["bbox"][1] + g["bbox"][3]) / H] for g in gts
        ]

    # Sweep — for each threshold, tally globally then compute P/R/F1
    rows = []
    for T in THRESHOLDS:
        tp = fp = fn = 0
        for image_id, gt_norm in per_image_gt.items():
            _, _, pred_boxes, pred_scores = cached_preds[image_id]
            t_tp, t_fp, t_fn = count_tp_fp_fn(gt_norm, pred_boxes, pred_scores, T)
            tp += t_tp
            fp += t_fp
            fn += t_fn
        p, r, f1 = compute_pr_metrics(tp, fp, fn)
        rows.append({
            "threshold": round(float(T), 3),
            "TP": tp, "FP": fp, "FN": fn,
            "precision": round(p, 4),
            "recall":    round(r, 4),
            "f1":        round(f1, 4),
        })

    ap = trapezoidal_ap(rows)
    best = max(rows, key=lambda r: r["f1"])

    # Print every threshold row (* marks the max-F1 row)
    best_t = round(best["threshold"], 3)
    print(f"\n  Threshold  TP   FP   FN   Precision  Recall   F1")
    print(  "  ---------  ---  ---  ---  ---------  -------  ----")
    for row in rows:
        marker = " *" if row["threshold"] == best_t else ""
        print(f"  T={row['threshold']:.2f}     {row['TP']:>3}  {row['FP']:>3}  {row['FN']:>3}  "
              f"{row['precision']*100:>7.2f}%   {row['recall']*100:>5.2f}%  {row['f1']:.3f}{marker}")
    print(f"\n  Best F1 = {best['f1']:.3f} at threshold {best['threshold']:.2f}  "
          f"(TP={best['TP']}, FP={best['FP']}, FN={best['FN']}, "
          f"P={best['precision']*100:.1f}%, R={best['recall']*100:.1f}%)")
    print(f"  Trapezoidal AP (from sweep PR points) ~= {ap*100:.1f}%")

    return rows, cached_preds, ap, best


# ─── Main ──────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUT_DIR.relative_to(REPO_ROOT)}")
    print(f"Threshold sweep: {THRESHOLDS[0]:.2f} to {THRESHOLDS[-1]:.2f} step {SWEEP_STEP:.2f}  "
          f"({len(THRESHOLDS)} thresholds)")

    all_results: dict = {}
    summary_rows: list[dict] = []

    for m in MODELS:
        # Reuse inference results across full and near+mid sets (same model, same images)
        cached: dict | None = None
        for ts in TEST_SETS:
            key = f"{m['name']}__{ts['name']}"
            rows, cached, ap, best = sweep_one(m, ts, cached_preds=cached)
            all_results[key] = rows

            sub_dir = OUT_DIR / key
            sub_dir.mkdir(parents=True, exist_ok=True)

            # Write CSV
            csv_path = sub_dir / "sweep.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)

            # Plots
            title = f"{m['name']} on field set ({ts['name']})"
            plot_pr_vs_threshold(rows, title + "\nPrecision and Recall vs threshold",
                                 sub_dir / "pr_vs_threshold.png")
            plot_f1_vs_threshold(rows, title + "\nF1 score vs threshold",
                                 sub_dir / "f1_vs_threshold.png")
            plot_pr_curve(rows, title + "\nPrecision–Recall curve (colored by threshold)",
                          sub_dir / "pr_curve.png")

            summary_rows.append({
                "model": m["name"],
                "test_set": ts["name"],
                "best_f1": best["f1"],
                "best_threshold": best["threshold"],
                "best_TP": best["TP"],
                "best_FP": best["FP"],
                "best_FN": best["FN"],
                "best_precision": best["precision"],
                "best_recall": best["recall"],
                "trapezoidal_AP": round(ap, 4),
            })

    # Comparison plot
    plot_f1_comparison(all_results, OUT_DIR / "f1_comparison_all.png")

    # Summary CSV
    summary_path = OUT_DIR / "summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    print(f"\n{'='*78}")
    print("ALL DONE")
    print(f"{'='*78}")
    print(f"  Per-combo subdirs:  {OUT_DIR.relative_to(REPO_ROOT)}/<model>__<set>/")
    print(f"    sweep.csv             - all (threshold, TP, FP, FN, P, R, F1) rows")
    print(f"    pr_vs_threshold.png   - P and R as two lines vs threshold")
    print(f"    f1_vs_threshold.png   - F1 vs threshold (max F1 circled)")
    print(f"    pr_curve.png          - classic PR curve, colored by threshold")
    print(f"  Comparison plot:    {(OUT_DIR / 'f1_comparison_all.png').relative_to(REPO_ROOT)}")
    print(f"  Summary CSV:        {summary_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
