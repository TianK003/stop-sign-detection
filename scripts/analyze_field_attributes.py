"""
Per-attribute breakdown for the SLO field-test set.

Runs both deployed INT8 models on the field set and reports TP / FN per CVAT
attribute value, so the thesis-section-4 failure-mode table can be populated
with concrete recall numbers per (distance / occlusion / condition / face / angle)
bin. Also dumps full-resolution annotated images with GT (green) and predicted
(red) boxes for thesis figures.

Numbers reconcile with stm32ai_main.py --config-name my_field_eval_int8_*
because the inference + postprocessing path is shared (we import
get_nmsed_detections directly).

Run from repo root:
    python scripts/analyze_field_attributes.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from object_detection.tf.src.postprocessing import get_nmsed_detections

DATASET_DIR = REPO_ROOT / "object_detection" / "datasets" / "SLO_stop_sign_field_test"
LABELS_PATH = DATASET_DIR / "labels.json"
IMAGES_DIR = DATASET_DIR / "data"

# Shared anchors (parse_config.py:453 defaults — what both models were trained with)
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

ATTRS = ["distance", "occlusion", "condition", "face", "angle"]
CONFIDENCE_THRESH = 0.7      # operating threshold chosen by F1 sweep (thesis sec:thresh)
NMS_THRESH = 0.5
IOU_MATCH_THRESH = 0.5
MAX_DETECTIONS = 10

# Architectural floor — anchors[0] = 0.018 * input_size px at model input.
# At native 4000x3000 with 256² input: 0.018 * 256 * (4000/256) = 72 px short side.
# Signs below this are architecturally impossible to localize regardless of
# domain shift; failure is the model's representational ceiling, not its training.
SHORT_SIDE_FLOOR_PX = 72

# IoU bands used to classify FPs. Near-miss = the model FOUND the sign but the
# box geometry was wrong by enough to miss the 0.5 IoU threshold. Spurious = the
# box has nothing to do with any GT sign.
FP_NEAR_MISS_IOU_LO = 0.3
FP_NEAR_MISS_IOU_HI = 0.5


def build_cfg(input_size: int):
    """Minimal cfg that get_nmsed_detections requires (cf. postprocess.py:627).

    SimpleNamespace because OmegaConf rejects np.ndarray values, and parse_config.py
    only converts yolo_anchors to ndarray AFTER OmegaConf-building when stm32ai_main
    runs — we're bypassing that path and need the ndarray ready-to-use.
    """
    return SimpleNamespace(
        model=SimpleNamespace(model_type="st_yololcv1", framework="tf"),
        dataset=SimpleNamespace(class_names=["stop_sign"]),
        postprocessing=SimpleNamespace(
            confidence_thresh=CONFIDENCE_THRESH,
            NMS_thresh=NMS_THRESH,
            max_detection_boxes=MAX_DETECTIONS,
            yolo_anchors=np.array(DEFAULT_ANCHORS, dtype=np.float32).reshape(-1, 2),
            network_stride=16,
        ),
    )


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two boxes in (x1, y1, x2, y2) format, any coordinate system."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def classify_fn_mode(gt_attrs: dict, short_side_px: float) -> str:
    """Assign a primary failure mode to a missed GT (False Negative).

    Priority order matters: a far-distance sign that is *also* back-of-sign
    classifies as `below_detection_floor` because that's the dominant cause
    (the model architecturally cannot localize it regardless of orientation).
    Visual inspection of the residual `other` category is expected to take
    ~15-20 cases per model.
    """
    if short_side_px < SHORT_SIDE_FLOOR_PX:
        return "below_detection_floor"
    if gt_attrs.get("face") == "back":
        return "back_of_sign_OOD"
    if gt_attrs.get("occlusion") == "heavy":
        return "heavy_occlusion"
    if gt_attrs.get("occlusion") == "partial":
        return "partial_occlusion"
    if gt_attrs.get("condition") in ("worn", "vandalized"):
        return "condition_OOD"
    return "other_reasons"


def classify_fp_mode(best_iou_vs_any_gt: float) -> str:
    """Assign a primary failure mode to an unmatched prediction (False Positive).

    `near_miss_localization` = the model FOUND a sign-shaped region but the box
    geometry was off by enough to miss 0.5 IoU. These are recoverable with better
    box regression. `spurious_other` = the prediction has no nearby GT — the model
    confidently fired on something that is not a sign. These are the interesting
    failures (octagonal-non-signs, red blobs, gridline artifacts) — visual
    inspection required to subdivide.
    """
    if FP_NEAR_MISS_IOU_LO <= best_iou_vs_any_gt < FP_NEAR_MISS_IOU_HI:
        return "near_miss_localization"
    return "spurious_other"


def infer_one(interpreter, input_details, output_details, img: Image.Image, input_size: int, cfg):
    """Run inference on one image. Returns predicted boxes (normalized xyxy), scores, classes."""
    resized = img.resize((input_size, input_size), Image.NEAREST).convert("RGB")
    arr = np.asarray(resized, dtype=np.float32) / 255.0  # rescaling: 1/255, offset 0
    arr = arr[None, ...]  # batch dim

    input_dtype = input_details["dtype"]
    if np.issubdtype(input_dtype, np.floating):
        feed = arr.astype(input_dtype)
    else:
        scale, zero_point = input_details["quantization"]
        feed = (arr / scale + zero_point).astype(input_dtype)

    interpreter.set_tensor(input_details["index"], feed)
    interpreter.invoke()
    raw = interpreter.get_tensor(output_details["index"])

    boxes, scores, classes = get_nmsed_detections(cfg, raw, (input_size, input_size))
    # boxes are normalized [0,1] xyxy in resized canvas. Because aspect_ratio: fit uses
    # a plain stretch (dataset_loaders.py:410), canvas-normalized == original-normalized.
    return boxes.numpy()[0], scores.numpy()[0], classes.numpy()[0]


def _draw_boxes(img: Image.Image, gt_boxes_xyxy: list[list[float]], pred_boxes_xyxy: list[list[float]],
                pred_scores: list[float], font_scale: float = 1.0) -> Image.Image:
    """Draw GT (green) + predictions (red, with score) on a PIL image. Returns annotated copy."""
    img = img.copy()
    draw = ImageDraw.Draw(img)
    W, H = img.size
    line_w = max(1, min(W, H) // 200)

    try:
        font_size = max(int(10 * font_scale), int(W // 30))
        font = ImageFont.truetype("arial.ttf", size=font_size)
    except OSError:
        font = ImageFont.load_default()

    for x1, y1, x2, y2 in gt_boxes_xyxy:
        draw.rectangle([(x1, y1), (x2, y2)], outline=(0, 200, 0), width=line_w)

    for (x1, y1, x2, y2), score in zip(pred_boxes_xyxy, pred_scores):
        draw.rectangle([(x1, y1), (x2, y2)], outline=(220, 20, 20), width=line_w)
        label = f"{score:.2f}"
        tb = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle(tb, fill=(220, 20, 20))
        draw.text((x1, y1), label, fill=(255, 255, 255), font=font)

    return img


def annotate_thesis_pair(img_path: Path, gt_norm_xyxy: list[list[float]], pred_norm_xyxy: list[list[float]],
                         pred_scores: list[float], input_size: int,
                         full_res_path: Path, model_res_path: Path) -> bool:
    """Save TWO annotated versions: full resolution and model-input resolution.

    Both use the same normalized box coords (which is valid because aspect_ratio: fit uses
    a plain stretch — see dataset_loaders.py:410), so the same boxes are denormalized to
    each canvas size independently.

    Skips writing if BOTH output files already exist (delete a stale file to force a
    rebuild of just that one). Returns True if anything was written.
    """
    if full_res_path.exists() and model_res_path.exists():
        return False

    img_full = Image.open(img_path).convert("RGB")
    W, H = img_full.size

    if not full_res_path.exists():
        gt_full = [[b[0] * W, b[1] * H, b[2] * W, b[3] * H] for b in gt_norm_xyxy]
        pr_full = [[b[0] * W, b[1] * H, b[2] * W, b[3] * H] for b in pred_norm_xyxy]
        annotated_full = _draw_boxes(img_full, gt_full, pr_full, pred_scores, font_scale=1.0)
        full_res_path.parent.mkdir(parents=True, exist_ok=True)
        annotated_full.save(full_res_path, quality=92)

    if not model_res_path.exists():
        img_small = img_full.resize((input_size, input_size), Image.NEAREST)
        gt_small = [[b[0] * input_size, b[1] * input_size, b[2] * input_size, b[3] * input_size] for b in gt_norm_xyxy]
        pr_small = [[b[0] * input_size, b[1] * input_size, b[2] * input_size, b[3] * input_size] for b in pred_norm_xyxy]
        annotated_small = _draw_boxes(img_small, gt_small, pr_small, pred_scores, font_scale=0.8)
        model_res_path.parent.mkdir(parents=True, exist_ok=True)
        annotated_small.save(model_res_path, quality=92)

    return True


def analyze_model(model_info: dict, gt_data: dict, save_thesis_images: bool = True):
    """Run one model end-to-end and return per-attribute counts + FP count."""
    print(f"\n{'='*78}")
    print(f"Model: {model_info['name']}  ({model_info['input_size']}^2)")
    print(f"  {model_info['tflite']}")
    print(f"{'='*78}")

    cfg = build_cfg(model_info["input_size"])
    interpreter = tf.lite.Interpreter(model_path=str(model_info["tflite"]))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # tp/fn counts: {attr_name: {value: {"tp": int, "fn": int}}}
    counts = {a: defaultdict(lambda: {"tp": 0, "fn": 0}) for a in ATTRS}
    total_tp = 0
    total_fn = 0
    total_fp = 0

    # Per-failure rows for the failure-mode taxonomy CSVs (option b deepener)
    fn_rows: list[dict] = []
    fp_rows: list[dict] = []

    thesis_dir = REPO_ROOT / "object_detection" / "tf" / "thesis_figures" / model_info["name"]
    full_res_dir = thesis_dir / "full_resolution"
    model_res_dir = thesis_dir / f"{model_info['input_size']}_resolution"
    n_images_written = 0
    n_images_skipped = 0

    images_by_id = {im["id"]: im for im in gt_data["images"]}
    anns_by_image = defaultdict(list)
    for a in gt_data["annotations"]:
        anns_by_image[a["image_id"]].append(a)

    for image_id, image_meta in images_by_id.items():
        img_path = IMAGES_DIR / image_meta["file_name"]
        if not img_path.exists():
            continue
        img = Image.open(img_path)
        W, H = img.size

        # GT in normalized xyxy
        gts = anns_by_image.get(image_id, [])
        gt_norm = []
        gt_attrs_list = []
        for g in gts:
            x, y, w, h = g["bbox"]
            gt_norm.append([x / W, y / H, (x + w) / W, (y + h) / H])
            gt_attrs_list.append(g.get("attributes", {}))

        # Predictions in normalized xyxy
        pred_boxes, pred_scores, _ = infer_one(interpreter, input_details, output_details,
                                                img, model_info["input_size"], cfg)
        # Filter predictions below confidence threshold (NMS already applied scores >= threshold)
        keep = pred_scores >= CONFIDENCE_THRESH
        pred_boxes = pred_boxes[keep]
        pred_scores = pred_scores[keep]

        # Greedy match: each GT picks its highest-IoU unmatched prediction
        matched_pred = set()
        for gi, gt in enumerate(gt_norm):
            best_iou = 0.0
            best_pi = -1
            for pi in range(len(pred_boxes)):
                if pi in matched_pred:
                    continue
                iou = iou_xyxy(np.asarray(gt), pred_boxes[pi])
                if iou > best_iou:
                    best_iou = iou
                    best_pi = pi
            if best_iou >= IOU_MATCH_THRESH and best_pi >= 0:
                matched_pred.add(best_pi)
                tp_fn = "tp"
                total_tp += 1
            else:
                tp_fn = "fn"
                total_fn += 1
                # Record FN row for failure-mode CSV
                x_gt, y_gt, w_gt, h_gt = gts[gi]["bbox"]
                short_side_px = min(w_gt, h_gt)
                fn_rows.append({
                    "image": image_meta["file_name"],
                    "bbox_x": round(x_gt, 1),
                    "bbox_y": round(y_gt, 1),
                    "bbox_w": round(w_gt, 1),
                    "bbox_h": round(h_gt, 1),
                    "short_side_px": round(short_side_px, 1),
                    "best_iou_with_pred": round(best_iou, 3),
                    "distance": gt_attrs_list[gi].get("distance", ""),
                    "occlusion": gt_attrs_list[gi].get("occlusion", ""),
                    "condition": gt_attrs_list[gi].get("condition", ""),
                    "face": gt_attrs_list[gi].get("face", ""),
                    "angle": gt_attrs_list[gi].get("angle", ""),
                    "mode_auto": classify_fn_mode(gt_attrs_list[gi], short_side_px),
                    "mode_manual": "",
                    "notes": "",
                })
            for attr in ATTRS:
                val = gt_attrs_list[gi].get(attr, "<missing>")
                counts[attr][val][tp_fn] += 1

        # Record FP rows: any prediction that didn't claim a GT (matched_pred set).
        # For each unmatched pred, find its best IoU vs ANY GT (matched or not) —
        # that signal tells us whether it's a near-miss (model found the sign but
        # box was wrong) or spurious (no nearby sign at all).
        for pi in range(len(pred_boxes)):
            if pi in matched_pred:
                continue
            best_iou_vs_any = 0.0
            for gt in gt_norm:
                iou = iou_xyxy(np.asarray(gt), pred_boxes[pi])
                if iou > best_iou_vs_any:
                    best_iou_vs_any = iou
            # Convert normalized pred box back to absolute pixel coords for the CSV
            px1, py1, px2, py2 = pred_boxes[pi]
            fp_rows.append({
                "image": image_meta["file_name"],
                "pred_x1": round(px1 * W, 1),
                "pred_y1": round(py1 * H, 1),
                "pred_x2": round(px2 * W, 1),
                "pred_y2": round(py2 * H, 1),
                "pred_short_side_px": round(min(px2 - px1, py2 - py1) * min(W, H), 1),
                "pred_score": round(float(pred_scores[pi]), 3),
                "best_iou_vs_any_gt": round(best_iou_vs_any, 3),
                "mode_auto": classify_fp_mode(best_iou_vs_any),
                "mode_manual": "",
                "notes": "",
            })

        unmatched_preds = len(pred_boxes) - len(matched_pred)
        total_fp += unmatched_preds

        # Save thesis images: both full-resolution and model-input-resolution versions.
        # The model-resolution version shows what the network actually sees post-stretch;
        # the full-resolution version is for the thesis failure gallery.
        if save_thesis_images:
            wrote = annotate_thesis_pair(
                img_path=img_path,
                gt_norm_xyxy=gt_norm,
                pred_norm_xyxy=pred_boxes.tolist(),
                pred_scores=pred_scores.tolist(),
                input_size=model_info["input_size"],
                full_res_path=full_res_dir / image_meta["file_name"],
                model_res_path=model_res_dir / image_meta["file_name"],
            )
            if wrote:
                n_images_written += 1
            else:
                n_images_skipped += 1

    # Print results
    print(f"\nTotals: TP={total_tp}  FN={total_fn}  FP={total_fp}")
    overall_recall = total_tp / max(1, total_tp + total_fn)
    overall_precision = total_tp / max(1, total_tp + total_fp)
    print(f"Recall    = TP/(TP+FN) = {overall_recall*100:5.1f}%")
    print(f"Precision = TP/(TP+FP) = {overall_precision*100:5.1f}%")

    for attr in ATTRS:
        print(f"\n  {attr}:")
        for val, c in sorted(counts[attr].items()):
            denom = c["tp"] + c["fn"]
            rec = c["tp"] / denom if denom else 0.0
            print(f"    {val:<12} TP={c['tp']:>3}  FN={c['fn']:>3}  (recall {rec*100:5.1f}%  of {denom})")

    if save_thesis_images:
        print(f"\nThesis images: {thesis_dir.relative_to(REPO_ROOT)}")
        print(f"  written: {n_images_written}   skipped (already existed): {n_images_skipped}")
        print(f"  full_resolution/        — original 4000x3000 with boxes (failure gallery)")
        print(f"  {model_info['input_size']}_resolution/         — what the model actually sees post-stretch")

    # ─── Failure-mode taxonomy outputs (option b deepener) ───────────────
    out_dir = REPO_ROOT / "object_detection" / "tf" / "src" / "experiments_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    fn_csv = out_dir / f"field_failures_fn_{model_info['name']}.csv"
    fp_csv = out_dir / f"field_failures_fp_{model_info['name']}.csv"

    if fn_rows:
        with open(fn_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(fn_rows[0].keys()))
            w.writeheader()
            w.writerows(fn_rows)
    if fp_rows:
        with open(fp_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(fp_rows[0].keys()))
            w.writeheader()
            w.writerows(fp_rows)

    # Print FN taxonomy summary table — this goes straight into §4
    print(f"\n  FN failure modes (denominator = {total_fn}):")
    mode_counts_fn = Counter(r["mode_auto"] for r in fn_rows)
    for mode, n in mode_counts_fn.most_common():
        pct = 100 * n / max(1, total_fn)
        print(f"    {mode:<35} {n:>3}  ({pct:5.1f}%)")
    other_files = [r["image"] for r in fn_rows if r["mode_auto"] == "other_reasons"]
    if other_files:
        print(f"\n  -> {len(other_files)} FNs in 'other_reasons' — open these images")
        print(f"     in object_detection/tf/thesis_figures/{model_info['name']}/full_resolution/")
        print(f"     and fill the 'mode_manual' column in {fn_csv.name}:")
        for f in other_files[:10]:
            print(f"       {f}")
        if len(other_files) > 10:
            print(f"       ... ({len(other_files) - 10} more in CSV)")

    # FP taxonomy summary
    print(f"\n  FP failure modes (denominator = {total_fp}):")
    mode_counts_fp = Counter(r["mode_auto"] for r in fp_rows)
    for mode, n in mode_counts_fp.most_common():
        pct = 100 * n / max(1, total_fp)
        print(f"    {mode:<35} {n:>3}  ({pct:5.1f}%)")

    print(f"\n  CSVs written:")
    print(f"    {fn_csv.relative_to(REPO_ROOT)}  ({len(fn_rows)} rows)")
    print(f"    {fp_csv.relative_to(REPO_ROOT)}  ({len(fp_rows)} rows)")
    print(f"  Fill 'mode_manual' / 'notes' columns by visual inspection, then aggregate for §4.")

    return {
        "counts": {a: dict(counts[a]) for a in ATTRS},
        "tp": total_tp, "fn": total_fn, "fp": total_fp,
        "fn_mode_counts": dict(mode_counts_fn),
        "fp_mode_counts": dict(mode_counts_fp),
    }


def main():
    print(f"Loading GT from {LABELS_PATH.relative_to(REPO_ROOT)}")
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    print(f"  {len(gt_data['images'])} images, {len(gt_data['annotations'])} annotations")

    # Set save_thesis_images=False here if you only want to refresh the
    # per-attribute counts and failure-mode CSVs without regenerating the
    # 472-jpg thesis_figures tree. Default True keeps everything in sync.
    save_imgs = True
    results = {}
    for m in MODELS:
        results[m["name"]] = analyze_model(m, gt_data, save_thesis_images=save_imgs)

    # Side-by-side summary
    print("\n" + "=" * 78)
    print("SIDE-BY-SIDE PER-ATTRIBUTE RECALL")
    print("=" * 78)
    print(f"{'Attribute':<12} {'Value':<12} {'GT':>4} | {'256 TP':>7} {'256 R%':>7} | {'192 TP':>7} {'192 R%':>7}")
    print("-" * 78)
    for attr in ATTRS:
        all_vals = set()
        for r in results.values():
            all_vals.update(r["counts"][attr].keys())
        for val in sorted(all_vals):
            row = [attr, val]
            denom_any = None
            cells = []
            for name in ("256_default_k2", "192v1_k0"):
                c = results[name]["counts"][attr].get(val, {"tp": 0, "fn": 0})
                denom = c["tp"] + c["fn"]
                denom_any = denom if denom_any is None else denom_any
                rec = c["tp"] / denom if denom else 0.0
                cells.append((c["tp"], rec * 100))
            row.append(denom_any)
            print(f"{attr:<12} {val:<12} {denom_any:>4} | "
                  f"{cells[0][0]:>7} {cells[0][1]:>6.1f}% | "
                  f"{cells[1][0]:>7} {cells[1][1]:>6.1f}%")

    # Save summary JSON for §4 writing
    out_json = REPO_ROOT / "object_detection" / "tf" / "src" / "experiments_outputs" / "field_attribute_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=lambda o: dict(o) if hasattr(o, "items") else str(o))
    print(f"\nSummary JSON: {out_json.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
