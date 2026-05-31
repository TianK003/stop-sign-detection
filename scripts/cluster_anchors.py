"""K-means cluster bounding-box (w, h) dims from a COCO JSON to produce YOLO anchors.

Usage:
  python scripts/cluster_anchors.py                                    # defaults
  python scripts/cluster_anchors.py --labels <coco.json> --k 5

Reads box dims (normalized by image w,h) from the COCO JSON, runs k-means
(IoU distance, not euclidean — the standard YOLO recipe), and prints anchors
in the flat-list form the modelzoo's `postprocessing.yolo_anchors` expects.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LABELS = REPO_ROOT / "object_detection" / "datasets" / "coco_stop_sign" / "train" / "labels.json"


def load_box_dims(coco_json: Path) -> np.ndarray:
    """Return Nx2 array of (w_rel, h_rel) for every annotation, normalized by image size."""
    with coco_json.open() as f:
        data = json.load(f)

    img_dims = {img["id"]: (img["width"], img["height"]) for img in data["images"]}
    dims = []
    for ann in data["annotations"]:
        iw, ih = img_dims[ann["image_id"]]
        _, _, bw, bh = ann["bbox"]
        if bw <= 0 or bh <= 0:
            continue
        dims.append((bw / iw, bh / ih))
    return np.asarray(dims, dtype=np.float32)


def iou_distance(boxes: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """1 - IoU of every box vs every anchor, treating both as origin-aligned (w,h only)."""
    inter = np.minimum(boxes[:, None, 0], anchors[None, :, 0]) * \
            np.minimum(boxes[:, None, 1], anchors[None, :, 1])
    box_area = boxes[:, 0] * boxes[:, 1]
    anchor_area = anchors[:, 0] * anchors[:, 1]
    union = box_area[:, None] + anchor_area[None, :] - inter
    return 1.0 - inter / union


def kmeans_iou(boxes: np.ndarray, k: int, max_iter: int = 200, seed: int = 127) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = boxes.shape[0]
    init_idx = rng.choice(n, size=k, replace=False)
    anchors = boxes[init_idx].copy()
    last_assign = -np.ones(n, dtype=np.int64)

    for _ in range(max_iter):
        d = iou_distance(boxes, anchors)
        assign = np.argmin(d, axis=1)
        if np.array_equal(assign, last_assign):
            break
        for i in range(k):
            members = boxes[assign == i]
            if len(members) > 0:
                anchors[i] = np.median(members, axis=0)
        last_assign = assign

    # Sort by area (smallest first) — matches the convention of the stock anchors
    order = np.argsort(anchors[:, 0] * anchors[:, 1])
    return anchors[order]


def mean_iou(boxes: np.ndarray, anchors: np.ndarray) -> float:
    return float((1.0 - iou_distance(boxes, anchors).min(axis=1)).mean())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    p.add_argument("--k", type=int, default=5)
    args = p.parse_args()

    boxes = load_box_dims(args.labels)
    print(f"Loaded {len(boxes)} boxes from {args.labels}")
    print(f"Box w stats: min {boxes[:, 0].min():.4f}, mean {boxes[:, 0].mean():.4f}, max {boxes[:, 0].max():.4f}")
    print(f"Box h stats: min {boxes[:, 1].min():.4f}, mean {boxes[:, 1].mean():.4f}, max {boxes[:, 1].max():.4f}")
    aspect = boxes[:, 0] / boxes[:, 1]
    print(f"Aspect ratios (w/h): min {aspect.min():.3f}, median {np.median(aspect):.3f}, max {aspect.max():.3f}")

    anchors = kmeans_iou(boxes, k=args.k)
    miou = mean_iou(boxes, anchors)
    print(f"\nK-means (k={args.k}, IoU distance) -> mean best-anchor IoU: {miou:.4f}")
    print("Anchors (sorted by area, w h):")
    for i, (w, h) in enumerate(anchors):
        print(f"  [{w:.6f} {h:.6f}]   aspect {w/h:.3f}   area {w*h:.4f}")

    flat = ", ".join(f"{v:.6f}" for row in anchors for v in row)
    print("\nFor my_training_config_*.yaml (under postprocessing:):")
    print(f"  yolo_anchors: [{flat}]")


if __name__ == "__main__":
    main()
