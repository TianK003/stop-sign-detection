"""
Generate a filtered labels.json containing only `distance in {near, mid}` annotations.

Used by the *_nearmid.yaml eval configs to measure mAP excluding the 34 boxes that
are below the 256² model's effective minimum (~72 px native short side) — see
LOGBOOK 2026-05-11 evening "Effective-range analysis". Headline question: how much
of the 9.6 pp COCO-vs-field gap is the architectural size limit vs. genuine domain
shift?

Re-run after any re-tag of the field set.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO_ROOT / "object_detection" / "datasets" / "SLO_stop_sign_field_test"
SRC = DATASET_DIR / "labels.json"
DST = DATASET_DIR / "labels_near_mid.json"

KEEP_VALUES = {"near", "mid"}


def main():
    with open(SRC, "r", encoding="utf-8") as f:
        d = json.load(f)

    kept_anns = [a for a in d["annotations"] if a.get("attributes", {}).get("distance") in KEEP_VALUES]
    dropped = len(d["annotations"]) - len(kept_anns)
    print(f"Source: {SRC.relative_to(REPO_ROOT)}")
    print(f"  total annotations: {len(d['annotations'])}")
    print(f"  kept (distance in {KEEP_VALUES}): {len(kept_anns)}")
    print(f"  dropped (distance = far or missing): {dropped}")

    # Keep all images, just like the original — images with all-far signs become
    # background images, which is what we want (any far-sign detection becomes an FP).
    out = {**d, "annotations": kept_anns}
    with open(DST, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print(f"\nWrote: {DST.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
