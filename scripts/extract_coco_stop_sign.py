"""Filter COCO 2017 to single-class 'stop sign' subset for thesis training.

Outputs:
  object_detection/datasets/coco_stop_sign/train/labels.json + data/*.jpg
  object_detection/datasets/coco_stop_sign/val/labels.json   + data/*.jpg

Class id is remapped 13 -> 0 (modelzoo expects contiguous 0-indexed class IDs).
"""
import json
import os
import pathlib
import urllib.request
import zipfile

ROOT = pathlib.Path("object_detection/datasets/coco_stop_sign")
TMP = pathlib.Path(os.environ.get("TEMP", "/tmp")) / "coco_ann"
ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

# 1) Download + extract the official COCO annotations once (~250 MB)
TMP.mkdir(parents=True, exist_ok=True)
zip_path = TMP / "annotations_trainval2017.zip"
if not zip_path.exists():
    print(f"Downloading {ANN_URL} ...")
    urllib.request.urlretrieve(ANN_URL, zip_path)
    print(f"  saved to {zip_path}")
ann_dir = TMP / "annotations"
if not ann_dir.exists():
    print("Extracting annotations zip ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(TMP)

# 2) For each split, filter to stop sign and remap class id 13 -> 0
for split, ann_name in [("train", "instances_train2017.json"),
                        ("val",   "instances_val2017.json")]:
    print(f"\n--- {split} ---")
    with open(ann_dir / ann_name) as f:
        d = json.load(f)
    cat = next(c for c in d["categories"] if c["name"] == "stop sign")
    print(f"  source COCO category: {cat}")  # expect id=13
    keep_ann = [a for a in d["annotations"] if a["category_id"] == cat["id"]]
    keep_img_ids = {a["image_id"] for a in keep_ann}
    keep_img = [im for im in d["images"] if im["id"] in keep_img_ids]
    # Critical: remap category_id 13 -> 0 (modelzoo expects 0-indexed)
    for a in keep_ann:
        a["category_id"] = 0
    new = {
        "info":        d.get("info", {}),
        "licenses":    d.get("licenses", []),
        "images":      keep_img,
        "annotations": keep_ann,
        "categories":  [{"id": 0, "name": "stop_sign", "supercategory": "outdoor"}],
    }
    out = ROOT / split
    (out / "data").mkdir(parents=True, exist_ok=True)
    with open(out / "labels.json", "w") as f:
        json.dump(new, f)
    print(f"  wrote labels.json: {len(keep_img)} images, {len(keep_ann)} boxes")

    # 3) Download just the matching images (NOT the full 18 GB COCO)
    failed = []
    for i, im in enumerate(keep_img):
        dest = out / "data" / im["file_name"]
        if dest.exists():
            continue
        try:
            urllib.request.urlretrieve(im["coco_url"], dest)
        except Exception as e:
            failed.append((im["file_name"], str(e)))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(keep_img)} images downloaded")
    print(f"  images: {len(keep_img) - len(failed)} downloaded, {len(failed)} failed")
    if failed:
        print(f"  first failure: {failed[0]}")
        print(f"  retry by re-running this script (already-downloaded images skipped)")

print("\nDone. Verify with:")
print(f"  python -c \"import json; d=json.load(open('{ROOT / 'train' / 'labels.json'}')); "
      f"print('train images:', len(d['images']), 'boxes:', len(d['annotations']), 'cats:', d['categories'])\"")
