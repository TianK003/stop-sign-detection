# DATA.md — datasets are not shipped; here is how to reconstruct them

No image data is included in this repository. The two datasets used in the thesis are
withheld for licensing and privacy reasons, but both are described here so the pipeline
is reproducible.

## 1. COCO 2017 stop-sign subset (training / validation) — reconstructable

- **What it is.** Every COCO 2017 image annotated with the `stop sign` category
  (category_id 13), remapped to a single contiguous class `category_id 13 → 0`
  (the modelzoo expects 0-indexed classes). Roughly **~1 700 train / ~75 val** images.
- **How to rebuild it.** Run `python scripts/extract_coco_stop_sign.py`. It downloads
  the official COCO 2017 annotations once, then fetches only the matching images, and
  writes them under `object_detection/datasets/coco_stop_sign/{train,val}/`. The OD
  pipeline auto-converts the COCO JSON to YOLO Darknet `.tfs` labels on first run.
- **Why it is not shipped.** COCO annotations are CC-BY 4.0, but the underlying images
  are sourced from Flickr under assorted per-image licenses. We therefore ship the
  extraction script rather than redistributing the images.

## 2. Slovenian field / validation photos — withheld for privacy

- **What it is.** 118 self-collected photos of real Slovenian street STOP signs
  (5–50 m, multiple angles and conditions), with 148 labeled stop-sign boxes, annotated
  in CVAT with five per-box attributes (`distance`, `occlusion`, `condition`, `face`,
  `angle`). Used for real-world validation, the failure-mode taxonomy, and the live
  field test.
- **Why it is not shipped.** The photos contain license plates, faces, and private
  property. They are **deliberately withheld** and are not present anywhere in this
  repository or its history. The CVAT label exports of these photos are withheld for the
  same reason.
- **Role in the thesis.** They back the "Field mAP" columns, the near/mid vs far
  detection-floor analysis, and the FN/FP breakdown. To reproduce that analysis you would
  need to collect and label an equivalent set; the scripts
  (`scripts/analyze_field_attributes.py`, `scripts/filter_field_labels.py`,
  `scripts/threshold_sweep.py`) expect the CVAT-style layout under
  `object_detection/datasets/SLO_stop_sign_field_test/`.

## Pretrained starting checkpoint

Fine-tuning starts from ST's COCO-Person `st_yololcv1` checkpoint, hosted in the separate
[`stm32ai-modelzoo`](https://github.com/STMicroelectronics/stm32ai-modelzoo) repo (not
this services repo). Clone it next to this one so the configs' relative path
`../../stm32ai-modelzoo/...` resolves:

```powershell
cd ..
git clone --depth 1 https://github.com/STMicroelectronics/stm32ai-modelzoo.git
```
