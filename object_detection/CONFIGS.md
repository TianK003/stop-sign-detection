# Configuration guide

Every stage of this project runs through a single entry point driven by one YAML
file:

```powershell
cd object_detection
python stm32ai_main.py --config-name <file_without_.yaml>
```

The framework is [Hydra](https://hydra.cc) + OmegaConf, so any field can be
overridden on the command line, e.g. `python stm32ai_main.py --config-name train_config training.epochs=50`.
Each run is written to `tf/src/experiments_outputs/<timestamp>/` (models, plots,
logs) and tracked in MLflow under `tf/src/experiments_outputs/mlruns/`.

Rather than ship the ~30 ablation configs used in the thesis, this release
provides one annotated **template** plus four ready-to-run **examples**. Copy an
example, point the paths at your data/model, and go.

## Files

| File | `operation_mode` | What it does |
|---|---|---|
| `config_template.yaml` | (all) | Annotated reference documenting every section and field. Not run directly — copy blocks from it. |
| `train_config.yaml` | `chain_tqe` | Fine-tune the 256² model from the pretrained checkpoint, then quantize and evaluate it. |
| `deploy_config.yaml` | `deployment` | Compile a shipped `.tflite` (`../MODELS/256k2.tflite`) to C and build/flash the STM32H7 app. |
| `eval_config.yaml` | `evaluation` | Measure mAP@0.5 of an INT8 model on a labeled test set. |
| `chain_eqe_config.yaml` | `chain_eqe` | Evaluate a float model → PTQ to int8 → evaluate int8 (PTQ-retention debugging, no retraining). |

ST's stock upstream examples for every mode also live in
[`config_file_examples/`](config_file_examples/) if you need a starting point for
something not covered above.

## Operation modes

You rarely need a new file for a new task — just change `operation_mode` (and the
relevant `model_path`):

| Mode | Pipeline |
|---|---|
| `training` | train a float model only |
| `evaluation` | evaluate a float `.keras` or int8 `.tflite` (mAP@0.5) |
| `quantization` | PTQ a float model to int8 `.tflite` |
| `prediction` | run inference on a folder of images and draw boxes |
| `benchmarking` | measure latency/memory on ST hardware (needs the `tools:` block) |
| `deployment` | generate + build the CubeIDE project for the H7 |
| `chain_tqe` | train → quantize → evaluate |
| `chain_tqeb` | train → quantize → evaluate → benchmark |
| `chain_eqe` | evaluate float → quantize → evaluate int8 |
| `chain_qd` | quantize → deploy |

## Typical workflows

- **Reproduce a model from scratch:** rebuild the dataset
  (`python ../scripts/extract_coco_stop_sign.py`, see [`../DATA.md`](../DATA.md)),
  then `--config-name train_config`. Quantized `.tflite` appears under the run's
  `quantized_models/`.
- **Deploy a shipped model without training:** `--config-name deploy_config`
  (already points at `../MODELS/256k2.tflite`).
- **Score a model:** set `model.model_path` in `eval_config.yaml`, then
  `--config-name eval_config`.
- **Tune PTQ:** point `chain_eqe_config.yaml` at a float `best_model.keras` and
  sweep `dataset.quantization_split`.

## Gotchas baked into these configs

- **`quantization_input_type: uint8` is mandatory** for the H7 app. `int8`
  compiles but the board's image path delivers uint8 pixels → garbage inference.
- **Do not set custom `yolo_anchors`** when fine-tuning a pretrained head. Leaving
  the field out falls through to the modelzoo defaults
  (`tf/src/utils/parse_config.py:449-453`); custom k-means anchors cause logit
  drift that destroys int8 PTQ retention.
- **`confidence_thresh`**: use `~0.001` for mAP evaluation (integrates the full PR
  curve) and `~0.5` for a live demo.
- **Toolchain paths** (`path_to_stedgeai`, `path_to_cubeIDE`) are absolute Windows
  paths — edit them for your install, and avoid any whitespace in them.
