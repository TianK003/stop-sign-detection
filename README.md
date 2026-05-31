# STM32H7 stop-sign detection — bachelor's thesis code release

Public code companion to a FRI UL bachelor's thesis:
*fine-tuning ST Yolo LC v1 (INT8) on a single-class "stop sign" subset of COCO 2017,
deployed to an STM32H747I-DISCO + B-CAMS-OMV for live detection of real Slovenian STOP
signs.* It contains the training/quantization/evaluation pipeline, the STM32H7 application
code, the thesis-specific tooling, and the two final INT8 models — **no image datasets**
(see [`DATA.md`](DATA.md)).

---

## Origin & attribution

This repository is a **trimmed, standalone snapshot** (2026-05-11) of
[STMicroelectronics/stm32ai-modelzoo-services](https://github.com/STMicroelectronics/stm32ai-modelzoo-services),
reduced to just the `object_detection` use case needed for the thesis. Upstream changes
after the snapshot date are not tracked.

**Upstream ST code retains its original licenses** — Apache 2.0 for `api/`, `common/`, and
the `object_detection/` Python sources; the STMicroelectronics SLA / "Ultimate Liberty"
license for the `application_code/.../STM32H7/` tree and its middleware/BSP components.
Every upstream `LICENSE.md` / `LICENSE.txt` is preserved in place. The top-level
[`LICENSE.md`](LICENSE.md) is the upstream license bill-of-materials.

**Thesis-specific additions (not part of the upstream ST modelzoo):**

- `scripts/` — dataset extraction, PTQ diagnostics, obj-logit rescale, anchor clustering,
  field-attribute analysis, threshold sweep, plotting.
- The config template + example configs and `CONFIGS.md` under `object_detection/`.
- `MODELS/` — the two final INT8 models (derivatives of ST's pretrained checkpoint; see
  [`MODELS/README.md`](MODELS/README.md)).
- `benchmarking/` — saved ST Edge AI Developer Cloud benchmark output.
- This `README.md`, [`DATA.md`](DATA.md), and `.gitignore` / `.gitattributes`.

Use cases other than `object_detection/`, the PyTorch (`pt/`) tree, ST's `tutorials/`, and
all image data have been removed as out-of-scope for this release.

---

## Thesis at a glance

**Goal.** Fine-tune ST Yolo LC v1 (INT8, single-class `stop_sign`) from the modelzoo's
COCO-Person checkpoint, deploy to STM32H747I-DISCO + B-CAMS-OMV, and field-test on real
Slovenian street photos.

**Target hardware.**

| Component | Part |
|---|---|
| Discovery board | **STM32H747I-DISCO** (dual-core Cortex-M7 + M4, 2 MiB Flash, 1 MiB SRAM, 512 KiB AXI-SRAM, LTDC, DCMI, SDRAM) |
| Camera | **B-CAMS-OMV** bundle |

**Tool stack.**

| Tool | Version | Notes |
|---|---|---|
| Python | 3.12.9 | TF 2.18.0 / Keras 3.8.0 wheels |
| ST Edge AI Core | 4.0.0 | Local at `C:/ST/STEdgeAI/4.0/`; Dev Cloud (https://stedgeai-dc.st.com/) for the benchmark numbers |
| STM32CubeIDE | **1.17.0** (pinned) | Newer versions have caused path-resolution bugs in `chain_qd` |

---

## Quick-glance results

Two INT8 models were trained, evaluated, and deployed; the 256² model is the thesis
champion, the 192² model is kept as a compact comparison point. Field set =
118 Slovenian street photos, 148 labeled stop-sign boxes, evaluated at `mAP@0.5`. Both
shipped models are in [`MODELS/`](MODELS/).

| Model | Input | COCO val mAP | Field mAP (all) | Field mAP (near + mid only) | F1 @ best threshold | Flash | Activation RAM | Inference latency (H7 @ 400 MHz) |
|---|---|---|---|---|---|---|---|---|
| **256_default + k=2** (champion, `MODELS/256k2.tflite`) | 256² | 42.3 % | 32.7 % | **41.6 %** | 0.476 @ T=0.70 | 308 KiB | 278 KiB | **320.9 ms** (3.1 FPS) |
| 192v1 + k=0 (compact alt, `MODELS/192k1.tflite`) | 192² | 27.9 % | 6.1 % | 7.3 % | 0.254 @ T=0.70 | 308 KiB | 225 KiB | 185.7 ms (5.4 FPS) |

Memory and latency are ST Edge AI Developer Cloud benchmarks on STM32H747I-DISCO
(`optimization: balanced`). Field/COCO mAP and the F1 sweep come from
`operation_mode: evaluation` and `scripts/threshold_sweep.py`.

**Headline finding (256²).** On the subset of field signs above the architectural
detection floor (`distance ∈ {near, mid}`), the COCO→Slovenian domain gap is essentially
zero (**41.6 % field vs 42.3 % COCO val, ~0.7 pp drop**). The full-set gap is dominated by
far-distance boxes below the smallest anchor's effective scale at 256² input — an
architectural ceiling, not a generalization failure.

---

## Methodology summary

1. **Dataset.** Filter COCO 2017 to the single `stop sign` category, remap
   `category_id 13 → 0`, download just the matching images
   (`scripts/extract_coco_stop_sign.py`) → ~1 700 train / ~75 val. Field test:
   118 Slovenian street photos labeled in CVAT with 5 per-box attributes. See
   [`DATA.md`](DATA.md) — **no images are shipped**.
2. **Model.** ST Yolo LC v1, YOLOv2-style anchor head, fine-tuned from the COCO-Person
   pretrained checkpoint. Resolution 192² (compact) or 256² (champion). **Anchors are the
   pretrained defaults** — custom k-means anchors break PTQ retention via logit drift.
3. **Training.** Hydra/YAML-driven. Adam + LR warmup/cosine decay, EarlyStopping on
   `val_map`. `random_flip: horizontal` on (the octagon is symmetric); aggressive
   resize/crop off (would destroy small distant signs).
4. **Quantization.** TFLite PTQ to INT8, `quantization_input_type: uint8` (mandatory for
   the H7 app — `int8` input compiles but passes garbage pixels). For runs where naïve PTQ
   collapses, post-hoc head rescale via `scripts/rescale_obj_logits.py` — the champion is
   k=2.
5. **Deployment.** ST Edge AI Developer Cloud generates C sources from the `.tflite`;
   `chain_qd` / `operation_mode: deployment` drives STM32CubeIDE 1.17.0 headless to build
   and flash. The generated project lands under `application_code/object_detection/STM32H7/`.
6. **Evaluation.** `evaluation` (mAP@0.5), `prediction` (visualized boxes),
   `benchmarking` (Dev Cloud latency/memory), plus `scripts/analyze_field_attributes.py`
   (per-attribute recall + FN/FP taxonomy) and `scripts/threshold_sweep.py` (best-F1
   operating point).

---

## Setup

### Prerequisites

- **Python 3.12.9.** On Windows, tick "Add python.exe to PATH" during install.
- **ST Edge AI Core 4.0** locally at `C:/ST/STEdgeAI/4.0/` (or a Dev Cloud account at
  https://stedgeai-dc.st.com/). The `path_to_stedgeai` field in the example configs
  assumes the local path.
- **STM32CubeIDE 1.17.0** at `C:/ST/STM32CubeIDE_1.17.0/` — pin to 1.17.0.
- **Avoid whitespace in any toolchain path.** The deployment scripts break on spaces.
- (Optional) NVIDIA GPU for training — 3–6 h on CPU vs 30–90 min on a recent GPU.

### Environment

From the repo root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

The STM32H7 application code (`application_code/object_detection/STM32H7/`) is included
directly in this repository — no submodule init is required.

### Pretrained weights (sibling repo)

The pretrained COCO-Person checkpoint used as the fine-tuning starting point lives in the
separate `stm32ai-modelzoo` repo. Clone it next to this one so the configs' relative path
`../../stm32ai-modelzoo/...` resolves:

```powershell
cd ..
git clone --depth 1 https://github.com/STMicroelectronics/stm32ai-modelzoo.git
```

### Running the pipeline

All operations go through one entry point driven by YAML:

```powershell
cd object_detection
python stm32ai_main.py --config-name <config_name>
```

Rather than the ~30 ablation configs from the thesis, this release ships one
annotated template plus four ready-to-run examples. Full details, the
`operation_mode` reference, and how to adapt them are in
[`object_detection/CONFIGS.md`](object_detection/CONFIGS.md).

| Config | `operation_mode` | Purpose |
|---|---|---|
| `config_template.yaml` | (all) | Annotated reference for every field — copy blocks from it |
| `train_config.yaml` | `chain_tqe` | Fine-tune the 256² model → quantize → evaluate |
| `deploy_config.yaml` | `deployment` | Compile + flash a shipped `.tflite` to the H7 |
| `eval_config.yaml` | `evaluation` | mAP@0.5 of an INT8 model on a labeled test set |
| `chain_eqe_config.yaml` | `chain_eqe` | Evaluate float → PTQ int8 → evaluate int8 (no retrain) |

**Fast path — deploy a shipped model without retraining.** `deploy_config.yaml`
already points at `../MODELS/256k2.tflite` (swap to `../MODELS/192k1.tflite` for
the compact alternative); run `python stm32ai_main.py --config-name deploy_config`.

**Full reproduction from scratch:**

1. `python scripts/extract_coco_stop_sign.py` — rebuild the COCO subset (see `DATA.md`).
2. `cd object_detection && python stm32ai_main.py --config-name train_config` —
   train + quantize the 256² champion.
3. Deploy: set `model.model_path` in `deploy_config.yaml` to the resulting
   `quantized_model.tflite`, run `python stm32ai_main.py --config-name deploy_config`.
4. `python stm32ai_main.py --config-name eval_config` to score it (edit `model_path`).
5. From repo root: `python scripts/analyze_field_attributes.py` and
   `python scripts/threshold_sweep.py`.

---

## Repository layout

| Path | Contents |
|---|---|
| `MODELS/` | The two final INT8 models + their stats — see [`MODELS/README.md`](MODELS/README.md) |
| `object_detection/` | `stm32ai_main.py`, `tf/` (model + train + quantize + eval + postprocess), the config template + examples and [`CONFIGS.md`](object_detection/CONFIGS.md), ST's stock `config_file_examples/`, `datasets/` (code + templates only, **no images**) |
| `application_code/object_detection/STM32H7/` | The H7 OD application (CubeMX `.ioc`, BSP, OD glue). Build output (`Debug/`, `Release/`) is gitignored and regenerated by CubeIDE |
| `common/` | Framework-agnostic services: benchmarking, quantization, deployment, `stm32ai_dc/` (Dev Cloud client), `stm32ai_local/` (Core CLI wrapper) |
| `api/api.py` | Factory dispatcher: `get_model`, `get_trainer`, `get_quantizer`, etc. |
| `scripts/` | Thesis-specific tooling |
| `benchmarking/` | Saved ST Edge AI Developer Cloud benchmark output for both models |
| `DATA.md` | Dataset provenance + how to reconstruct (no data shipped) |

---

## Practical notes

> [!CAUTION]
> Any whitespace in the Python, STM32CubeIDE, or ST Edge AI Core path will break the
> deployment scripts. Avoid `C:\Program Files\…` installs for these tools.

> [!CAUTION]
> `quantization_input_type` must be `uint8` for the H7 app. Setting `int8` compiles but
> the H7 image path delivers uint8 pixels, so inference receives garbage. The most
> expensive footgun in the OD path.

> [!NOTE]
> Windows long-path limit (256 chars) can break MLflow logging deep inside Hydra output
> dirs. Enable long paths via Registry
> (`HKLM\SYSTEM\CurrentControlSet\Control\FileSystem\LongPathsEnabled = 1`) and
> `git config --system core.longpaths true`.
