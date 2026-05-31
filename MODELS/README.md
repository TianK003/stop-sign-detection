# MODELS/

Two INT8 TFLite models from the thesis *"Fine-tuning ST Yolo LC v1 (INT8) for
single-class stop-sign detection on STM32H747I-DISCO"*. Both are single-class
(`stop_sign`) ST Yolo LC v1 detectors, post-training-quantized to INT8 with
`uint8` input, ready to feed to ST Edge AI for the H7 application.

| File | Input | Recipe | COCO val mAP@0.5 | Field mAP (all) | Field mAP (near+mid) | Activation RAM | Weights (Flash) | H7 latency @400 MHz |
|---|---|---|---|---|---|---|---|---|
| `256k2.tflite` | 256×256 | `256_default` anchors + obj-logit rescale **k=2** | **42.3 %** | 32.7 % | **41.6 %** | ~278 KiB (284 972 B) | ~277 KiB (283 368 B) | **320.9 ms** (3.1 FPS) |
| `192k1.tflite` | 192×192 | `192v1` default anchors, **no rescale** (k=0) | 27.9 % | 6.1 % | 7.3 % | ~225 KiB | ~277 KiB | 185.7 ms (5.4 FPS) |

> **Naming note.** `192k1.tflite` is the **k=0** (no obj-logit rescale) 192² model —
> the filename uses `k1` purely as a label; no rescale was applied. `256k2.tflite`
> is the k=2 rescaled 256² model, which is the deployed thesis champion.

Memory/latency are ST Edge AI Developer Cloud benchmarks on STM32H747I-DISCO
(`optimization: balanced`); raw console dumps are in
[`../benchmarking/256k2_benchmark.md`](../benchmarking/256k2_benchmark.md) and
[`../benchmarking/192k0_benchmark.md`](../benchmarking/192k0_benchmark.md). Field
mAP / COCO mAP / F1 come from `operation_mode: evaluation` and
`scripts/threshold_sweep.py`. Operating threshold used in the field analysis is
**T = 0.70**.

## Which one to use

- **`256k2.tflite` — the champion.** Best accuracy; the COCO→Slovenian domain gap is
  essentially zero on signs above the architectural detection floor
  (41.6 % field vs 42.3 % COCO val on near+mid signs). Use this unless latency is critical.
- **`192k1.tflite` — the compact/real-time alternative.** ~1.7× faster, ~53 KiB less
  activation RAM, but ~14 mAP points lower. Useful where 5+ FPS matters more than recall.

## How they were produced

1. Fine-tune ST Yolo LC v1 from the modelzoo's COCO-Person `st_yololcv1` checkpoint on the
   COCO 2017 single-class stop-sign subset (see [`../DATA.md`](../DATA.md)).
2. PTQ to INT8 via TFLite converter with `quantization_input_type: uint8`.
3. For `256k2`, run `scripts/rescale_obj_logits.py --k 2` before re-quantizing to recover
   PTQ retention (the technique is net-positive only when float→int8 retention is < ~80 %,
   which the 256² model needs and the 192² model does not).

End-to-end this is reproducible from a clean checkout with the example configs under
`object_detection/` (see [`object_detection/CONFIGS.md`](../object_detection/CONFIGS.md));
start from the root [`README.md`](../README.md).

## Deploying one of these directly

To skip training/quantization and deploy a shipped model, point the deployment config at it:

```yaml
# object_detection/deploy_config.yaml
model:
  model_path: ../MODELS/256k2.tflite   # or ../MODELS/192k1.tflite
```

then from `object_detection/`: `python stm32ai_main.py --config-name deploy_config`.

## License / provenance

These are INT8 derivatives fine-tuned from STMicroelectronics' publicly released
COCO-Person `st_yololcv1` checkpoint. They inherit the upstream model's license terms
(STMicroelectronics SLA / "Ultimate Liberty"); see the license files preserved under
`object_detection/` and `application_code/`. The fine-tuned weights themselves are
© Tian Ključanin and provided for academic reproducibility of the thesis results.
