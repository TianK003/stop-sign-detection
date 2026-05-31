#!/usr/bin/env python
"""
measure_latency_local.py  --  fully off-cloud on-device latency for the thesis.

Measures real inference latency of the deployed INT8 model on the physically
connected STM32H747I-DISCO, using ST's AiRunner harness over the ST-LINK virtual
COM port. This is the local equivalent of the ST Edge AI Developer Cloud
"Duration (ms)" figure -- same proto-buffer protocol, same board, no cloud.

PREREQUISITE (one-time): the board must be running ST's *aiValidation* /
*System Performance* firmware, NOT the object-detection demo. The demo answers
serial but rejects the protocol ("E801 HwIOError: Invalid firmware"). Generate +
flash the validation FW once via CubeMX -> X-CUBE-AI -> "Validation" application
(see the companion notes), then run this script. Re-flash the demo afterwards.

Usage (from repo root, st_zoo env active, board on COMx):
    python scripts/measure_latency_local.py --desc serial:COM4:115200 --runs 20
    python scripts/measure_latency_local.py --desc serial:COM4:115200 --res 192

The AiRunner package ships with ST Edge AI Core; we add it to sys.path here.
"""
import argparse
import statistics
import sys
from pathlib import Path

# ST Edge AI Core bundles the AiRunner harness here on this machine.
AI_RUNNER_PATH = r"C:\ST\STEdgeAI\4.0\scripts\ai_runner"
if AI_RUNNER_PATH not in sys.path:
    sys.path.insert(0, AI_RUNNER_PATH)

try:
    import numpy as np
    from stm_ai_runner import AiRunner
except ImportError as exc:  # pragma: no cover - environment guard
    sys.exit(
        f"[FATAL] could not import AiRunner/numpy ({exc}).\n"
        f"        Check that {AI_RUNNER_PATH!r} exists and numpy is installed."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Local on-device latency via AiRunner")
    ap.add_argument("--desc", "-d", default="serial:COM4:115200",
                    help="AiRunner descriptor (serial[:COMPORT][:baud]). Board on COM4 here.")
    ap.add_argument("--runs", "-n", type=int, default=20,
                    help="number of inferences to average")
    ap.add_argument("--res", type=int, default=None,
                    help="input side length (e.g. 192 or 256); auto-detected from FW if omitted")
    ap.add_argument("--per-layer", action="store_true",
                    help="request per-layer timing breakdown (slower)")
    args = ap.parse_args()

    runner = AiRunner()
    print(f"[*] connecting to {args.desc} ...", flush=True)
    runner.connect(args.desc)
    if not runner.is_connected:
        print("[FATAL] no c-model on the board. Is the aiValidation/System-Performance "
              "FW flashed? (the OD demo will NOT work here.)")
        print(f"        AiRunner error: {runner.get_error()}")
        return 1

    print()
    print(runner, flush=True)
    runner.summary()

    # Random uint8 frame matching the network input (auto-detected unless --res given).
    if args.res is not None:
        inputs = [np.random.randint(0, 256, size=(1, args.res, args.res, 3), dtype=np.uint8)]
    else:
        inputs = runner.generate_rnd_inputs(batch_size=1)

    mode = AiRunner.Mode.IO_ONLY
    if args.per_layer:
        mode = AiRunner.Mode.PER_LAYER

    durations = []
    print(f"\n[*] running {args.runs} inference(s) ...", flush=True)
    for i in range(args.runs):
        _, profile = runner.invoke(inputs, mode=mode)
        d = statistics.mean(profile["c_durations"])
        durations.append(d)
        print(f"    run {i + 1:>3}/{args.runs}: {d:8.3f} ms", flush=True)

    runner.disconnect()

    print("\n==================== RESULT (off-cloud, on STM32H747I-DISCO) ====================")
    print(f" samples           : {len(durations)}")
    print(f" mean latency      : {statistics.mean(durations):.3f} ms")
    print(f" min / max         : {min(durations):.3f} / {max(durations):.3f} ms")
    if len(durations) > 1:
        print(f" stdev             : {statistics.pstdev(durations):.3f} ms")
        print(f" throughput        : {1000.0 / statistics.mean(durations):.2f} FPS")
    print("=================================================================================")
    print("Cite alongside the Dev Cloud row in tab:memlatency; both use Core v4.0.0-20500,")
    print("optimization=balanced, STM32H747I-DISCO @400 MHz.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
