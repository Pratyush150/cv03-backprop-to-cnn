"""Run every example in order and regenerate every committed figure.

Run:  python3 examples/run_all.py

Each script is executed in its own subprocess so that one failing example
cannot leave shared state behind for the next, and so the exit code is the
honest answer to "does the whole set still run".
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

SCRIPTS = [
    "01_one_neuron.py",
    "02_why_nonlinearity.py",
    "03_backprop_mlp.py",
    "04_gradient_check.py",
    "05_softmax_stability.py",
    "06_convolution.py",
    "07_im2col_speedup.py",
    "08_pooling_shapes.py",
    "09_train_cnn.py",
]


def main() -> int:
    failures = []
    for name in SCRIPTS:
        print(f"\n### {name}", flush=True)
        t0 = time.perf_counter()
        result = subprocess.run([sys.executable, str(HERE / name)], check=False)
        dt = time.perf_counter() - t0
        status = "ok" if result.returncode == 0 else f"FAILED ({result.returncode})"
        print(f"### {name}: {status} in {dt:.1f}s", flush=True)
        if result.returncode != 0:
            failures.append(name)
    print("\n" + "=" * 60)
    if failures:
        print("FAILED:", ", ".join(failures))
        return 1
    print(f"all {len(SCRIPTS)} examples ran; figures are in docs/figures/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
