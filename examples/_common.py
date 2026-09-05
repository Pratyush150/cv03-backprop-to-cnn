"""Shared plumbing for the example scripts.

matplotlib is imported here and nowhere in src/netfs. The library -- and the
test suite -- therefore run on a headless machine with no plotting stack
installed, which is the normal state of a CI runner.

Figures are written to docs/figures/ on a WHITE background. These are teaching
materials meant to be read next to the text, printed, and pasted into a README,
not decoration for a dark portfolio page.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
FIGURES = REPO_ROOT / "docs" / "figures"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def figure(*args: Any, **kwargs: Any):
    """A matplotlib figure on a non-interactive backend with a light theme."""
    import matplotlib

    matplotlib.use("Agg")  # never needs a display
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
    })
    return plt.subplots(*args, **kwargs)


def save(fig, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"wrote {path.relative_to(REPO_ROOT)}")
    return path


def rule(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
