# src/analysis/viz.py
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt

def savefig(path: str | Path, dpi: int = 200) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
