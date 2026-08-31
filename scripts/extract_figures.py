"""Extract the figures embedded in an executed notebook into outputs/figures/.

The notebook renders its figures inline, so a run that predates the `savefig`
calls leaves `outputs/figures/` empty even though the images exist as base64
PNGs inside the `.ipynb`. This pulls them out without retraining anything.

Usage:
    python scripts/extract_figures.py [notebook.ipynb]
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NB = REPO_ROOT / "notebooks" / "btr_transformer.ipynb"
FIG_DIR = REPO_ROOT / "outputs" / "figures"

# Cell index -> output filename, so extracted names match the savefig names.
KNOWN_FIGURES = {
    42: "09_training_curves_base.png",
    46: "10_architecture_grid.png",
}


def main() -> int:
    nb_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_NB
    if not nb_path.exists():
        print(f"notebook not found: {nb_path}")
        return 1

    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    written = 0
    for idx, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        images = [
            out["data"]["image/png"]
            for out in cell.get("outputs", [])
            if "data" in out and "image/png" in out["data"]
        ]
        for n, payload in enumerate(images):
            name = KNOWN_FIGURES.get(idx)
            if name is None or len(images) > 1:
                name = f"cell{idx:02d}_{n}.png"
            target = FIG_DIR / name
            target.write_bytes(base64.b64decode(payload))
            print(f"cell {idx:>2} -> {target.relative_to(REPO_ROOT)} ({target.stat().st_size:,} bytes)")
            written += 1

    if not written:
        print(f"no embedded images found in {nb_path.name} - was it executed?")
        return 1
    print(f"\n{written} figure(s) written to {FIG_DIR.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
