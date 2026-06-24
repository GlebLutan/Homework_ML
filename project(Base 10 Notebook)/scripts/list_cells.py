import nbformat
from pathlib import Path

nb_path = Path(__file__).resolve().parent.parent / "[Base]_Boosting_part1_[2025_2026].ipynb"
nb = nbformat.read(nb_path, as_version=4)
for i, cell in enumerate(nb.cells):
    if i >= 50:
        snippet = ''.join(cell.source).strip().split('\n')[0][:80]
        print(f"Cell {i}: {cell.cell_type} - {snippet}")
