import nbformat
from pathlib import Path

nb_path = Path(__file__).resolve().parent.parent / "[Base]_Boosting_part1_[2025_2026].ipynb"
nb = nbformat.read(nb_path, as_version=4)
start, end = 55, 57
for i in range(start, end + 1):
    cell = nb.cells[i]
    print('-' * 60)
    print(f"Cell {i} ({cell.cell_type}):\n" + ''.join(cell.source))
