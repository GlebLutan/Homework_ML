import nbformat, pathlib
nb_path = pathlib.Path(r"c:/Users/glebl/Desktop/Проекты для VSCode/projects_Notebooks/project(Base 10 Notebook)/[Base]_Boosting_part1_[2025_2026].ipynb")
nb = nbformat.read(nb_path, as_version=4)
for idx, cell in enumerate(nb.cells):
    if cell.cell_type == 'markdown' and '**Задание' in ''.join(cell.source):
        print('-' * 80)
        print(f"Cell #{idx}: \n{''.join(cell.source)}")
        print('Next code cells:')
        printed = 0
        offset = 1
        while printed < 2 and idx + offset < len(nb.cells):
            next_cell = nb.cells[idx + offset]
            if next_cell.cell_type == 'code':
                print(f"--- code cell #{idx+offset} ---")
                print(''.join(next_cell.source))
                printed += 1
            offset += 1
