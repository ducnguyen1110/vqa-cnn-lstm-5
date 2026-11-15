"""Validate notebook schema and compile Python cells without running training."""
import ast
from pathlib import Path
import nbformat

root = Path(__file__).resolve().parents[1]
path = root / 'notebooks' / 'vqa_cnn_lstm.ipynb'
notebook = nbformat.read(path, as_version=4)
nbformat.validate(notebook)
count = 0
for index, cell in enumerate(notebook.cells):
    if cell.cell_type != 'code':
        continue
    # IPython package-install commands are not Python syntax.
    source = '\n'.join(line for line in cell.source.splitlines()
                       if not line.lstrip().startswith(('%', '!')))
    ast.parse(source, filename=f'{path.name}:cell-{index}')
    if cell.outputs or cell.execution_count is not None:
        raise ValueError(f'Cell {index} contains stale execution state')
    count += 1
print(f'Validated {path.name}: {count} code cells; no stale outputs.')
