import json
from pathlib import Path

nb_path = Path(__file__).parent / 'train_analysis.ipynb'
print('Checking notebook:', nb_path)
if not nb_path.exists():
    raise SystemExit('Notebook not found')

nb = json.loads(nb_path.read_text(encoding='utf-8'))
code_cells = [c for c in nb.get('cells', []) if c.get('cell_type') == 'code']
errors = []
for i, cell in enumerate(code_cells, start=1):
    src_lines = cell.get('source', [])
    src = '\n'.join(src_lines)
    try:
        compile(src, f'<cell {i}>', 'exec')
    except Exception as e:
        errors.append((i, repr(e), src_lines[:10]))

if not errors:
    print('No syntax errors found in code cells')
else:
    print(f'Found {len(errors)} syntax errors:')
    for cell_no, err, snippet in errors:
        print('\n-- Cell', cell_no, '--')
        print('Error:', err)
        print('First lines of cell:')
        for L in snippet:
            print(L)
    raise SystemExit('Syntax errors detected')
