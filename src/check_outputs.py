"""check_outputs.py — Cuenta los outputs del notebook ejecutado."""
import json

nb = json.load(open('notebook/gemelo_digital_se_ejecutado.ipynb', encoding='utf-8'))

total = 0
cells_with_output = 0
n_code = 0
n_md = 0
for c in nb['cells']:
    if c['cell_type'] == 'code':
        n_code += 1
        outs = c.get('outputs', [])
        if outs:
            cells_with_output += 1
        total += len(outs)
    elif c['cell_type'] == 'markdown':
        n_md += 1

print(f'Total celdas: {len(nb["cells"])}')
print(f'Code cells: {n_code}')
print(f'Markdown cells: {n_md}')
print(f'Code cells con outputs: {cells_with_output}')
print(f'Total outputs: {total}')