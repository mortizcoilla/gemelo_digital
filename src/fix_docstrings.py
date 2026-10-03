"""fix_docstrings.py — Reemplaza los escapes problematicos en build_notebook.py."""
from pathlib import Path

src_path = Path('src/build_notebook.py')
content = src_path.read_text(encoding='utf-8')

# Reemplazar los triples-comillas-doble-escapados por triples-comillas-sencillas.
# Patron problematico: \"\"\"  (que aparece cuando el codigo Python fuente
# quiere escribir """  dentro de un raw string r"""...""" )
# Lo cambiamos a "'''" para evitar el conflicto de cierre de string.
content = content.replace('\\"\\"\\"', "'''")

src_path.write_text(content, encoding='utf-8')
print(f'Hecho. Tamanho: {len(content)} chars')