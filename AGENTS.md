# AGENTS.md — Reglas para agentes que trabajen en este workspace

## Que es este proyecto

Proyecto de investigacion sobre **deteccion temprana no supervisada en
subestaciones electricas** mediante gemelo digital ligero, conformal
prediction y aprendizaje federado. Consta de tres entregables
principales:

- **dashboard/**: panel interactivo D3.js que muestra el pipeline en
  operacion. No requiere backend (solo `index.html` + `js/` + `css/`).
- **notebook/gemelo_digital_se.ipynb**: notebook Jupyter ejecutable
  con el pipeline completo (gemelo, autoencoder, conformal, FedAvg).
- **informe/**: documento LaTeX tipo tesis con siete capitulos,
  bibliografia y tres apendices.

## Comandos utiles

```bash
# Regenerar el notebook desde codigo
python src/build_notebook.py

# Ejecutar el notebook (5-15 min en CPU)
python -m jupyter nbconvert --to notebook --execute \
    notebook/gemelo_digital_se.ipynb \
    --output gemelo_digital_se_ejecutado.ipynb \
    --ExecutePreprocessor.timeout=900

# Servir el dashboard
cd dashboard && python -m http.server 8765
# abrir http://localhost:8765/

# Compilar el informe LaTeX (requiere pdflatex + bibtex)
cd informe && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Reglas duras

- **NO incluir titulos universitarios** en ningun entregable publico. Solo
  el nombre en bold y los 3 botones de contacto. Ver regla user-level
  en la memoria principal.
- **NO usar bash-style escapes** dentro de strings Python (PowerShell
  en Windows). El bug `\"\"\"` vs `'''` ya consumio tiempo.
- **Encoding**: archivos en UTF-8 sin BOM. Para batch edit in shell usar
  Python con `encoding='utf-8'`.
- **Verificar antes de borrar**: este proyecto no tiene git, todos los
  archivos son entregables.
- **Cada cambio en el codigo** que afecta el comportamiento debe
  reflejarse en el notebook (re-generar via `build_notebook.py`).
- **Footer del dashboard**: ya esta implementado con la regla hard
  aplicada. NO modificar para anadir titulos.

## Convenciones de estilo editorial (del user memory)

- Texto en espanol de Chile.
- Tono directo, sin hedge words (NO "podria", "resulta relevante",
  "muestra que").
- Mirada editorial con opinion, no resumen tibio.
- Datos como evidencia, no como contenido acumulativo.
- En secciones tecnicas del paper: opinion directa + numeros duros.

## Estructura canonica

```
.
├── dashboard/                      # D3.js (servir con http.server, no file://)
│   ├── index.html
│   ├── css/style.css
│   └── js/
│       ├── main.js                 # Orquestador del stream + controles
│       ├── data.js                  # Gemelo + generador sintetico (22 feats)
│       ├── topology.js             # Unifilar SVG animado
│       ├── signals.js              # Series SCADA/PMU obs vs esperado
│       ├── residuals.js            # Score + contribuciones + heatmap 22xN
│       ├── conformal.js            # Detector conformal en vivo
│       ├── federated.js            # Mapa FedAvg + convergencia
│       ├── kpis.js                 # KPIs con sparklines
│       ├── timeline.js             # Timeline de eventos 4 h
│       └── metrics.js              # Tabla de metricas (vivo + notebook)
│
├── notebook/                       # Jupyter ejecutable
│   ├── gemelo_digital_se.ipynb
│   ├── gemelo_digital_se_ejecutado.ipynb   # con outputs
│   ├── data/                       # tabla_metricas.csv, residuos npz
│   └── figures/                    # figuras generadas
│
├── informe/                        # LaTeX tipo tesis
│   ├── main.tex
│   ├── chapters/                    # 7 capitulos + abstract
│   ├── appendix/                   # 3 apendices
│   └── references.bib
│
├── src/                            # Scripts auxiliares
│   ├── build_notebook.py           # Genera el .ipynb
│   ├── check_outputs.py            # Cuenta outputs del ejecutado
│   └── fix_docstrings.py           # Fix escape de docstrings
│
├── AGENTS.md                        # Este archivo
└── README.md
```

## Handoff

Sesion de creacion inicial: 2026-10-02. Notebook y dashboard operativos.
Informe LaTeX completo pero no compilado a PDF en este momento
(requiere entorno LaTeX local).

Sesion de auditoria y correccion: 2026-10-02. Fixes: orden de
inyeccion de fallas en el generador del notebook (antes las firmas se
sobrescribian y el TPR era ~0), split temporal por bloques 70/15/15,
calibracion conformal solo-normal con correccion de muestra finita,
cobertura garantizada por tipo de falla en cada bloque, 22 features
reales (S_total, cos_phi, desbalance), y dashboard reconstruido
(KPIs, controles de velocidad/inyeccion, timeline, heatmap corregido,
crosslinks entre paneles). Metricas test reproducibles (seed 42):
FPR 4.91%, TPR 76.3% (central) / 80.5% (federado), latencia 0 s.

Sesion de versionado: 2026-10-02. Repositorio inicializado en
github.com/mortizcoilla/gemelo_digital (rama main). Dashboard listo
para Vercel: D3 v7 vendido en dashboard/vendor/ (sin CDN externo),
vercel.json en la raiz con outputDirectory=dashboard; el deploy
sirve el dashboard en / y el resto del repo queda fuera del build.