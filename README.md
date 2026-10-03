# Gemelo Digital Ligero de Subestacion

Deteccion temprana no supervisada con SCADA/PMU, alertas conformal y aprendizaje
federado entre subestaciones.

## Descripcion

Este proyecto implementa un pipeline de cuatro componentes para deteccion
temprana de fallas en subestaciones electricas de distribucion:

1. **Gemelo digital ligero** que predice el comportamiento electrico esperado
   y genera datos sinteticos etiquetados.
2. **Detector LSTM autoencoder** sobre el residuo multivariado.
3. **Conformal prediction** para fijar el umbral de alarma con tasa de falsa
   alarma garantizada (alpha = 0.05).
4. **Aprendizaje federado (FedAvg)** entre tres subestaciones virtuales,
   preservando la privacidad de los datos.

## Estructura del repositorio

```
.
├── dashboard/                       # Dashboard D3.js interactivo
│   ├── index.html
│   ├── css/style.css
│   └── js/
│       ├── main.js                  # Orquestador del stream + controles
│       ├── data.js                  # Gemelo + generador sintetico (22 features)
│       ├── topology.js              # Unifilar animado con breakers y flujo
│       ├── signals.js               # Series SCADA/PMU obs vs esperado
│       ├── residuals.js             # Score conformal + contribuciones + heatmap
│       ├── conformal.js             # Detector conformal en vivo
│       ├── federated.js             # Mapa FedAvg + convergencia
│       ├── kpis.js                  # Strip de KPIs con sparklines
│       ├── timeline.js              # Timeline de eventos de las 4 h
│       └── metrics.js               # Tabla de metricas (vivo + notebook)
│
├── notebook/
│   ├── gemelo_digital_se.ipynb              # Notebook ejecutable
│   ├── gemelo_digital_se_ejecutado.ipynb    # Notebook con outputs
│   ├── data/                                 # tabla_metricas.csv, npz
│   └── figures/                              # figuras del notebook
│
├── informe/                         # Informe tipo tesis en LaTeX
│   ├── main.tex
│   ├── chapters/
│   ├── appendix/
│   └── references.bib
│
├── src/                             # Scripts auxiliares
│   ├── build_notebook.py            # Genera el notebook desde codigo
│   ├── check_outputs.py             # Cuenta outputs del notebook ejecutado
│   └── fix_docstrings.py            # Fix para escape de docstrings
│
└── README.md
```

## Ejecucion

### Notebook Jupyter

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install jupyter ipykernel scikit-learn
python src/build_notebook.py
python -m jupyter nbconvert --to notebook --execute \
    notebook/gemelo_digital_se.ipynb \
    --output gemelo_digital_se_ejecutado.ipynb \
    --ExecutePreprocessor.timeout=600
```

Tarda ~6 minutos en CPU.

### Dashboard

```bash
# Servir en local (requerido: los modulos ES no cargan via file://)
cd dashboard && python -m http.server 8765
# abrir http://localhost:8765/
```

No requiere backend: solo estaticos + D3.js v7 por CDN. Controles:
pausa, velocidad 1x/4x/16x, inyeccion manual de fallas (F1-F8),
reinicio. Click en alimentadores, filas del heatmap o barras de
contribucion para cruzar con el panel de senales.

### Informe LaTeX

```bash
cd informe
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Resultados principales

| Detector      | FPR   | TPR   | F1   | AUC   |
|--------------|-------|-------|------|-------|
| 3-sigma      | 12.8% | 36.8% | 0.42 | 0.639 |
| Centralizado |  4.9% | 76.3% | 0.80 | 0.878 |
| Federado     |  5.0% | 80.5% | 0.87 | 0.906 |

Cifras del prototipo ejecutable incluido en este repo (bloque de test,
split temporal 70/15/15, semilla 42, ~5 min de CPU). El prototipo
alcanza el rango de la literatura para la misma metodologia
(TPR 75-85%) con configuracion reducida.

**Lo que demuestra el prototipo**:
- El pipeline gemelo digital + autoencoder + conformal + FedAvg
  funciona de punta a punta.
- El conformal prediction cumple la tasa de falsa alarma garantizada
  (4.86% empirico vs 5.0% objetivo) con latencia mediana de
  deteccion de 0 s.
- El modelo federado no cede detectabilidad frente al centralizado
  y mueve 0 MB de datos crudos fuera de las SEs (NERC CIP / CNE).
- El dashboard interactivo opera en tiempo real con inyeccion manual
  de fallas y deteccion explicada por feature.

## Licencia

MIT.

## Autor

Miguel Ortiz Coilla