// residuals.js — Panel de deteccion:
//  - ScorePlot: score de reconstruccion + umbral conformal q_hat
//  - FeatureBars: top features que impulsan el score actual
//  - HeatmapPlot: mapa de calor 22 features x tiempo (streaming)

import { FEATURES, N_FEATURES, FEATURE_LABELS } from './data.js';

function formatT(t) {
  const m = Math.floor(t / 60);
  const s = Math.floor(t % 60);
  return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
}

// -------------------------------------------------------------
// Score del detector vs umbral conformal
// -------------------------------------------------------------
export class ScorePlot {
  constructor(container, tooltip) {
    this.container = container;
    this.tooltip = tooltip;
    this.maxPoints = 180;
    this.data = [];   // {t, score, alert, inEvent}
    this.q_hat = null;
    this.onHover = null;
    this._init();
  }

  _init() {
    const W = 860, H = 170;
    const M = { top: 14, right: 64, bottom: 24, left: 56 };

    const svg = d3.select(this.container)
      .append('svg')
      .attr('viewBox', `0 0 ${W} ${H}`)
      .style('width', '100%')
      .style('height', 'auto');

    this.svg = svg;
    this.g = svg.append('g').attr('transform', `translate(${M.left},${M.top})`);
    this.W = W - M.left - M.right;
    this.H = H - M.top - M.bottom;

    this.g.append('g').attr('class', 'fault-layer');
    this.g.append('g').attr('class', 'alert-layer');
    this.g.append('g').attr('class', 'line-layer');
    this.g.append('g').attr('class', 'threshold-layer');
    this.g.append('g').attr('class', 'cursor-layer');
    this.g.append('g').attr('class', 'x-axis').attr('transform', `translate(0,${this.H})`);
    this.g.append('g').attr('class', 'y-axis');

    this.x = d3.scaleLinear().range([0, this.W]);
    this.y = d3.scaleLinear().range([this.H, 0]);
    this.line = d3.line()
      .x(d => this.x(d.t))
      .y(d => this.y(d.score))
      .curve(d3.curveMonotoneX);

    const legend = svg.append('g').attr('transform', `translate(${M.left + 8}, 0)`);
    legend.append('text').attr('x', 0).attr('y', 8).attr('fill', '#aab3c0')
      .attr('font-size', 10).text('Score de reconstruccion');
    legend.append('line').attr('x1', 168).attr('x2', 184).attr('y1', 5).attr('y2', 5)
      .attr('stroke', '#e63946').attr('stroke-width', 1.4).attr('stroke-dasharray', '6 3');
    legend.append('text').attr('x', 188).attr('y', 8).attr('fill', '#aab3c0')
      .attr('font-size', 10).text('q-hat (conformal)');

    this.vLine = this.g.select('.cursor-layer').append('line')
      .attr('class', 'crosshair')
      .attr('y1', 0).attr('y2', this.H)
      .style('visibility', 'hidden');
    this.cursorDot = this.g.select('.cursor-layer').append('circle')
      .attr('r', 3).attr('fill', '#b294ff').style('visibility', 'hidden');

    svg.append('rect')
      .attr('x', M.left).attr('y', M.top)
      .attr('width', this.W).attr('height', this.H)
      .attr('fill', 'transparent')
      .on('mousemove', ev => this._onMove(ev))
      .on('mouseleave', () => this._onLeave());
  }

  setThreshold(q) { this.q_hat = q; }

  push(entry) {
    this.data.push(entry);
    if (this.data.length > this.maxPoints) this.data.shift();
    this._render();
  }

  _onMove(ev) {
    if (this.data.length < 2) return;
    const [mx] = d3.pointer(ev, this.g.node());
    const t = this.x.invert(Math.max(0, Math.min(this.W, mx)));
    let best = this.data[0];
    for (const d of this.data) {
      if (Math.abs(d.t - t) < Math.abs(best.t - t)) best = d;
    }
    const x = this.x(best.t);
    this.vLine.attr('x1', x).attr('x2', x).style('visibility', 'visible');
    this.cursorDot.attr('cx', x).attr('cy', this.y(best.score)).style('visibility', 'visible');
    this.tooltip.show(`
      <div class="tt-title">Detector · ${formatT(best.t)}</div>
      <div class="tt-row"><span>Score</span><b>${best.score.toFixed(3)}</b></div>
      ${this.q_hat !== null ? `<div class="tt-row"><span>q-hat</span><b>${this.q_hat.toFixed(3)}</b></div>` : ''}
      <div class="tt-row"><span>Estado</span><b style="color:${best.alert ? '#e63946' : '#6ad36e'}">${best.alert ? 'ALERTA' : 'normal'}</b></div>
      ${best.eventName ? `<div class="tt-row"><span>Evento</span><b>${best.eventName}</b></div>` : ''}
    `, ev);
    if (this.onHover) this.onHover(best.t);
  }

  _onLeave() {
    this.vLine.style('visibility', 'hidden');
    this.cursorDot.style('visibility', 'hidden');
    this.tooltip.hide();
    if (this.onHover) this.onHover(null);
  }

  showCursorAt(t) {
    if (t === null || this.data.length < 2) {
      this.vLine.style('visibility', 'hidden');
      this.cursorDot.style('visibility', 'hidden');
      return;
    }
    let best = this.data[0];
    for (const d of this.data) {
      if (Math.abs(d.t - t) < Math.abs(best.t - t)) best = d;
    }
    const x = this.x(best.t);
    this.vLine.attr('x1', x).attr('x2', x).style('visibility', 'visible');
    this.cursorDot.attr('cx', x).attr('cy', this.y(best.score)).style('visibility', 'visible');
  }

  _render() {
    if (this.data.length < 2) return;
    const tmin = this.data[0].t, tmax = this.data[this.data.length - 1].t;
    this.x.domain([tmin, tmax + 0.1]);
    const smax = Math.max(d3.max(this.data, d => d.score), this.q_hat ?? 0) * 1.15 || 1;
    this.y.domain([0, smax]);

    this.g.select('.x-axis').call(
      d3.axisBottom(this.x).ticks(6).tickFormat(d => formatT(d)));
    this.g.select('.y-axis').call(d3.axisLeft(this.y).ticks(4).tickFormat(d => d.toFixed(2)));

    // Umbral conformal
    const thr = this.g.select('.threshold-layer').selectAll('line.qhat')
      .data(this.q_hat !== null ? [this.q_hat] : []);
    thr.enter().append('line').attr('class', 'qhat')
      .attr('stroke', '#e63946').attr('stroke-width', 1.4).attr('stroke-dasharray', '6 3')
      .merge(thr)
      .attr('x1', 0).attr('x2', this.W)
      .attr('y1', d => this.y(d)).attr('y2', d => this.y(d));
    thr.exit().remove();

    // Zonas de alerta (columnas donde score > q_hat)
    const alerts = this.data.filter(d => d.alert)
      .map(d => ({ t0: d.t, t1: d.t + 1 }));
    const alertLayer = this.g.select('.alert-layer').selectAll('rect.alert-zone')
      .data(alerts, d => d.t0);
    alertLayer.enter().append('rect').attr('class', 'alert-zone')
      .attr('y', 0).attr('height', this.H)
      .attr('fill', '#e63946').attr('opacity', 0.16)
      .merge(alertLayer)
      .attr('x', d => this.x(d.t0))
      .attr('width', d => Math.max(1.5, this.x(d.t1) - this.x(d.t0)));
    alertLayer.exit().remove();

    // Zonas de evento real (borde superior)
    const events = this.data.filter(d => d.inEvent).map(d => d.t);
    const faultLayer = this.g.select('.fault-layer').selectAll('rect.ev-mark')
      .data(events, d => d);
    faultLayer.enter().append('rect').attr('class', 'ev-mark')
      .attr('y', 0).attr('height', 4)
      .attr('fill', '#f4d35e')
      .merge(faultLayer)
      .attr('x', d => this.x(d))
      .attr('width', d => Math.max(1.5, this.x(d + 1) - this.x(d)));
    faultLayer.exit().remove();

    const path = this.g.select('.line-layer').selectAll('path.score-line').data([this.data]);
    path.enter().append('path').attr('class', 'score-line')
      .attr('fill', 'none').attr('stroke', '#b294ff').attr('stroke-width', 1.5)
      .merge(path).attr('d', this.line);
    path.exit().remove();
  }
}

// -------------------------------------------------------------
// Barras de contribucion por feature (explicabilidad de la alerta)
// -------------------------------------------------------------
export class FeatureBars {
  constructor(container, onSelectFeature) {
    this.container = container;
    this.onSelectFeature = onSelectFeature;
    this._init();
  }

  _init() {
    const svg = d3.select(this.container)
      .append('svg')
      .attr('viewBox', `0 0 320 200`)
      .style('width', '100%')
      .style('height', 'auto');
    this.svg = svg;
    this.g = svg.append('g').attr('transform', 'translate(96, 4)');
    this.W = 320 - 96 - 30;
    this.rowH = 200 / 9;
    this.scale = d3.scaleLinear().range([0, this.W]).domain([0, 1]);
  }

  update(contrib, scoreTotal) {
    const maxV = contrib.length ? contrib[0].v : 1;
    this.scale.domain([0, maxV || 1]);
    const rows = this.g.selectAll('g.fbar').data(contrib, d => d.feat);
    const enter = rows.enter().append('g').attr('class', 'fbar')
      .attr('transform', (d, i) => `translate(0, ${i * this.rowH})`)
      .style('cursor', 'pointer')
      .on('click', (ev, d) => this.onSelectFeature && this.onSelectFeature(d.feat));
    enter.append('rect')
      .attr('class', 'fbar-bg')
      .attr('x', -92).attr('width', this.W + 92)
      .attr('y', 2).attr('height', this.rowH - 5)
      .attr('fill', 'transparent');
    enter.append('text')
      .attr('class', 'fbar-label')
      .attr('x', -6).attr('y', this.rowH / 2 + 2)
      .attr('text-anchor', 'end')
      .attr('font-size', 10)
      .attr('fill', '#aab3c0');
    enter.append('rect')
      .attr('class', 'fbar-rect')
      .attr('x', 0).attr('y', 4)
      .attr('height', this.rowH - 10)
      .attr('rx', 2);
    rows.select('.fbar-label').text(d => FEATURE_LABELS[d.feat] || d.feat);
    rows.select('.fbar-rect')
      .transition().duration(300)
      .attr('width', d => Math.max(2, this.scale(d.v)))
      .attr('fill', (d, i) => (i === 0 ? '#e63946' : i < 3 ? '#f4d35e' : '#4ea1ff'))
      .attr('opacity', 0.85);
    rows.exit().remove();
  }
}

// -------------------------------------------------------------
// Heatmap 22 x N de residuos normalizados (streaming correcto:
// cada fila es una feature, cada columna un paso de tiempo)
// -------------------------------------------------------------
export class HeatmapPlot {
  constructor(container, tooltip, engine, onSelectFeature) {
    this.container = container;
    this.tooltip = tooltip;
    this.engine = engine;
    this.onSelectFeature = onSelectFeature;
    this.nCols = 90;
    this.rows = N_FEATURES;
    // matrix[r] = serie temporal (max nCols) del residuo normalizado de la feature r
    this.matrix = Array.from({ length: this.rows }, () => []);
    this.tStamps = [];
    this._init();
  }

  _init() {
    const W = 500, H = 400;
    const M = { top: 26, right: 14, bottom: 26, left: 86 };

    const svg = d3.select(this.container)
      .append('svg')
      .attr('viewBox', `0 0 ${W} ${H}`)
      .style('width', '100%')
      .style('height', 'auto');

    this.svg = svg;
    this.cellW = (W - M.left - M.right) / this.nCols;
    this.cellH = (H - M.top - M.bottom) / this.rows;
    this.g = svg.append('g').attr('transform', `translate(${M.left},${M.top})`);

    this.color = d3.scaleSequential(d3.interpolateInferno).domain([0, 1]);

    // Etiquetas de features (clicables: seleccionan la feature en senales)
    const labels = svg.append('g').attr('transform', `translate(${M.left - 4}, ${M.top})`);
    labels.selectAll('text').data(FEATURES).enter().append('text')
      .attr('class', 'hm-label')
      .attr('x', 0).attr('y', (d, i) => i * this.cellH + this.cellH * 0.72)
      .attr('text-anchor', 'end').attr('fill', '#aab3c0').attr('font-size', 8.6)
      .style('cursor', 'pointer')
      .text(d => FEATURE_LABELS[d] || d)
      .on('click', (ev, d) => this.onSelectFeature && this.onSelectFeature(d));

    // Cabecera temporal
    svg.append('text').attr('x', M.left).attr('y', 13)
      .attr('fill', '#6b7280').attr('font-size', 9)
      .text('← hace 90 s');
    svg.append('text').attr('x', W - M.right).attr('y', 13)
      .attr('text-anchor', 'end').attr('fill', '#6b7280').attr('font-size', 9)
      .text('ahora');

    this.g.append('rect')
      .attr('class', 'hm-overlay')
      .attr('width', this.nCols * this.cellW)
      .attr('height', this.rows * this.cellH)
      .attr('fill', 'transparent');

    // Overlay para tooltip
    svg.append('rect')
      .attr('x', M.left).attr('y', M.top)
      .attr('width', this.nCols * this.cellW)
      .attr('height', this.rows * this.cellH)
      .attr('fill', 'transparent')
      .on('mousemove', ev => this._onMove(ev))
      .on('mouseleave', () => this.tooltip.hide());
  }

  _onMove(ev) {
    const [mx, my] = d3.pointer(ev, this.g.node());
    const c = Math.max(0, Math.min(this.nCols - 1, Math.floor(mx / this.cellW)));
    const r = Math.max(0, Math.min(this.rows - 1, Math.floor(my / this.cellH)));
    const v = this.matrix[r][c];
    if (v === undefined) return;
    const feat = FEATURES[r];
    const t = this.tStamps[c];
    this.tooltip.show(`
      <div class="tt-title">${FEATURE_LABELS[feat]} · ${t !== undefined ? formatT(t) : '--'}</div>
      <div class="tt-row"><span>|residuo| norm.</span><b>${v.toFixed(2)}σ</b></div>
      <div class="tt-row muted"><span>Click en la etiqueta</span><b>ver en senales</b></div>
    `, ev);
  }

  push(sample) {
    // Normalizacion identica al detector: |r| / sigma_tipico
    const NORM = [
      0.005, 0.005, 0.005, 0.005, 0.005,
      2.0, 2.0, 2.0, 2.0,
      0.05,
      0.05, 0.03,
      0.5, 0.5,
      0.01, 0.02,
      0.3, 0.4,
      0.1, 0.1,
      0.01, 0.01,
    ];
    for (let r = 0; r < this.rows; r++) {
      const v = Math.abs(sample.obs[r] - sample.exp[r]) / NORM[r];
      this.matrix[r].push(Math.min(1, v));
      if (this.matrix[r].length > this.nCols) this.matrix[r].shift();
    }
    this.tStamps.push(sample.t);
    if (this.tStamps.length > this.nCols) this.tStamps.shift();

    // Re-render completo del join (sin claves: las columnas se desplazan)
    const flat = [];
    for (let c = 0; c < this.nCols; c++) {
      for (let r = 0; r < this.rows; r++) {
        flat.push({ r, c, v: this.matrix[r][c] });
      }
    }
    const cells = this.g.selectAll('rect.cell').data(flat);
    cells.enter()
      .append('rect')
      .attr('class', 'cell')
      .attr('width', Math.max(1, this.cellW - 0.5))
      .attr('height', Math.max(1, this.cellH - 0.5))
      .merge(cells)
      .attr('x', d => d.c * this.cellW)
      .attr('y', d => d.r * this.cellH)
      .attr('fill', d => d.v === undefined ? '#161b22' : this.color(d.v));
    cells.exit().remove();
  }
}
