// signals.js — Series de tiempo SCADA/PMU con D3 streaming.
// Muestra observado (solido) + esperado por el gemelo (punteado),
// bandas de falla del motor y crosshair con tooltip.

import { FEATURES, FEATURE_LABELS } from './data.js';

const MARGIN = { top: 18, right: 64, bottom: 26, left: 56 };
const W = 860;
const H = 250;

const FEATURE_META = {
  V_at:   { unit: 'pu',  guides: [0.93, 1.07], fmt: v => v.toFixed(3) },
  V_bt1:  { unit: 'pu',  guides: [0.93, 1.07], fmt: v => v.toFixed(3) },
  V_bt2:  { unit: 'pu',  guides: [0.93, 1.07], fmt: v => v.toFixed(3) },
  V_bt3:  { unit: 'pu',  guides: [0.93, 1.07], fmt: v => v.toFixed(3) },
  V_bt4:  { unit: 'pu',  guides: [0.93, 1.07], fmt: v => v.toFixed(3) },
  I_L1:   { unit: 'A',   guides: [800], fmt: v => v.toFixed(0) },
  I_L2:   { unit: 'A',   guides: [800], fmt: v => v.toFixed(0) },
  I_L3:   { unit: 'A',   guides: [800], fmt: v => v.toFixed(0) },
  I_L4:   { unit: 'A',   guides: [800], fmt: v => v.toFixed(0) },
  S_total:{ unit: 'MVA', guides: [60], fmt: v => v.toFixed(1) },
  P_total:{ unit: 'MW',  guides: [], fmt: v => v.toFixed(1) },
  Q_total:{ unit: 'MVAR',guides: [], fmt: v => v.toFixed(1) },
  T_dev_T1: { unit: '°C', guides: [95], fmt: v => v.toFixed(1) },
  T_dev_T2: { unit: '°C', guides: [95], fmt: v => v.toFixed(1) },
  f:      { unit: 'Hz',  guides: [49.5, 50.5], fmt: v => v.toFixed(2) },
  rocof:  { unit: 'Hz/s',guides: [], fmt: v => v.toFixed(3) },
  thd_V:  { unit: '%',   guides: [5], fmt: v => v.toFixed(1) },
  thd_I:  { unit: '%',   guides: [8], fmt: v => v.toFixed(1) },
  ang_V:  { unit: 'rad', guides: [], fmt: v => v.toFixed(2) },
  ang_I:  { unit: 'rad', guides: [], fmt: v => v.toFixed(2) },
  cos_phi:{ unit: 'pu',  guides: [], fmt: v => v.toFixed(3) },
  desbalance: { unit: 'pu', guides: [], fmt: v => v.toFixed(3) },
};

function featureIndex(name) {
  return FEATURES.indexOf(name);
}

function formatT(t) {
  const m = Math.floor(t / 60);
  const s = Math.floor(t % 60);
  return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
}

export class SignalsPlot {
  constructor(container, engine, feature = 'V_at', tooltip) {
    this.container = container;
    this.engine = engine;
    this.feature = feature;
    this.tooltip = tooltip;
    this.maxPoints = 180;
    this.data = [];   // {t, obs, exp, ev}
    this.onHover = null;   // callback(t | null) para sincronizar cursor
    this._init();
  }

  _init() {
    const svg = d3.select(this.container)
      .append('svg')
      .attr('viewBox', `0 0 ${W} ${H}`)
      .style('width', '100%')
      .style('height', 'auto');

    this.svg = svg;
    this.g = svg.append('g')
      .attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    this.width = W - MARGIN.left - MARGIN.right;
    this.height = H - MARGIN.top - MARGIN.bottom;

    this.g.append('g').attr('class', 'guide-layer');
    this.g.append('g').attr('class', 'fault-layer');
    this.g.append('g').attr('class', 'exp-layer');
    this.g.append('g').attr('class', 'line-layer');
    this.g.append('g').attr('class', 'cursor-layer');
    this.g.append('g').attr('class', 'x-axis').attr('transform', `translate(0,${this.height})`);
    this.g.append('g').attr('class', 'y-axis');

    this.xScale = d3.scaleLinear().range([0, this.width]);
    this.yScale = d3.scaleLinear().range([this.height, 0]);

    this.lineObs = d3.line()
      .x(d => this.xScale(d.t))
      .y(d => this.yScale(d.obs))
      .curve(d3.curveMonotoneX);
    this.lineExp = d3.line()
      .x(d => this.xScale(d.t))
      .y(d => this.yScale(d.exp))
      .curve(d3.curveMonotoneX);

    // Leyenda
    const legend = svg.append('g').attr('transform', `translate(${MARGIN.left + 8}, 4)`);
    legend.append('line').attr('x1', 0).attr('x2', 16).attr('y1', 8).attr('y2', 8)
      .attr('stroke', '#4ea1ff').attr('stroke-width', 1.6);
    legend.append('text').attr('x', 20).attr('y', 11).attr('fill', '#aab3c0')
      .attr('font-size', 10).text('Observado');
    legend.append('line').attr('x1', 96).attr('x2', 112).attr('y1', 8).attr('y2', 8)
      .attr('stroke', '#f4d35e').attr('stroke-width', 1.4).attr('stroke-dasharray', '4 2');
    legend.append('text').attr('x', 116).attr('y', 11).attr('fill', '#aab3c0')
      .attr('font-size', 10).text('Esperado (gemelo)');

    // Crosshair
    const cursor = this.g.select('.cursor-layer');
    this.vLine = cursor.append('line')
      .attr('class', 'crosshair')
      .attr('y1', 0).attr('y2', this.height)
      .style('visibility', 'hidden');
    this.cursorDotO = cursor.append('circle').attr('r', 3.5)
      .attr('fill', '#4ea1ff').style('visibility', 'hidden');
    this.cursorDotE = cursor.append('circle').attr('r', 3)
      .attr('fill', '#f4d35e').style('visibility', 'hidden');

    svg.append('rect')
      .attr('x', MARGIN.left).attr('y', MARGIN.top)
      .attr('width', this.width).attr('height', this.height)
      .attr('fill', 'transparent')
      .on('mousemove', ev => this._onMove(ev))
      .on('mouseleave', () => this._onLeave());
  }

  setFeature(feat) {
    this.feature = feat;
    this.data = [];
    this._render();
    this._onLeave();
  }

  push(sample) {
    const featIdx = featureIndex(this.feature);
    this.data.push({
      t: sample.t,
      obs: sample.obs[featIdx],
      exp: sample.exp[featIdx],
      ev: sample.event ? sample.event.codigo : null,
    });
    if (this.data.length > this.maxPoints) this.data.shift();
    this._render();
  }

  _onMove(ev) {
    if (this.data.length < 2) return;
    const [mx] = d3.pointer(ev, this.g.node());
    const t = this.xScale.invert(clamp(mx, 0, this.width));
    // muestra mas cercana
    let best = this.data[0];
    for (const d of this.data) {
      if (Math.abs(d.t - t) < Math.abs(best.t - t)) best = d;
    }
    const x = this.xScale(best.t);
    this.vLine.attr('x1', x).attr('x2', x).style('visibility', 'visible');
    this.cursorDotO.attr('cx', x).attr('cy', this.yScale(best.obs)).style('visibility', 'visible');
    this.cursorDotE.attr('cx', x).attr('cy', this.yScale(best.exp)).style('visibility', 'visible');

    const meta = FEATURE_META[this.feature] || { unit: '', fmt: v => v.toFixed(2) };
    const resid = best.obs - best.exp;
    this.tooltip.show(`
      <div class="tt-title">${FEATURE_LABELS[this.feature]} · ${formatT(best.t)}</div>
      <div class="tt-row"><span>Observado</span><b>${meta.fmt(best.obs)} ${meta.unit}</b></div>
      <div class="tt-row"><span>Esperado</span><b>${meta.fmt(best.exp)} ${meta.unit}</b></div>
      <div class="tt-row"><span>Residuo</span><b style="color:${Math.abs(resid) > 3 * (meta.sig || 1) ? '#e63946' : '#6ad36e'}">${meta.fmt(resid)} ${meta.unit}</b></div>
      ${best.ev ? `<div class="tt-row"><span>Evento</span><b style="color:#e63946">${best.ev}</b></div>` : ''}
    `, ev);
    if (this.onHover) this.onHover(best.t);
  }

  _onLeave() {
    this.vLine.style('visibility', 'hidden');
    this.cursorDotO.style('visibility', 'hidden');
    this.cursorDotE.style('visibility', 'hidden');
    this.tooltip.hide();
    if (this.onHover) this.onHover(null);
  }

  // Cursor externo (sincronizado desde otros paneles)
  showCursorAt(t) {
    if (t === null || this.data.length < 2) {
      this.vLine.style('visibility', 'hidden');
      this.cursorDotO.style('visibility', 'hidden');
      this.cursorDotE.style('visibility', 'hidden');
      return;
    }
    let best = this.data[0];
    for (const d of this.data) {
      if (Math.abs(d.t - t) < Math.abs(best.t - t)) best = d;
    }
    const x = this.xScale(best.t);
    this.vLine.attr('x1', x).attr('x2', x).style('visibility', 'visible');
    this.cursorDotO.attr('cx', x).attr('cy', this.yScale(best.obs)).style('visibility', 'visible');
    this.cursorDotE.attr('cx', x).attr('cy', this.yScale(best.exp)).style('visibility', 'visible');
  }

  _render() {
    if (this.data.length < 2) return;

    const tmin = this.data[0].t;
    const tmax = this.data[this.data.length - 1].t;
    this.xScale.domain([tmin, tmax + 0.1]);

    const allV = this.data.flatMap(d => [d.obs, d.exp]);
    let vmin = d3.min(allV), vmax = d3.max(allV);
    const pad = (vmax - vmin) * 0.18 || 0.5;
    this.yScale.domain([vmin - pad, vmax + pad]);

    this.g.select('.x-axis').call(
      d3.axisBottom(this.xScale).ticks(6).tickFormat(d => formatT(d)));
    const meta = FEATURE_META[this.feature] || { unit: '', fmt: v => v.toFixed(2) };
    this.g.select('.y-axis').call(
      d3.axisLeft(this.yScale).ticks(5).tickFormat(meta.fmt));

    // Lineas guia (limites operativos)
    const guides = this.g.select('.guide-layer').selectAll('line.guide')
      .data((meta.guides || []).filter(gv => gv > vmin - pad && gv < vmax + pad));
    guides.enter()
      .append('line').attr('class', 'guide')
      .attr('stroke-dasharray', '2 4')
      .attr('stroke', '#e07a5f')
      .merge(guides)
      .attr('x1', 0).attr('x2', this.width)
      .attr('y1', d => this.yScale(d)).attr('y2', d => this.yScale(d));
    guides.exit().remove();

    // Bandas de falla: eventos del motor que cruzan la ventana visible
    const i0 = Math.max(0, Math.floor(tmin * this.engine.fs));
    const i1 = Math.floor((tmax + 2) * this.engine.fs);
    const visibles = this.engine.eventosEnRango(i0, i1);
    const faultLayer = this.g.select('.fault-layer').selectAll('rect.fault-zone')
      .data(visibles, d => `${d.codigo}_${d.idx_ini}`);
    faultLayer.enter()
      .append('rect')
      .attr('class', 'fault-zone')
      .attr('y', 0)
      .attr('height', this.height)
      .attr('fill', '#e63946')
      .attr('opacity', 0.10)
      .merge(faultLayer)
      .attr('x', d => this.xScale(d.t_ini))
      .attr('width', d => Math.max(2, this.xScale(d.t_fin) - this.xScale(d.t_ini)));
    faultLayer.exit().remove();

    const p1 = this.g.select('.exp-layer').selectAll('path').data([this.data]);
    p1.enter().append('path')
      .attr('fill', 'none').attr('stroke', '#f4d35e')
      .attr('stroke-width', 1.3).attr('stroke-dasharray', '4 2')
      .merge(p1).attr('d', this.lineExp);
    p1.exit().remove();

    const p2 = this.g.select('.line-layer').selectAll('path').data([this.data]);
    p2.enter().append('path')
      .attr('fill', 'none').attr('stroke', '#4ea1ff').attr('stroke-width', 1.5)
      .merge(p2).attr('d', this.lineObs);
    p2.exit().remove();
  }
}

function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }

export { FEATURE_META };
