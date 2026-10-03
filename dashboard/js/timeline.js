// timeline.js — Timeline horizontal de eventos programados sobre las 4 h
// de la simulacion. Playhead + detecciones marcadas + tooltip.

import { CATALOGO } from './data.js';

const FAULT_ORDER = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8'];
const FAULT_SHORT = {
  F1: 'Corto L1', F2: 'Sobrecarga L2', F3: 'Sobretension AT',
  F4: 'Degrad. aisl.', F5: 'Perdida PMU', F6: 'Desbalance',
  F7: 'Falla buje', F8: 'Evento climatico',
};

export class Timeline {
  constructor(container, engine, duracionS, tooltip) {
    this.engine = engine;
    this.tooltip = tooltip;
    this.duracion = duracionS;

    const W = 1180, H = 190;
    const M = { top: 18, right: 30, bottom: 20, left: 110 };

    const svg = d3.select(container)
      .append('svg')
      .attr('viewBox', `0 0 ${W} ${H}`)
      .style('width', '100%')
      .style('height', 'auto');

    this.x = d3.scaleLinear()
      .domain([0, duracionS])
      .range([M.left, W - M.right]);

    const rowH = (H - M.top - M.bottom) / FAULT_ORDER.length;

    // Etiquetas de carril
    svg.selectAll('text.tl-label').data(FAULT_ORDER).enter().append('text')
      .attr('class', 'tl-label')
      .attr('x', M.left - 8)
      .attr('y', (d, i) => M.top + i * rowH + rowH / 2 + 3)
      .attr('text-anchor', 'end')
      .attr('fill', '#aab3c0')
      .attr('font-size', 9.5)
      .text(d => `${d} · ${FAULT_SHORT[d]}`);

    // Lineas de carril
    svg.selectAll('line.tl-lane').data(FAULT_ORDER).enter().append('line')
      .attr('class', 'tl-lane')
      .attr('x1', M.left).attr('x2', W - M.right)
      .attr('y1', (d, i) => M.top + i * rowH + rowH / 2)
      .attr('y2', (d, i) => M.top + i * rowH + rowH / 2);

    // Eje temporal (horas)
    const horas = d3.range(0, duracionS + 1, 1800);
    svg.append('g').attr('transform', `translate(0, ${H - 4})`)
      .call(d3.axisBottom(this.x)
        .tickValues(horas)
        .tickFormat(d => `${Math.floor(d / 3600)}h`)
        .tickSize(3));
    svg.selectAll('.tl-axis text').attr('fill', '#6b7280').attr('font-size', 9);

    // Capas
    this.blocksG = svg.append('g');
    this.detectG = svg.append('g');
    this.playhead = svg.append('line')
      .attr('class', 'tl-playhead')
      .attr('y1', M.top - 6)
      .attr('y2', H - M.bottom);

    this.svg = svg;
    this.rowH = rowH;
    this.M = M;

    // Interaccion: tooltip global por bloques se asigna en update
    this._lastKey = null;
  }

  update(currentT, detecciones) {
    const self = this;
    // detecciones: Map idx_ini -> t_deteccion (primer alerta del evento)
    const eventos = this.engine.eventos;

    const blocks = this.blocksG.selectAll('rect.tl-block')
      .data(eventos, d => `${d.codigo}_${d.idx_ini}_${d.manual ? 'm' : 'p'}`);
    blocks.enter()
      .append('rect')
      .attr('class', 'tl-block')
      .attr('height', this.rowH - 7)
      .attr('rx', 2)
      .on('mousemove', function (ev, d) {
        self.tooltip.show(`
          <div class="tt-title">${d.codigo} · ${d.nombre}</div>
          <div class="tt-row"><span>Inicio</span><b>${fmtT(d.t_ini)}</b></div>
          <div class="tt-row"><span>Duracion</span><b>${(d.t_fin - d.t_ini).toFixed(0)} s</b></div>
          <div class="tt-row"><span>Severidad</span><b>${(d.severidad * 100).toFixed(0)}%</b></div>
          <div class="tt-row"><span>Deteccion</span><b style="color:${d.detectado ? '#6ad36e' : '#e63946'}">${d.detectado ? 'detectado' : 'sin detectar'}</b></div>
          ${d.manual ? '<div class="tt-row"><span>Origen</span><b>manual (UI)</b></div>' : ''}
        `, ev);
      })
      .on('mouseleave', () => self.tooltip.hide())
      .merge(blocks)
      .attr('x', d => this.x(d.t_ini))
      .attr('width', d => Math.max(3, this.x(d.t_fin) - this.x(d.t_ini)))
      .attr('y', d => this.M.top + FAULT_ORDER.indexOf(d.codigo) * this.rowH + 3)
      .attr('fill', d => d.detectado ? 'rgba(106,211,110,0.55)' : 'rgba(230,57,70,0.5)')
      .attr('stroke', d => d.detectado ? '#6ad36e' : '#e63946')
      .attr('stroke-width', 0.8);
    blocks.exit().remove();

    // Marcas de deteccion (diamantes)
    const dets = detecciones || [];
    const marks = this.detectG.selectAll('path.tl-det')
      .data(dets, d => d.key);
    marks.enter()
      .append('path')
      .attr('class', 'tl-det')
      .attr('d', d3.symbol().type(d3.symbolDiamond).size(28))
      .attr('fill', '#6ad36e')
      .merge(marks)
      .attr('transform', d =>
        `translate(${this.x(d.t)}, ${this.M.top + FAULT_ORDER.indexOf(d.codigo) * this.rowH + this.rowH / 2})`);
    marks.exit().remove();

    this.playhead.attr('x1', this.x(Math.min(currentT, this.duracion)))
      .attr('x2', this.x(Math.min(currentT, this.duracion)));
  }
}

function fmtT(t) {
  const m = Math.floor(t / 60);
  const s = Math.floor(t % 60);
  return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
}
