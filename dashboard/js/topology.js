// topology.js — Diagrama unifilar interactivo con D3.js.
// 110 kV -> 2 trafos 30 MVA -> barra 23 kV -> 4 alimentadores.
// Hover = tooltip, click en alimentador = selecciona su corriente
// en el panel de senales.

const W = 480;
const H = 440;

const FAULT_TARGETS = {
  F1: { lineas: [0] },
  F2: { lineas: [1] },
  F3: { trafos: [0, 1], barra: 'at' },
  F4: { trafos: [0], barra: 'bt' },
  F5: { pmu: true },
  F6: { lineas: [0, 2] },
  F7: { trafos: [0] },
  F8: { trafos: [0, 1], lineas: [0, 1, 2, 3], barra: 'at' },
};

export function initTopology(container, engine, tooltip, onSelectFeature) {
  const svg = d3.select(container)
    .append('svg')
    .attr('viewBox', `0 0 ${W} ${H}`)
    .attr('preserveAspectRatio', 'xMidYMid meet')
    .style('width', '100%')
    .style('height', 'auto');

  const g = svg.append('g');

  // ---------------------------------------------------------------
  // Llegadas 110 kV
  // ---------------------------------------------------------------
  const inY = 34;
  const inX = [W * 0.35, W * 0.65];
  const llegadas = g.selectAll('.llegada')
    .data(inX).enter().append('g').attr('class', 'llegada');
  llegadas.append('line')
    .attr('class', 'topo-flow')
    .attr('x1', d => d).attr('y1', 6)
    .attr('x2', d => d).attr('y2', inY);
  llegadas.append('text')
    .attr('x', d => d).attr('y', 2)
    .attr('text-anchor', 'middle')
    .attr('class', 'topo-label')
    .attr('dy', '-2')
    .text((d, i) => `Llegada ${i + 1} · 110 kV`);

  // Disyuntores AT
  llegadas.append('rect')
    .attr('class', 'breaker')
    .attr('x', d => d - 5).attr('y', inY + 14)
    .attr('width', 10).attr('height', 14)
    .attr('fill', 'none');

  // ---------------------------------------------------------------
  // Barra AT
  // ---------------------------------------------------------------
  const barraAtY = 78;
  g.append('line')
    .attr('x1', 40).attr('y1', barraAtY)
    .attr('x2', W - 40).attr('y2', barraAtY)
    .attr('class', 'bus-at');
  g.append('text')
    .attr('x', 40).attr('y', barraAtY - 6)
    .attr('class', 'topo-label')
    .text('BARRA 110 kV');
  const vAtLabel = g.append('text')
    .attr('x', W - 40).attr('y', barraAtY - 6)
    .attr('class', 'value-label')
    .attr('text-anchor', 'end')
    .text('1.000 pu');

  // ---------------------------------------------------------------
  // Trafos T1 / T2
  // ---------------------------------------------------------------
  const trafoY = 158;
  const trafoCx = [W * 0.35, W * 0.65];

  // Conexion barra -> trafo (con flujo animado)
  const connT = g.selectAll('.conn-trafo')
    .data(trafoCx).enter().append('line')
    .attr('class', 'topo-flow')
    .attr('x1', d => d).attr('y1', barraAtY)
    .attr('x2', d => d).attr('y2', trafoY - 32);

  const trafos = g.selectAll('.trafo-group')
    .data(trafoCx)
    .enter()
    .append('g')
    .attr('class', 'trafo-group')
    .attr('transform', d => `translate(${d}, ${trafoY})`);

  // Simbolo clasico: dos circulos solapados
  trafos.append('circle').attr('class', 'trafo-body c1').attr('r', 22).attr('cy', -10);
  trafos.append('circle').attr('class', 'trafo-body c2').attr('r', 22).attr('cy', 10);
  trafos.append('text')
    .attr('class', 'trafo-id')
    .attr('text-anchor', 'middle')
    .attr('y', 4)
    .text((d, i) => `T${i + 1}`);
  const trafoMva = trafos.append('text')
    .attr('class', 'value-label')
    .attr('text-anchor', 'middle')
    .attr('y', -34)
    .text('-- MVA');
  const trafoTap = trafos.append('text')
    .attr('class', 'topo-label')
    .attr('text-anchor', 'middle')
    .attr('y', 44)
    .text('tap 0');
  const trafoTemp = trafos.append('text')
    .attr('class', 'topo-label')
    .attr('text-anchor', 'middle')
    .attr('y', 56)
    .attr('fill', '#6b7280')
    .text('-- °C');

  // ---------------------------------------------------------------
  // Barra BT
  // ---------------------------------------------------------------
  const barraBtY = 244;
  g.append('line')
    .attr('x1', 30).attr('y1', barraBtY)
    .attr('x2', W - 30).attr('y2', barraBtY)
    .attr('class', 'bus-bt');
  g.append('text')
    .attr('x', 30).attr('y', barraBtY - 6)
    .attr('class', 'topo-label')
    .text('BARRA 23 kV');

  // Conexion trafo -> BT
  g.selectAll('.conn-bt')
    .data(trafoCx).enter().append('line')
    .attr('class', 'topo-flow')
    .attr('x1', d => d).attr('y1', trafoY + 32)
    .attr('x2', d => d).attr('y2', barraBtY);

  // ---------------------------------------------------------------
  // Alimentadores L1..L4
  // ---------------------------------------------------------------
  const lineasY = 360;
  const lineaX = [80, 190, 290, 400];

  const lineas = g.selectAll('.linea-group')
    .data(lineaX)
    .enter()
    .append('g')
    .attr('class', 'linea-group')
    .attr('transform', d => `translate(${d}, 0)`)
    .style('cursor', 'pointer');

  lineas.append('line')
    .attr('class', 'topo-flow linea-line')
    .attr('y1', barraBtY)
    .attr('y2', lineasY - 28);

  // Disyuntor por alimentador
  lineas.append('rect')
    .attr('class', 'breaker')
    .attr('x', -5).attr('y', barraBtY + 26)
    .attr('width', 10).attr('height', 14)
    .attr('fill', 'none');

  // Carga (flecha hacia abajo)
  lineas.append('path')
    .attr('d', 'M -7 0 L 7 0 L 0 12 Z')
    .attr('class', 'load-arrow')
    .attr('transform', `translate(0, ${lineasY - 14})`);

  const linI = lineas.append('text')
    .attr('class', 'value-label lin-i')
    .attr('text-anchor', 'middle')
    .attr('y', lineasY - 34)
    .text('-- A');

  lineas.append('text')
    .attr('class', 'topo-label')
    .attr('text-anchor', 'middle')
    .attr('y', lineasY + 24)
    .text((d, i) => `L${i + 1} · 52-${i + 1}`);

  // Interacciones
  lineas
    .on('click', (ev, d) => {
      const idx = lineaX.indexOf(d);
      if (onSelectFeature) onSelectFeature(`I_L${idx + 1}`);
    })
    .on('mousemove', (ev, d) => {
      const idx = lineaX.indexOf(d);
      const I = engine.lastSample ? engine.lastSample.obs[5 + idx] : 0;
      const pct = 100 * I / 800;
      tooltip.show(`
        <div class="tt-title">Alimentador L${idx + 1}</div>
        <div class="tt-row"><span>Corriente</span><b>${I.toFixed(0)} A</b></div>
        <div class="tt-row"><span>Carga</span><b>${(pct).toFixed(0)}% de 800 A</b></div>
        <div class="tt-row"><span>Click</span><b>ver I_L${idx + 1} en senales</b></div>
      `, ev);
    })
    .on('mouseleave', () => tooltip.hide());

  trafos
    .on('mousemove', (ev, d) => {
      const idx = trafoCx.indexOf(d);
      const S = engine.lastSample ? engine.lastSample.obs[9] / 2 : 0;
      const T = engine.lastSample ? engine.lastSample.obs[12 + Math.min(idx, 1)] : 0;
      tooltip.show(`
        <div class="tt-title">Transformador T${idx + 1} · 30 MVA</div>
        <div class="tt-row"><span>Carga aprox.</span><b>${S.toFixed(1)} MVA (${(S / 30 * 100).toFixed(0)}%)</b></div>
        <div class="tt-row"><span>T devanado</span><b>${T.toFixed(1)} °C</b></div>
        <div class="tt-row"><span>Limite</span><b>95 °C</b></div>
      `, ev);
    })
    .on('mouseleave', () => tooltip.hide());

  // ---------------------------------------------------------------
  // Badge PMU (para F5)
  // ---------------------------------------------------------------
  const pmuBadge = g.append('g')
    .attr('class', 'pmu-badge')
    .attr('transform', `translate(${W - 78}, ${barraAtY + 8})`)
    .style('visibility', 'hidden');
  pmuBadge.append('rect')
    .attr('width', 74).attr('height', 20).attr('rx', 4)
    .attr('fill', 'rgba(230,57,70,0.15)')
    .attr('stroke', '#e63946');
  pmuBadge.append('text')
    .attr('x', 37).attr('y', 14)
    .attr('text-anchor', 'middle')
    .attr('fill', '#e63946').attr('font-size', 10).attr('font-weight', 700)
    .text('PMU SIN DATOS');

  return {
    svg,
    update(sample) {
      const obs = sample.obs;
      engine.lastSample = sample;

      vAtLabel.text(`${obs[0].toFixed(3)} pu`)
        .attr('fill', Math.abs(obs[0] - 1) > 0.07 ? '#e63946' : '#e6edf3');

      // Carga por trafo (S_total / 2), temperatura y tap
      trafoMva.text((d, i) => `${(obs[9] / 2).toFixed(1)} MVA`);
      trafoTemp.text((d, i) => `${obs[12 + Math.min(i, 1)].toFixed(1)} °C`)
        .attr('fill', obs[12] > 95 ? '#e63946' : '#6b7280');
      trafoTap.text((d, i) => `tap ${((1 + Math.sin(sample.t / 600 + i) * 3)).toFixed(0)}`);

      // Corrientes y color por carga
      linI.text((d, i) => `${obs[5 + i].toFixed(0)} A`)
        .attr('fill', (d, i) => {
          const pct = obs[5 + i] / 800;
          if (pct > 1.0) return '#e63946';
          if (pct > 0.8) return '#f4d35e';
          return '#e6edf3';
        });
      lineas.select('.linea-line').attr('stroke-width', (d, i) =>
        2 + Math.min(2.5, obs[5 + i] / 320));

      // Marcado de fallas
      const targets = sample.event ? (FAULT_TARGETS[sample.event.codigo] || {}) : {};
      trafos.select('.trafo-body')
        .classed('fault', false);
      lineas.select('.linea-line').classed('fault', false);
      lineas.select('.load-arrow').classed('fault', false);
      g.select('.bus-at').classed('fault', false);
      g.select('.bus-bt').classed('fault', false);
      pmuBadge.style('visibility', 'hidden');

      if (sample.event) {
        (targets.trafos || []).forEach(i => {
          trafos.filter((d, j) => j === i).selectAll('.trafo-body').classed('fault', true);
        });
        (targets.lineas || []).forEach(i => {
          lineas.filter((d, j) => j === i).select('.linea-line').classed('fault', true);
          lineas.filter((d, j) => j === i).select('.load-arrow').classed('fault', true);
        });
        if (targets.barra === 'at') g.select('.bus-at').classed('fault', true);
        if (targets.barra === 'bt') g.select('.bus-bt').classed('fault', true);
        if (targets.pmu) pmuBadge.style('visibility', 'visible');
      }
    },
  };
}
