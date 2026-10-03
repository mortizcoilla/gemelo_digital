// federated.js — Vista de aprendizaje federado entre 3 SEs.
// Mapa en estrella (FedAvg) + convergencia de loss + estado por SE.

import { ConformalDetector } from './conformal.js';

const K = 3; // numero de subestaciones (prototipo; en produccion 5+)
export const ROUND_SAMPLES = 200;

export class FederatedView {
  constructor() {
    this.rounds = 0;
    this.maxRounds = 4;
    this.lossHistory = [];
    this.centralLoss = 0.12; // referencia del modelo centralizado (notebook)
    this.clients = Array.from({ length: K }, (_, i) => ({
      id: `SE-${(i + 1).toString().padStart(2, '0')}`,
      region: ['Valparaiso', 'Biobio', 'RM'][i],
      loss: 0.4 + Math.random() * 0.2,
      samples: 0,
      alerts: 0,
      detector: new ConformalDetector(),
    }));
  }

  tickAll(sample) {
    let totalLoss = 0;
    for (const cli of this.clients) {
      const perturbed = this._perturb(sample, cli);
      const r = cli.detector.update(perturbed);
      cli.samples++;
      if (r.alert) cli.alerts++;
      // Loss tipo MSE: decrece con rondas (convergencia del modelo global)
      const loss = 0.10 + Math.abs(r.score) * 0.05;
      cli.loss = 0.7 * cli.loss + 0.3 * loss;
      totalLoss += cli.loss;
    }
    const globalLoss = totalLoss / this.clients.length;
    this.lossHistory.push(globalLoss);
    if (this.lossHistory.length > 1200) this.lossHistory.shift();

    // Avanzar ronda cada ROUND_SAMPLES muestras
    const done = this.clients[0].samples;
    if (done % ROUND_SAMPLES === 0 && done > 0 && this.rounds < this.maxRounds) {
      this.rounds++;
    }
    return { globalLoss, round: this.rounds };
  }

  _perturb(sample, client) {
    // Cada SE ve una version con sesgo leve (heterogeneidad de hardware)
    const bias = (client.region.charCodeAt(0) % 7) * 0.002 - 0.005;
    const perturbed = {
      ...sample,
      obs: sample.obs.slice(),
      exp: sample.exp.slice(),
    };
    for (let i = 0; i < perturbed.obs.length; i++) {
      perturbed.obs[i] += bias * (1 + 0.1 * i);
    }
    return perturbed;
  }

  getState() {
    return {
      rounds: this.rounds,
      maxRounds: this.maxRounds,
      globalLoss: this.lossHistory[this.lossHistory.length - 1] ?? 0,
      centralLoss: this.centralLoss,
      history: this.lossHistory.slice(-600),
      clients: this.clients.map(c => ({
        id: c.id, region: c.region, loss: c.loss,
        samples: c.samples, alerts: c.alerts,
      })),
    };
  }
}

// ---------------------------------------------------------------
// Mapa en estrella + grafico de convergencia
// ---------------------------------------------------------------
export function renderFederatedMap(container, fed, tooltip) {
  const wrap = d3.select(container).append('div').attr('class', 'fed-wrap');

  // --- Mapa ---
  const mapDiv = wrap.append('div').attr('class', 'fed-map');
  const W = 420, H = 230;
  const svg = mapDiv.append('svg')
    .attr('viewBox', `0 0 ${W} ${H}`)
    .style('width', '100%')
    .style('height', 'auto');

  const cx = W / 2, cy = H / 2;
  const radius = 88;

  const angles = fed.clients.map((_, i) => (i / fed.clients.length) * Math.PI * 2 - Math.PI / 2);
  const nodes = fed.clients.map((c, i) => ({
    ...c,
    x: cx + Math.cos(angles[i]) * radius,
    y: cy + Math.sin(angles[i]) * radius,
  }));

  // Edges: animacion CSS continua (flujo de pesos, no de datos crudos)
  const edges = svg.append('g').selectAll('line.edge')
    .data(nodes)
    .enter().append('line')
    .attr('class', 'edge fed-pulse')
    .attr('x1', cx).attr('y1', cy)
    .attr('x2', d => d.x).attr('y2', d => d.y);

  // Etiqueta "solo pesos" abajo a la izquierda (libre de nodos y edges)
  svg.append('text')
    .attr('x', 10).attr('y', H - 6)
    .attr('fill', '#6b7280').attr('font-size', 9)
    .text('se comparten pesos, no datos crudos');

  // Nodo central (orquestador FedAvg)
  svg.append('circle')
    .attr('cx', cx).attr('cy', cy).attr('r', 26)
    .attr('fill', '#161b22').attr('stroke', '#f4d35e').attr('stroke-width', 1.8);
  svg.append('text').attr('x', cx).attr('y', cy + 4)
    .attr('text-anchor', 'middle')
    .attr('fill', '#f4d35e')
    .attr('font-size', 10).attr('font-weight', 700)
    .text('FedAvg');

  // Nodos de SE
  const groups = svg.append('g').selectAll('g.se')
    .data(nodes)
    .enter().append('g')
    .attr('class', 'se')
    .attr('transform', d => `translate(${d.x}, ${d.y})`)
    .style('cursor', 'default');

  const nodeCircles = groups.append('circle')
    .attr('r', 17)
    .attr('fill-opacity', 0.15)
    .attr('stroke-width', 1.5);
  const nodeIds = groups.append('text')
    .attr('text-anchor', 'middle')
    .attr('y', 3)
    .attr('fill', '#e6edf3')
    .attr('font-size', 9.5)
    .attr('font-weight', 600)
    .text(d => d.id.replace('SE-', 'SE'));
  const nodeAlerts = groups.append('text')
    .attr('class', 'node-alerts')
    .attr('x', 14).attr('y', -10)
    .attr('text-anchor', 'middle')
    .attr('font-size', 8.5)
    .attr('font-weight', 700)
    .text('');
  groups.append('text')
    .attr('text-anchor', 'middle')
    .attr('y', 30)
    .attr('fill', '#888')
    .attr('font-size', 8.5)
    .text(d => d.region);

  groups
    .on('mousemove', (ev, d) => {
      const st = fed.getState();
      const cli = st.clients.find(c => c.id === d.id);
      tooltip.show(`
        <div class="tt-title">${cli.id} · ${cli.region}</div>
        <div class="tt-row"><span>Loss local</span><b>${cli.loss.toFixed(3)}</b></div>
        <div class="tt-row"><span>Muestras</span><b>${cli.samples}</b></div>
        <div class="tt-row"><span>Alertas</span><b style="color:${cli.alerts > 0 ? '#e63946' : '#6ad36e'}">${cli.alerts}</b></div>
      `, ev);
    })
    .on('mouseleave', () => tooltip.hide());

  function restyleNodes(state) {
    nodeCircles
      .attr('fill', d => {
        const cli = state.clients.find(c => c.id === d.id);
        return cli && cli.alerts > 0 ? '#e63946' : '#2ca02c';
      })
      .attr('stroke', d => {
        const cli = state.clients.find(c => c.id === d.id);
        return cli && cli.alerts > 0 ? '#e63946' : '#2ca02c';
      });
    nodeAlerts
      .attr('fill', d => {
        const cli = state.clients.find(c => c.id === d.id);
        return cli && cli.alerts > 0 ? '#e63946' : 'transparent';
      })
      .text(d => {
        const cli = state.clients.find(c => c.id === d.id);
        return cli && cli.alerts > 0 ? String(cli.alerts) : '';
      });
  }
  restyleNodes(fed.getState());

  // --- Convergencia ---
  const convDiv = wrap.append('div').attr('class', 'fed-conv');
  const CW = 420, CH = 96;
  const csvg = convDiv.append('svg')
    .attr('viewBox', `0 0 ${CW} ${CH}`)
    .style('width', '100%')
    .style('height', 'auto');
  const cg = csvg.append('g').attr('transform', 'translate(34, 8)');
  const cW = CW - 44, cH = CH - 24;

  csvg.append('text').attr('x', 4).attr('y', 8)
    .attr('fill', '#aab3c0').attr('font-size', 9.5)
    .text('Loss global (FedAvg) vs centralizado');

  const xScale = d3.scaleLinear().range([0, cW]);
  const yScale = d3.scaleLinear().range([cH, 4]);
  const line = d3.line()
    .x((d, i) => xScale(i))
    .y(d => yScale(d))
    .curve(d3.curveMonotoneX);
  const path = cg.append('path')
    .attr('fill', 'none').attr('stroke', '#4ea1ff').attr('stroke-width', 1.4);
  const refLine = cg.append('line')
    .attr('stroke', '#e63946').attr('stroke-dasharray', '5 3').attr('stroke-width', 1.2);
  const refText = csvg.append('text')
    .attr('fill', '#e63946').attr('font-size', 8.5)
    .text('centralizado');
  const axisG = cg.append('g').attr('transform', `translate(0, ${cH})`);

  function update() {
    const state = fed.getState();
    restyleNodes(state);

    const hist = state.history;
    if (hist.length > 2) {
      xScale.domain([0, hist.length - 1]);
      const lo = Math.min(d3.min(hist), state.centralLoss);
      const hi = Math.max(d3.max(hist), state.centralLoss);
      yScale.domain([lo * 0.97, hi * 1.03]);
      path.attr('d', line(hist));
      refLine
        .attr('x1', 0).attr('x2', cW)
        .attr('y1', yScale(state.centralLoss))
        .attr('y2', yScale(state.centralLoss));
      refText.attr('x', cW - 66).attr('y', yScale(state.centralLoss) + 12);
      axisG.call(d3.axisBottom(xScale).ticks(4).tickFormat((d, i) => {
        const total = hist.length;
        return `${Math.round(total - d)}`;
      }).tickSize(2));
    }
  }

  return { update };
}
