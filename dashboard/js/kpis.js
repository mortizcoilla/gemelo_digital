// kpis.js — Strip de indicadores clave con sparkline en vivo.

function makeKpi(def) {
  return {
    id: def.id,
    label: def.label,
    unit: def.unit,
    idx: def.idx,
    fmt: def.fmt,
    limits: def.limits,       // {warn, alarm} o null
    history: [],
    maxHist: 120,
  };
}

const KPI_DEFS = [
  { id: 'v_at', label: 'V AT', unit: 'pu', idx: 0, fmt: v => v.toFixed(3), limits: { loAlarm: 0.93, hiAlarm: 1.07 } },
  { id: 'v_bt', label: 'V BT', unit: 'pu', idx: 1, fmt: v => v.toFixed(3), limits: { loAlarm: 0.93, hiAlarm: 1.07 } },
  { id: 'i_tot', label: 'I total', unit: 'A', idx: [5, 6, 7, 8], fmt: v => v.toFixed(0), limits: { hiWarn: 2600, hiAlarm: 3200 } },
  { id: 'p', label: 'P', unit: 'MW', idx: 10, fmt: v => v.toFixed(1), limits: { hiWarn: 34, hiAlarm: 40 } },
  { id: 'q', label: 'Q', unit: 'MVAr', idx: 11, fmt: v => v.toFixed(1), limits: null },
  { id: 'f', label: 'f', unit: 'Hz', idx: 14, fmt: v => v.toFixed(3), limits: { loAlarm: 49.5, hiAlarm: 50.5 } },
  { id: 't_dev', label: 'T dev', unit: '°C', idx: 12, fmt: v => v.toFixed(1), limits: { hiWarn: 80, hiAlarm: 95 } },
  { id: 'thd_v', label: 'THD V', unit: '%', idx: 16, fmt: v => v.toFixed(1), limits: { hiWarn: 5, hiAlarm: 8 } },
];

export class KpiStrip {
  constructor(container, tooltip) {
    this.tooltip = tooltip;
    this.kpis = KPI_DEFS.map(makeKpi);

    const strip = d3.select(container).append('div').attr('class', 'kpi-strip');

    const cards = strip.selectAll('div.kpi')
      .data(this.kpis)
      .enter()
      .append('div')
      .attr('class', 'kpi')
      .attr('id', d => `kpi_${d.id}`);

    cards.append('div').attr('class', 'kpi-label').text(d => d.label);
    const valRow = cards.append('div').attr('class', 'kpi-val-row');
    valRow.append('span').attr('class', 'kpi-value').text('--');
    valRow.append('span').attr('class', 'kpi-unit').text(d => d.unit);
    const spark = cards.append('svg')
      .attr('class', 'kpi-spark')
      .attr('viewBox', '0 0 100 26')
      .attr('preserveAspectRatio', 'none');
    spark.append('path').attr('class', 'kpi-spark-line');
    cards.append('div').attr('class', 'kpi-badge').text('');

    cards
      .on('mousemove', (ev, d) => {
        const cur = d.history[d.history.length - 1];
        const min = Math.min(...d.history), max = Math.max(...d.history);
        this.tooltip.show(`
          <div class="tt-title">${d.label} [${d.unit}]</div>
          <div class="tt-row"><span>Actual</span><b>${cur !== undefined ? d.fmt(cur) : '--'}</b></div>
          <div class="tt-row"><span>Min / Max (2 min)</span><b>${d.fmt(min)} / ${d.fmt(max)}</b></div>
        `, ev);
      })
      .on('mouseleave', () => this.tooltip.hide());

    this.cards = cards;
  }

  push(sample) {
    const W = 100, H = 26;
    this.cards.each((kpi, i, nodes) => {
      let v;
      if (Array.isArray(kpi.idx)) {
        v = kpi.idx.reduce((acc, ix) => acc + sample.obs[ix], 0);
      } else {
        v = sample.obs[kpi.idx];
      }
      kpi.history.push(v);
      if (kpi.history.length > kpi.maxHist) kpi.history.shift();

      const node = d3.select(nodes[i]);
      const badge = kpi.limits ? this._estado(kpi, v) : 'ok';
      node.select('.kpi-value')
        .text(kpi.fmt(v))
        .attr('class', `kpi-value ${badge === 'ok' ? '' : 'kpi-' + badge}`);

      const b = node.select('.kpi-badge');
      if (kpi.limits) {
        b.attr('class', `kpi-badge ${badge === 'ok' ? 'kpi-badge-hidden' : 'kpi-badge-' + badge}`)
          .text(badge === 'warn' ? 'ALTA' : badge === 'alarm' ? 'ALERTA' : '');
      }

      // Sparkline
      const data = kpi.history;
      if (data.length > 2) {
        const lo = Math.min(...data), hi = Math.max(...data);
        const pad = (hi - lo) * 0.15 || 0.5;
        const x = d3.scaleLinear().domain([0, data.length - 1]).range([1, W - 1]);
        const y = d3.scaleLinear().domain([lo - pad, hi + pad]).range([H - 2, 2]);
        const line = d3.line()
          .x((d, j) => x(j))
          .y(d => y(d))
          .curve(d3.curveMonotoneX);
        node.select('.kpi-spark-line')
          .attr('d', line(data))
          .attr('stroke', badge === 'alarm' ? '#e63946' : badge === 'warn' ? '#f4d35e' : '#4ea1ff');
      }
    });
  }

  _estado(kpi, v) {
    const L = kpi.limits;
    if (L.hiAlarm !== undefined && v >= L.hiAlarm) return 'alarm';
    if (L.loAlarm !== undefined && v <= L.loAlarm) return 'alarm';
    if (L.hiWarn !== undefined && v >= L.hiWarn) return 'warn';
    return 'ok';
  }
}
