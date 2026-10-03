// metrics.js — Tabla de metricas: referencia del informe (bloque test del
// notebook) + detectores en vivo sobre el stream del dashboard.

// Valores de referencia del notebook ejecutado (bloque de test,
// gemelo_digital_se_ejecutado.ipynb, seed 42).
const REF = {
  static:    { fpr: 0.1282, tpr: 0.3676, f1: 0.4209, auc: 0.639, name: 'Estatico 3-sigma' },
  central:   { fpr: 0.0491, tpr: 0.7634, f1: 0.8000, auc: 0.878, name: 'AE-LSTM centralizado' },
  federated: { fpr: 0.0498, tpr: 0.8054, f1: 0.8669, auc: 0.906, name: 'AE-LSTM federado (3 SEs)' },
};

const ROWS = [
  { key: 'live_resid', name: 'Conformal residuos (este stream)',
    priv: 'si', badge: 'activo', live: true },
  { key: 'static', name: 'Estatico 3-sigma',
    priv: 'no', badge: 'legacy', ref: true },
  { key: 'central', name: 'AE-LSTM centralizado',
    priv: 'no (datos centralizados)', badge: 'referencia', ref: true },
  { key: 'federated', name: 'AE-LSTM federado (3 SEs)',
    priv: 'si (datos en origen)', badge: 'recomendado', ref: true },
];

export function initMetrics(container, tooltip) {
  const tb = d3.select(container);

  const rows = tb.selectAll('tr.met-row').data(ROWS, d => d.key)
    .enter().append('tr').attr('class', 'met-row');

  rows.append('td').text(d => d.name);
  const fmtPct = v => (v === null || v === undefined) ? '--' : `${(v * 100).toFixed(1)}%`;
  ['fpr', 'tpr', 'f1'].forEach(m => {
    rows.append('td')
      .attr('id', d => `m_${d.key}_${m}`)
      .attr('class', d => (d.live ? 'live-val' : ''))
      .text(d => d.ref ? fmtPct(REF[d.key]?.[m]) : '--');
  });
  rows.append('td')
    .attr('id', d => `m_${d.key}_auc`)
    .attr('class', d => (d.live ? 'live-val' : ''))
    .text(d => d.ref ? (REF[d.key]?.auc === null ? '--' : REF[d.key].auc.toFixed(3)) : '--');
  rows.append('td').text(d => d.priv);
  rows.append('td').append('span')
    .attr('class', d => `badge ${d.key === 'federated' ? 'badge-ok' : d.key === 'live_resid' ? 'badge-live' : 'badge-warn'}`)
    .text(d => d.badge);

  rows
    .on('mousemove', (ev, d) => {
      const fuente = d.live
        ? 'calculado en vivo sobre la ventana de 240 s de este stream'
        : 'bloque de test del notebook (gemelo_digital_se_ejecutado.ipynb)';
      tooltip.show(`
        <div class="tt-title">${d.name}</div>
        <div class="tt-row"><span>Fuente</span><b>${fuente}</b></div>
      `, ev);
    })
    .on('mouseleave', () => tooltip.hide());
}

export function updateLiveMetrics(stats) {
  if (!stats) return;
  const set = (id, v, pct = true) => {
    const el = document.getElementById(id);
    if (el) el.textContent = pct ? `${(v * 100).toFixed(1)}%` : v;
  };
  set('m_live_resid_fpr', stats.fpr);
  set('m_live_resid_tpr', stats.tpr);
  set('m_live_resid_f1', stats.f1);
  set('m_live_resid_auc', stats.auc === null ? '--' : stats.auc.toFixed(3), false);
}
