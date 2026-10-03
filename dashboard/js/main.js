// main.js — Orquestador del dashboard.
// Stream 1 Hz -> KPIs, topologia, senales, detector conformal, heatmap,
// federado, timeline y metricas. Todo cruzado: click en alimentador,
// fila del heatmap o barra de contribucion selecciona la feature activa.

import { DataEngine, CATALOGO } from './data.js';
import { initTopology } from './topology.js';
import { SignalsPlot } from './signals.js';
import { ScorePlot, FeatureBars, HeatmapPlot } from './residuals.js';
import { ConformalDetector } from './conformal.js';
import { FederatedView, renderFederatedMap } from './federated.js';
import { initMetrics, updateLiveMetrics } from './metrics.js';
import { KpiStrip } from './kpis.js';
import { Timeline } from './timeline.js';

// ---------------------------------------------------------------
// Tooltip global
// ---------------------------------------------------------------
const tooltip = {
  el: null,
  show(html, ev) {
    if (!this.el) {
      this.el = document.createElement('div');
      this.el.className = 'tooltip';
      document.body.appendChild(this.el);
    }
    this.el.innerHTML = html;
    this.el.classList.add('visible');
    const pad = 14;
    let x = ev.clientX + pad;
    let y = ev.clientY + pad;
    const r = this.el.getBoundingClientRect();
    if (x + r.width > window.innerWidth - 8) x = ev.clientX - r.width - pad;
    if (y + r.height > window.innerHeight - 8) y = ev.clientY - r.height - pad;
    this.el.style.left = `${x}px`;
    this.el.style.top = `${y}px`;
  },
  hide() {
    if (this.el) this.el.classList.remove('visible');
  },
};

// ---------------------------------------------------------------
// Estado
// ---------------------------------------------------------------
const engine = new DataEngine({ seId: 'SE-PADRE-01', duracionS: 14400, fs: 1, k: 1 });
const detector = new ConformalDetector();
const fed = new FederatedView();

const els = {
  globalStatus: document.getElementById('globalStatus'),
  seId: document.getElementById('seId'),
  simClock: document.getElementById('simClock'),
  fpsCounter: document.getElementById('fpsCounter'),
  topoEstado: document.getElementById('topoEstado'),
  tapPos: document.getElementById('tapPos'),
  residualSummary: document.getElementById('residualSummary'),
  alertScore: document.getElementById('alertScore'),
  alertArc: document.getElementById('alertArc'),
  alertThreshold: document.getElementById('alertThreshold'),
  alertFpr: document.getElementById('alertFpr'),
  alertCoverage: document.getElementById('alertCoverage'),
  alertCal: document.getElementById('alertCal'),
  alertList: document.getElementById('alertList'),
  fedRounds: document.getElementById('fedRounds'),
  fedLoss: document.getElementById('fedLoss'),
  fedGap: document.getElementById('fedGap'),
  fedBar: document.getElementById('fedBar'),
};

// ---------------------------------------------------------------
// Paneles
// ---------------------------------------------------------------
const kpis = new KpiStrip('#kpisBody', tooltip);
const topo = initTopology('#topologyBody', engine, tooltip, feat => setFeature(feat));
const signalsPlot = new SignalsPlot('#signalsBody', engine, 'V_at', tooltip);
const scorePlot = new ScorePlot('#scoreBody', tooltip);
const featBars = new FeatureBars('#featBarsBody', feat => setFeature(feat));
const heatmap = new HeatmapPlot('#heatmapBody', tooltip, engine, feat => setFeature(feat));
const fedMap = renderFederatedMap('#fedMap', fed, tooltip);
const timeline = new Timeline('#timelineBody', engine, engine.duracion, tooltip);

initMetrics('#metricsRows', tooltip);

// ---------------------------------------------------------------
// Seleccion de feature sincronizada (tabs + clicks cruzados)
// ---------------------------------------------------------------
function setFeature(feat) {
  signalsPlot.setFeature(feat);
  document.querySelectorAll('#signalTabs .tab').forEach(t => {
    t.classList.toggle('active', t.dataset.feat === feat);
  });
}

document.getElementById('signalTabs').addEventListener('click', e => {
  const btn = e.target.closest('.tab');
  if (!btn) return;
  setFeature(btn.dataset.feat);
});

// Cursor sincronizado senales <-> score
signalsPlot.onHover = t => { if (t !== null) scorePlot.showCursorAt(t); };
scorePlot.onHover = t => { if (t !== null) signalsPlot.showCursorAt(t); };

// ---------------------------------------------------------------
// Controles del stream
// ---------------------------------------------------------------
let running = true;

function startStream() {
  engine.start(onMsg);
  running = true;
  updatePlayBtn();
}
function stopStream() {
  engine.stop();
  running = false;
  updatePlayBtn();
}
function updatePlayBtn() {
  const btn = document.getElementById('btnPlay');
  btn.textContent = running ? '❚❚ Pausar' : '▶ Reanudar';
  btn.classList.toggle('paused', !running);
}

document.getElementById('btnPlay').addEventListener('click', () => {
  running ? stopStream() : startStream();
});

document.querySelectorAll('#speedCtl .spd').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('#speedCtl .spd').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const mult = Number(btn.dataset.mult);
    engine.setSpeed(mult);
    if (running) {
      engine.stop();
      engine.start(onMsg);
    }
  });
});

document.getElementById('btnInject').addEventListener('click', () => {
  const codigo = document.getElementById('injectSel').value;
  engine.inject(codigo, 40);
});

document.getElementById('btnReset').addEventListener('click', () => location.reload());

// ---------------------------------------------------------------
// Alertas
// ---------------------------------------------------------------
const MAX_ALERTS = 14;

function pushAlert({ tipo, nombre, t, sev, latencia }) {
  const li = document.createElement('li');
  li.className = `alert-item fade-in ${tipo === 'FP' ? 'warn' : ''}`;
  li.innerHTML = `
    <div class="alert-title">
      <span>${tipo === 'FP' ? 'Falsa alarma' : nombre}</span>
      <span>${formatT(t)}</span>
    </div>
    <div class="alert-meta">
      ${tipo === 'TP'
        ? `detectado en ${latencia}s · sev ${(sev * 100).toFixed(0)}% · TP`
        : 'sin evento activo · FP'}
    </div>`;
  els.alertList.insertBefore(li, els.alertList.firstChild);
  while (els.alertList.children.length > MAX_ALERTS) {
    els.alertList.removeChild(els.alertList.lastChild);
  }
}

function formatT(t) {
  const m = Math.floor(t / 60);
  const s = Math.floor(t % 60);
  return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
}

// Detecciones para el timeline: key -> marca
const detecciones = new Map();

// ---------------------------------------------------------------
// Loop principal
// ---------------------------------------------------------------
function onMsg(msg) {
  if (msg.type === 'fps') {
    els.fpsCounter.textContent = msg.value.toFixed(0);
    return;
  }

  const sample = msg.data;

  // 1. KPIs + paneles electricos
  kpis.push(sample);
  topo.update(sample);
  signalsPlot.push(sample);
  heatmap.push(sample);

  // 2. Detector conformal
  const r = detector.update(sample);
  scorePlot.setThreshold(r.q_hat);
  scorePlot.push({
    t: sample.t,
    score: r.score,
    alert: r.alert,
    inEvent: !!sample.event,
    eventName: sample.event ? sample.event.codigo : null,
  });
  featBars.update(r.contrib);

  // Gauge
  const pct = r.q_hat !== null ? Math.min(1, r.score / Math.max(1e-6, r.q_hat)) : 0;
  els.alertScore.textContent = r.score.toFixed(2);
  if (r.q_hat !== null) {
    els.alertThreshold.textContent = r.q_hat.toFixed(2);
    els.alertArc.setAttribute('stroke-dasharray', `${(pct * 314).toFixed(0)} 314`);
    els.alertArc.setAttribute('stroke',
      r.alert ? '#e63946' : pct > 0.7 ? '#f4d35e' : '#2ca02c');
    els.alertCal.textContent = 'calibrado';
  } else {
    els.alertThreshold.textContent = 'calibrando';
    els.alertCal.textContent = `${Math.round(r.progresoCal * 100)}%`;
  }

  // 3. Clasificacion de alertas (TP con latencia / FP)
  if (r.alert) {
    if (sample.event) {
      const ev = sample.event;
      const lat = Math.max(0, Math.round(sample.t - ev.t_ini));
      if (!ev.detectado) {
        ev.detectado = true;
        detecciones.set(`${ev.codigo}_${ev.idx_ini}`, {
          key: `${ev.codigo}_${ev.idx_ini}`,
          codigo: ev.codigo,
          t: sample.t,
        });
        pushAlert({ tipo: 'TP', nombre: ev.nombre, t: sample.t, sev: ev.severidad, latencia: lat });
      }
    } else {
      pushAlert({ tipo: 'FP', nombre: '', t: sample.t, sev: 0, latencia: 0 });
    }
  }

  // 4. Metricas en vivo (buffer alineado dentro del detector)
  const stats = detector.liveMetrics();
  updateLiveMetrics(stats);
  if (stats) {
    els.alertFpr.textContent = `${(stats.fpr * 100).toFixed(1)}%`;
  }
  const cov = detector.coverage();
  els.alertCoverage.textContent = cov === null ? '--' : `${(cov * 100).toFixed(0)}%`;

  // 5. Federado
  const fedState = fed.tickAll(sample);
  els.fedRounds.textContent = `${fedState.round} / ${fed.maxRounds}`;
  els.fedLoss.textContent = fedState.globalLoss.toFixed(4);
  const gap = (fedState.globalLoss - fed.centralLoss) / fed.centralLoss * 100;
  els.fedGap.textContent = `${gap >= 0 ? '+' : ''}${gap.toFixed(1)}%`;
  els.fedGap.style.color = Math.abs(gap) < 15 ? '#6ad36e' : '#f4d35e';
  if (els.fedBar) {
    els.fedBar.style.width = `${(fedState.round / fed.maxRounds) * 100}%`;
  }
  fedMap.update();

  // 6. Timeline
  timeline.update(sample.t, [...detecciones.values()]);

  // 7. Estado global
  els.simClock.textContent = formatT(sample.t);
  els.globalStatus.textContent = r.alert ? 'ALERTA' : running ? 'EN LINEA' : 'PAUSA';
  els.globalStatus.parentElement.classList.toggle('pill-bad', r.alert);
  els.globalStatus.parentElement.classList.toggle('pill-ok', !r.alert);

  // 8. Resumen de residuos y footer de topologia
  let rmag2 = 0;
  for (let i = 0; i < sample.obs.length; i++) {
    rmag2 += (sample.obs[i] - sample.exp[i]) ** 2;
  }
  els.residualSummary.textContent =
    `||r|| = ${Math.sqrt(rmag2).toFixed(4)} · ${r.alert ? 'ALERTA conformal' : 'dentro de banda'}`;
  els.residualSummary.style.color = r.alert ? '#e63946' : '#aab3c0';

  els.topoEstado.textContent = sample.event ? `${sample.event.codigo} · ${sample.event.nombre}` : 'Operacion normal';
  els.topoEstado.style.color = sample.event ? '#e63946' : '#2ca02c';
  els.tapPos.textContent =
    `${(1 + Math.sin(sample.t / 600) * 3).toFixed(0)} / ${(1 + Math.cos(sample.t / 700) * 3).toFixed(0)}`;
}

startStream();
window.addEventListener('beforeunload', () => engine.stop());
window.engine = engine;        // debug en consola
window.__onMsg = onMsg;        // debug: avanzar el stream manualmente
