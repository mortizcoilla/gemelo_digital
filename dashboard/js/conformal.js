// conformal.js — Detector no supervisado + conformal prediction.
//
// Score de reconstruccion simplificado para el navegador: suma ponderada
// de residuos normalizados por feature (proxy del AE-LSTM del notebook).
// El cuantil q_hat se calibra con las primeras CAL_WINDOW muestras LIMPIAS
// (sin evento activo): el conjunto de intercambiabilidad son datos normales.

import { N_FEATURES, FEATURES } from './data.js';

// Pesos por feature (sensibilidad a fallas del catalogo)
const W = [
  1.5, 1.0, 1.0, 1.0, 1.0,   // tensiones
  1.2, 1.2, 1.2, 1.2,        // corrientes
  0.8,                        // S_total
  0.8, 0.8,                   // P, Q
  0.7, 0.7,                   // termico
  1.6, 1.6,                   // f, rocof
  1.3, 1.3,                   // thd
  0.5, 0.5,                   // angulos
  0.6, 0.9,                   // cos_phi, desbalance
];

// Normalizacion por orden de magnitud tipico del residuo de cada feature
const NORM = [
  0.005, 0.005, 0.005, 0.005, 0.005,   // tensiones
  2.0, 2.0, 2.0, 2.0,                  // corrientes (sigma del ruido)
  0.05,                                 // S_total
  0.05, 0.03,                           // P, Q
  0.5, 0.5,                             // termico
  0.01, 0.02,                           // f, rocof
  0.3, 0.4,                             // thd
  0.1, 0.1,                             // angulos
  0.01, 0.01,                           // cos_phi, desbalance
];

export const ALPHA = 0.05;
const CAL_WINDOW = 200;
export const MAX_WINDOW = 240;

export class ConformalDetector {
  constructor() {
    this.calib = [];
    this.q_hat = null;
    // Buffer alineado: cada entrada { t, score, inEvent } — score y etiqueta
    // siempre corresponden a la MISMA muestra (bug corregido).
    this.window = [];
    this.alertActive = false;
  }

  residuosNormalizados(obs, exp) {
    const out = new Array(N_FEATURES);
    for (let i = 0; i < N_FEATURES; i++) {
      out[i] = (obs[i] - exp[i]) / NORM[i];
    }
    return out;
  }

  score(obs, exp) {
    const nr = this.residuosNormalizados(obs, exp);
    let s = 0;
    for (let i = 0; i < N_FEATURES; i++) s += Math.abs(nr[i]) * W[i];
    return s / N_FEATURES;
  }

  update(sample) {
    const nr = this.residuosNormalizados(sample.obs, sample.exp);
    let s = 0;
    const contrib = [];
    for (let i = 0; i < N_FEATURES; i++) {
      const c = Math.abs(nr[i]) * W[i];
      s += c;
      contrib.push({ feat: FEATURES[i], idx: i, v: c });
    }
    s /= N_FEATURES;
    contrib.sort((a, b) => b.v - a.v);

    this.window.push({ t: sample.t, score: s, inEvent: !!sample.event, alert: false });
    if (this.window.length > MAX_WINDOW) this.window.shift();

    // Calibracion conformal: solo muestras sin evento activo
    if (this.q_hat === null && !sample.event) {
      this.calib.push(s);
      if (this.calib.length >= CAL_WINDOW) {
        const sorted = [...this.calib].sort((a, b) => a - b);
        const n = sorted.length;
        // Correccion de muestra finita: cuantil ceil((n+1)(1-alpha))/n
        const rank = Math.min(n, Math.ceil((n + 1) * (1 - ALPHA)));
        this.q_hat = sorted[rank - 1];
      }
    }

    const alert = this.q_hat !== null && s > this.q_hat;
    this.alertActive = alert;
    if (this.window.length) this.window[this.window.length - 1].alert = alert;

    return {
      score: s,
      q_hat: this.q_hat,
      alert,
      contrib: contrib.slice(0, 8),
      calibrando: this.q_hat === null,
      progresoCal: this.calib.length / CAL_WINDOW,
    };
  }

  // Metricas en vivo sobre la ventana alineada (score y verdad juntas)
  liveMetrics() {
    if (this.q_hat === null || this.window.length < 50) return null;
    let tp = 0, fp = 0, tn = 0, fn = 0;
    for (const w of this.window) {
      if (w.alert && w.inEvent) tp++;
      else if (w.alert && !w.inEvent) fp++;
      else if (!w.alert && w.inEvent) fn++;
      else tn++;
    }
    const fpr = (fp + tn) > 0 ? fp / (fp + tn) : 0;
    const tpr = (tp + fn) > 0 ? tp / (tp + fn) : 0;
    const prec = (tp + fp) > 0 ? tp / (tp + fp) : 0;
    const f1 = (prec + tpr) > 0 ? 2 * prec * tpr / (prec + tpr) : 0;
    return { fpr, tpr, f1, auc: this.auc(), tp, fp, tn, fn };
  }

  // AUC por rango (Mann-Whitney) sobre la ventana
  auc() {
    const pos = this.window.filter(w => w.inEvent).map(w => w.score);
    const neg = this.window.filter(w => !w.inEvent).map(w => w.score);
    if (pos.length === 0 || neg.length === 0) return null;
    let wins = 0;
    for (const p of pos) {
      for (const n of neg) {
        if (p > n) wins += 1;
        else if (p === n) wins += 0.5;
      }
    }
    return wins / (pos.length * neg.length);
  }

  // Cobertura: fraccion de muestras EN evento que dispararon alerta
  coverage() {
    const ev = this.window.filter(w => w.inEvent);
    if (ev.length === 0) return null;
    return ev.filter(w => w.alert).length / ev.length;
  }
}
