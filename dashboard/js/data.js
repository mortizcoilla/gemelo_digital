// data.js — Generador de datos sinteticos para el dashboard.
// Reproduce el pipeline del notebook: gemelo digital de la SE,
// observaciones = gemelo + ruido + firmas de falla, y etiquetas.
// Orden de features identico al notebook (22 columnas).

export const SEED_BASE = 42;
let rng = mulberry32(SEED_BASE);

function mulberry32(a) {
  return function () {
    let t = (a += 0x6D2B79F5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function randn() {
  // Box-Muller
  const u = 1 - rng();
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }

// Topologia de la SE (parametros, misma base que el notebook)
export const SE = {
  S_nom: 30,           // MVA por trafo
  V_at: 110,           // kV
  V_bt: 23,            // kV
  I_max: 800,          // A nominal por linea
  V_min: 0.93, V_max: 1.07,
  P_nom: [6.0, 4.5, 5.0, 3.5],   // MW por linea
  Q_nom: [1.5, 1.2, 1.3, 0.9],
  R_linea: [0.04, 0.06, 0.05, 0.07],
  X_linea: [0.08, 0.10, 0.09, 0.11],
};

// 22 features, mismo orden que el notebook:
// 0 V_at | 1-4 V_bt1..4 | 5-8 I_L1..4 | 9 S_total | 10 P_total | 11 Q_total
// 12-13 T_dev_T1/T2 | 14 f | 15 rocof | 16 thd_V | 17 thd_I | 18 ang_V
// 19 ang_I | 20 cos_phi | 21 desbalance
export const FEATURES = [
  'V_at', 'V_bt1', 'V_bt2', 'V_bt3', 'V_bt4',
  'I_L1', 'I_L2', 'I_L3', 'I_L4',
  'S_total', 'P_total', 'Q_total',
  'T_dev_T1', 'T_dev_T2',
  'f', 'rocof',
  'thd_V', 'thd_I',
  'ang_V', 'ang_I',
  'cos_phi', 'desbalance',
];
export const N_FEATURES = FEATURES.length; // 22

export const FEATURE_LABELS = {
  V_at: 'V_AT', V_bt1: 'V_BT1', V_bt2: 'V_BT2', V_bt3: 'V_BT3', V_bt4: 'V_BT4',
  I_L1: 'I_L1', I_L2: 'I_L2', I_L3: 'I_L3', I_L4: 'I_L4',
  S_total: 'S_tot', P_total: 'P_tot', Q_total: 'Q_tot',
  T_dev_T1: 'T_dev T1', T_dev_T2: 'T_dev T2',
  f: 'Frecuencia', rocof: 'ROCOF',
  thd_V: 'THD V', thd_I: 'THD I',
  ang_V: 'ang V', ang_I: 'ang I',
  cos_phi: 'cos(phi)', desbalance: 'Desbalance',
};

// Catalogo de fallas (duracion minima 2 s: a 1 Hz, eventos mas cortos
// quedan sub-muestreados y no existen para el detector)
export const CATALOGO = {
  F1: { nombre: 'Cortocircuito monofasico L1', severidad: 0.85, duracion: [2, 10] },
  F2: { nombre: 'Sobrecarga sostenida L2',     severidad: 0.45, duracion: [60, 600] },
  F3: { nombre: 'Sobretension AT',             severidad: 0.90, duracion: [2, 8] },
  F4: { nombre: 'Degradacion de aislamiento',  severidad: 0.30, duracion: [1800, 7200] },
  F5: { nombre: 'Perdida de comunicacion PMU', severidad: 0.50, duracion: [10, 600] },
  F6: { nombre: 'Desbalance de cargas',        severidad: 0.35, duracion: [300, 3600] },
  F7: { nombre: 'Falla incipiente de buje',    severidad: 0.25, duracion: [3600, 14400] },
  F8: { nombre: 'Evento climatico extremo',    severidad: 0.70, duracion: [600, 14400] },
};

// Indices de features usados por las firmas de falla
const IX = {
  V_at: 0, V_bt1: 1, V_bt3: 3,
  I_L1: 5, I_L2: 6, I_L3: 7, I_L4: 8,
  S_total: 9, P_total: 10,
  f: 14, rocof: 15, thd_V: 16, thd_I: 17,
  cos_phi: 20, desbalance: 21,
};

// Estado del gemelo (predice lo esperado a partir de la carga)
class Gemelo {
  constructor() {
    this.T_dev = 38.0;
    this.t = 0;
  }
  predecir(P_load, Q_load) {
    // Tension BT regulada (tap en 0 para el estado esperado)
    const V_bt = 1.0;
    // Corriente por linea con caida de tension radial
    const I = P_load.map((P, k) => {
      const V_k = Math.max(0.7,
        V_bt - (SE.R_linea[k] * P + SE.X_linea[k] * Q_load[k]) / SE.S_nom);
      const S_k = Math.sqrt(P * P + Q_load[k] * Q_load[k]);
      return S_k * 1000 / (Math.sqrt(3) * SE.V_bt * V_k);
    });
    const P_total = P_load.reduce((a, b) => a + b, 0);
    const Q_total = Q_load.reduce((a, b) => a + b, 0);
    const S_total = Math.sqrt(P_total * P_total + Q_total * Q_total);
    const f = clamp(50.0 - 0.005 * Math.max(0, P_total - 25), 49.5, 50.5);
    // Termica de primer orden (tau = 90 min)
    const S_pu = S_total / (2 * SE.S_nom);
    const P_termico = 180 * S_pu * S_pu * 2;   // W, ambos trafos
    const h = 350, T_amb = 25;
    this.T_dev += (P_termico - h * (this.T_dev - T_amb)) / (h * 90 * 60);
    const I_mean = I.reduce((a, b) => a + b, 0) / I.length;
    const I_maxv = Math.max(...I);
    return {
      V_bt, V_at: 1.0, I, P_total, Q_total, S_total, f,
      T_dev: this.T_dev,
      cos_phi: P_total / Math.max(S_total, 1e-6),
      desbalance: (I_maxv - I_mean) / Math.max(I_mean, 1e-6),
    };
  }
}

export class DataEngine {
  constructor({ seId = 'SE-PADRE-01', duracionS = 14400, fs = 1, k = 1 } = {}) {
    this.seId = seId;
    this.duracion = duracionS;
    this.fs = fs;
    this.n = Math.floor(duracionS * fs);
    this.k = k;
    this.gemelo = new Gemelo();
    this.tickIdx = 0;
    this.speed = 1;
    this.exp = [];          // serie esperada del gemelo (n x 22)
    this.eventos = [];      // eventos programados + inyectados
    this.y = new Array(this.n).fill(0);
    this._currentFalla = null;
    this._fps = { frames: 0, t0: performance.now() };
    this._handle = null;
    this._generarBase();
    this._programarFallas();
  }

  _generarBase() {
    for (let i = 0; i < this.n; i++) {
      const t = i / this.fs;
      const hora = ((t / 3600) % 24 + 24) % 24;
      // Perfil horario (pico 20 h, valle madrugada) + variabilidad
      const pico = Math.exp(-0.5 * Math.pow((hora - 20) / 2.0, 2));
      const mediodia = Math.exp(-0.5 * Math.pow((hora - 12) / 1.5, 2));
      const factor = clamp(
        (0.55 + 0.30 * pico + 0.15 * mediodia) * (0.85 + 0.1 * this.k) + randn() * 0.04,
        0.25, 1.30);
      const P_load = SE.P_nom.map(p => p * factor);
      const Q_load = SE.Q_nom.map(q => q * factor);
      const est = this.gemelo.predecir(P_load, Q_load);
      this.exp.push([
        est.V_at,
        est.V_bt, est.V_bt, est.V_bt, est.V_bt,
        ...est.I,
        est.S_total,
        est.P_total,
        est.Q_total,
        est.T_dev,
        est.T_dev - 1,
        est.f,
        0,
        2.0, 3.0,
        0, 0,
        est.cos_phi,
        est.desbalance,
      ]);
    }
  }

  _programarFallas() {
    const lambdas = [0.5, 0.3, 0.1, 0.4, 0.35, 0.6, 0.2, 5 / 365];
    const codigos = Object.keys(CATALOGO);
    // Eventos Poisson (realismo estadistico del catalogo)
    codigos.forEach((codigo, idx) => {
      const lam = lambdas[idx] * (this.duracion / 86400);
      const nEv = Math.max(1, Math.round(lam * (0.5 + rng())));
      for (let e = 0; e < nEv; e++) {
        const ini = Math.floor(rng() * (this.n - 10));
        this._agregarEvento(codigo, ini, true);
      }
    });
    // Cobertura garantizada: 1 evento por tipo repartido en slots
    // uniformes de la ventana (mismo criterio que el notebook), con
    // duracion acotada (25% del rango del catalogo, tope 30 min)
    const slot = this.n / codigos.length;
    const orden = codigos.map((_, i) => i).sort(() => rng() - 0.5);
    orden.forEach((ci, k) => {
      const codigo = codigos[ci];
      const cat = CATALOGO[codigo];
      const ini = Math.floor(k * slot + rng() * slot * 0.5);
      const dur = Math.min(1800,
        cat.duracion[0] + (cat.duracion[1] - cat.duracion[0]) * 0.25);
      this._agregarEventoEn(codigo, ini, Math.max(2, Math.floor(dur)));
    });
    // Evento temprano garantizado (~min 4) para que la demo arranque con accion
    this._agregarEventoEn('F2', 4 * 60, 330);
    this.eventos.sort((a, b) => a.idx_ini - b.idx_ini);
    this._remarcarEtiquetas();
  }

  _agregarEvento(codigo, ini, aleatorio) {
    const cat = CATALOGO[codigo];
    const dur = aleatorio
      ? cat.duracion[0] + rng() * (cat.duracion[1] - cat.duracion[0])
      : (cat.duracion[0] + cat.duracion[1]) / 2;
    return this._agregarEventoEn(codigo, ini, Math.max(2, Math.floor(dur)));
  }

  _agregarEventoEn(codigo, ini, durSamples) {
    const cat = CATALOGO[codigo];
    const fin = Math.min(this.n, Math.max(ini + 2, ini + durSamples));
    const ev = {
      codigo,
      nombre: cat.nombre,
      severidad: cat.severidad,
      t_ini: ini / this.fs,
      t_fin: fin / this.fs,
      idx_ini: ini,
      idx_fin: fin,
      manual: false,
      detectado: false,
    };
    this.eventos.push(ev);
    return ev;
  }

  _remarcarEtiquetas() {
    this.y.fill(0);
    for (const ev of this.eventos) {
      for (let i = Math.max(0, ev.idx_ini); i < Math.min(this.n, ev.idx_fin); i++) {
        this.y[i] = 1;
      }
    }
  }

  // Inyeccion manual desde la UI: activa la falla en el playhead actual
  inject(codigo, durS) {
    const ini = this.tickIdx % this.n;
    const ev = this._agregarEventoEn(codigo, ini, Math.max(2, Math.floor(durS * this.fs)));
    ev.manual = true;
    this.eventos.sort((a, b) => a.idx_ini - b.idx_ini);
    this._remarcarEtiquetas();
    return ev;
  }

  start(cb) {
    this._handle = setInterval(() => {
      const sample = this._tick();
      this._fps.frames++;
      const now = performance.now();
      if (now - this._fps.t0 > 1000) {
        const fps = this._fps.frames / ((now - this._fps.t0) / 1000);
        this._fps = { frames: 0, t0: now };
        cb({ type: 'fps', value: fps * this.speed });
      }
      cb({ type: 'sample', data: sample });
    }, 1000 / (this.fs * this.speed));
  }

  stop() {
    if (this._handle) clearInterval(this._handle);
    this._handle = null;
  }

  setSpeed(mult) {
    // main.js se encarga de reiniciar el intervalo con el nuevo ritmo
    this.speed = mult;
  }

  _tick() {
    const i = this.tickIdx % this.n;
    const exp = this.exp[i];
    const obs = this._observar(exp, i);
    // Evento activo: el de inicio mas reciente entre los vigentes
    let activeEvt = null;
    for (const ev of this.eventos) {
      if (i >= ev.idx_ini && i < ev.idx_fin) {
        if (!activeEvt || ev.idx_ini >= activeEvt.idx_ini) activeEvt = ev;
      }
    }
    this._currentFalla = activeEvt;
    this.tickIdx++;
    return {
      i, t: i / this.fs,
      obs, exp,
      y: this.y[i],
      event: activeEvt,
    };
  }

  _observar(exp, i) {
    // Observacion = esperado + ruido de medicion; las firmas de falla
    // se aplican ENCIMA (mismo orden que el notebook corregido)
    const obs = exp.slice();
    obs[0] += randn() * 0.005; obs[1] += randn() * 0.005;
    obs[2] += randn() * 0.005; obs[3] += randn() * 0.005; obs[4] += randn() * 0.005;
    obs[5] += randn() * 2.0; obs[6] += randn() * 2.0;
    obs[7] += randn() * 2.0; obs[8] += randn() * 2.0;
    obs[9] += randn() * 0.05; obs[10] += randn() * 0.05; obs[11] += randn() * 0.03;
    obs[12] += randn() * 0.5; obs[13] += randn() * 0.5;
    obs[14] += randn() * 0.01;
    obs[15] = randn() * 0.02;
    obs[16] = 2.0 + randn() * 0.3;
    obs[17] = 3.0 + randn() * 0.4;
    obs[18] = randn() * 0.1; obs[19] = randn() * 0.1;
    obs[20] += randn() * 0.01;
    obs[21] = Math.max(0, obs[21] + randn() * 0.01);

    for (const ev of this.eventos) {
      if (i < ev.idx_ini || i >= ev.idx_fin) continue;
      const prog = (i - ev.idx_ini) / Math.max(1, ev.idx_fin - ev.idx_ini);
      const sev = ev.severidad;
      const ramp = Math.min(1, prog * 3);
      switch (ev.codigo) {
        case 'F1': // corto monofasico L1
          obs[IX.I_L1] += 350 * sev;
          obs[IX.V_at] -= 0.12 * sev;
          obs[IX.V_bt1] -= 0.18 * sev;
          obs[IX.f] += 0.5 * sev;
          obs[IX.thd_V] += 8.0 * sev;
          obs[IX.thd_I] += 6.0 * sev;
          break;
        case 'F2': // sobrecarga sostenida L2
          obs[IX.I_L2] += 80 * sev * ramp;
          obs[IX.P_total] += 1.5 * sev * ramp;
          obs[IX.S_total] += 1.5 * sev * ramp;
          obs[IX.cos_phi] -= 0.08 * sev * ramp;
          break;
        case 'F3': // sobretension AT
          obs[IX.V_at] += 0.08 * sev;
          obs[IX.f] += 0.15 * sev;
          break;
        case 'F4': // degradacion de aislamiento
          obs[IX.thd_V] += 5.0 * sev * ramp;
          obs[IX.thd_I] += 3.0 * sev * ramp;
          break;
        case 'F5': { // perdida PMU: f congelada en el ultimo valor
          const prev = this.exp[Math.max(0, i - 1)][IX.f] || 50.0;
          obs[IX.f] = prev;
          obs[IX.rocof] = 0;
          break;
        }
        case 'F6': // desbalance de cargas
          obs[IX.I_L1] += 40 * sev;
          obs[IX.I_L3] -= 30 * sev;
          obs[IX.V_bt3] -= 0.03 * sev;
          obs[IX.desbalance] += 0.08 * sev;
          break;
        case 'F7': // falla incipiente de buje (pulsos)
          obs[IX.thd_V] += (rng() > 0.95 ? 1 : 0) * 12 * sev;
          obs[IX.thd_I] += (rng() > 0.95 ? 1 : 0) * 8 * sev;
          break;
        case 'F8': // evento climatico
          obs[IX.V_at] -= 0.06 * sev;
          obs[IX.f] -= 0.12 * sev;
          obs[IX.I_L1] += 25 * sev;
          obs[IX.I_L2] += 25 * sev;
          obs[IX.I_L3] += 25 * sev;
          obs[IX.I_L4] += 25 * sev;
          break;
      }
    }
    obs[5] = Math.max(0, obs[5]);
    obs[6] = Math.max(0, obs[6]);
    obs[7] = Math.max(0, obs[7]);
    obs[8] = Math.max(0, obs[8]);
    return obs;
  }

  currentFalla() { return this._currentFalla; }
  currentIdx() { return this.tickIdx % this.n; }

  // Eventos visibles en un rango de indices (para bandas de falla)
  eventosEnRango(i0, i1) {
    return this.eventos.filter(ev => ev.idx_ini < i1 && ev.idx_fin > i0);
  }
}
