// src/game/telemetry.js
const ACTIONS = ['jump', 'left', 'right', 'none'];
const NONE_KEEP_EVERY = 8; // grava 1 a cada 8 frames "none"; ações reais sempre
const STATE_SIZE = 4;      // [altura, velocidade vertical, posição x, distância ao obstáculo]

export class Telemetry {
  constructor() {
    this.dataset = [];
    this.recording = false;
    this.noneCounter = 0;
    this.classCounts = [0, 0, 0, 0]; // amostras por ação, na ordem de ACTIONS
  }

  start() { this.recording = true; }
  stop()  { this.recording = false; }
  clear() { this.dataset = []; this.noneCounter = 0; this.classCounts = [0, 0, 0, 0]; }

  // Complexidade: O(1)
  record(state, action) {
    if (!this.recording) return;

    if (action === 'none') {
      this.noneCounter++;
      if (this.noneCounter % NONE_KEEP_EVERY !== 0) return;
    }

    const oneHot = ACTIONS.map(a => (a === action ? 1 : 0));
    this.dataset.push({ state: [...state], action: oneHot });
    this.classCounts[oneHot.indexOf(1)]++;
  }

  size() { return this.dataset.length; }

  // Cópia das contagens [jump, left, right, none]. Complexidade: O(1)
  counts() { return [...this.classCounts]; }

  // Complexidade: O(n) + O(batchSize)
  sampleBalancedBatch(batchSize) {
    const byClass = [[], [], [], []];
    for (const sample of this.dataset) {
      byClass[sample.action.indexOf(1)].push(sample);
    }

    const perClass = Math.ceil(batchSize / 4);
    const batch = [];

    for (const classSamples of byClass) {
      if (classSamples.length === 0) continue;
      for (let i = 0; i < perClass; i++) {
        batch.push(classSamples[Math.floor(Math.random() * classSamples.length)]);
      }
    }

    for (let i = batch.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [batch[i], batch[j]] = [batch[j], batch[i]];
    }
    return batch;
  }

  // Valida CADA amostra antes de substituir o dataset (tudo ou nada).
  // Complexidade: O(n * k), k = 4
  load(data) {
    if (!Array.isArray(data)) throw new Error('o JSON não é uma lista de amostras');
    const clean = [];
    const counts = [0, 0, 0, 0];
    data.forEach((s, idx) => {
      const okState = s && Array.isArray(s.state) && s.state.length === STATE_SIZE && s.state.every(Number.isFinite);
      const okAction = s && Array.isArray(s.action) && s.action.length === ACTIONS.length
        && s.action.every(v => v === 0 || v === 1)
        && s.action.reduce((a, b) => a + b, 0) === 1;
      if (!okState || !okAction) throw new Error(`amostra ${idx} com formato inválido`);
      clean.push({ state: [...s.state], action: [...s.action] });
      counts[s.action.indexOf(1)]++;
    });
    this.dataset = clean;
    this.classCounts = counts;
  }

  toJSON() { return JSON.stringify(this.dataset); }
}

export { ACTIONS };