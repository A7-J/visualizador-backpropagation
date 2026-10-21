// src/ui/charts.js
const MAX_POINTS = 200;
const STRIDE = 15;          // 15 frames viram 1 ponto: janela de 3000 frames (~50 s a 60 fps)
const SMOOTH = 0.6;         // média móvel exponencial sobre os pontos já agregados
const CHANCE = Math.log(4); // loss de um chute uniforme entre 4 ações (~1,386)

export class LossChart {
  constructor() {
    this.seriesSGD = [];
    this.seriesAdam = [];
    this.acc = { sgd: { sum: 0, n: 0 }, adam: { sum: 0, n: 0 } };
  }

  // L = -soma y_i * log(y_hat_i). Complexidade: O(n), n = 4
  static crossEntropy(yTarget, yPred) {
    const EPS = 1e-9;
    let loss = 0;
    for (let i = 0; i < yTarget.length; i++) loss -= yTarget[i] * Math.log(yPred[i] + EPS);
    return loss;
  }

  // Acumula STRIDE frames; só então gera 1 ponto (média + suavização).
  // Complexidade: O(1) por chamada (o shift de 200 itens ocorre 1 vez a cada 15 chamadas)
  push(optimizerName, lossValue) {
    const key = optimizerName === 'adam' ? 'adam' : 'sgd';
    const acc = this.acc[key];
    acc.sum += lossValue;
    acc.n++;
    if (acc.n < STRIDE) return;

    const avg = acc.sum / acc.n;
    acc.sum = 0;
    acc.n = 0;

    const series = key === 'adam' ? this.seriesAdam : this.seriesSGD;
    const prev = series.length ? series[series.length - 1] : avg;
    series.push(SMOOTH * prev + (1 - SMOOTH) * avg);
    if (series.length > MAX_POINTS) series.shift();
  }

  reset() {
    this.seriesSGD = [];
    this.seriesAdam = [];
    this.acc = { sgd: { sum: 0, n: 0 }, adam: { sum: 0, n: 0 } };
  }

  // Complexidade: O(MAX_POINTS), constante
  draw(ctx, width, height) {
    ctx.clearRect(0, 0, width, height);
    const all = [...this.seriesSGD, ...this.seriesAdam];
    if (all.length === 0) return;

    // Escala estável: sempre inclui o chute aleatório, então o eixo não "pula"
    const maxLoss = Math.max(...all, CHANCE) * 1.35;
    const toY = v => height - (v / maxLoss) * (height - 8) - 4;
    const toX = i => (i / (MAX_POINTS - 1)) * width;

    // Linha tracejada: nível do chute aleatório
    ctx.save();
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = '#475569';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, toY(CHANCE));
    ctx.lineTo(width, toY(CHANCE));
    ctx.stroke();
    ctx.restore();

    this._drawSeries(ctx, this.seriesSGD, toX, toY, '#f87171');
    this._drawSeries(ctx, this.seriesAdam, toX, toY, '#4ade80');
    this._drawLegend(ctx);
  }

  _drawSeries(ctx, series, toX, toY, color) {
    if (series.length < 2) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(toX(0), toY(series[0]));
    for (let i = 1; i < series.length; i++) ctx.lineTo(toX(i), toY(series[i]));
    ctx.stroke();
  }

  _drawLegend(ctx) {
    const last = s => (s.length ? s[s.length - 1].toFixed(3) : '-');
    ctx.font = '11px monospace';
    ctx.textAlign = 'left';
    ctx.fillStyle = '#f87171';
    ctx.fillText(`SGD  ${last(this.seriesSGD)}`, 8, 14);
    ctx.fillStyle = '#4ade80';
    ctx.fillText(`Adam ${last(this.seriesAdam)}`, 8, 28);
    ctx.fillStyle = '#64748b';
    ctx.fillText(`-- chute ${CHANCE.toFixed(3)} | janela ~50 s`, 8, 42);
  }
}