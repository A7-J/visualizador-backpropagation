// src/main.js
import { GameEngine } from './game/engine.js';
import { Telemetry, ACTIONS } from './game/telemetry.js';
import { NeuralNetwork } from './math/nn.js';
import { SGD, Adam } from './math/optimizers.js';
import { softmaxJacobian } from './math/activations.js';
import { gradientCheck } from './math/gradcheck.js';
import { drawNetworkDiagram, drawMatrixPanel } from './ui/visualizer.js';
import { LossChart } from './ui/charts.js';

// --- DOM ---
const canvasGame = document.getElementById('canvas-game');
const ctxGame = canvasGame.getContext('2d');
const canvasNetwork = document.getElementById('canvas-network');
const ctxNetwork = canvasNetwork.getContext('2d');
const canvasMatrix = document.getElementById('canvas-matrix');
const ctxMatrix = canvasMatrix.getContext('2d');
const canvasChart = document.getElementById('canvas-chart');
const ctxChart = canvasChart.getContext('2d');

const btnRecord = document.getElementById('btn-record');
const btnTrain = document.getElementById('btn-train');
const btnOptimizer = document.getElementById('btn-optimizer');
const optimizerLabel = document.getElementById('optimizer-label');
const btnActivation = document.getElementById('btn-activation');
const activationLabel = document.getElementById('activation-label');
const btnView = document.getElementById('btn-view');
const viewLabel = document.getElementById('view-label');
const panelTitle = document.getElementById('panel-matrix-title');
const btnGradCheck = document.getElementById('btn-gradcheck');
const btnExport = document.getElementById('btn-export');
const btnImport = document.getElementById('btn-import');
const fileImport = document.getElementById('file-import');
const btnReset = document.getElementById('btn-reset');
const phaseLabel = document.getElementById('phase-label');
const scoreValue = document.getElementById('score-value');
const datasetSize = document.getElementById('dataset-size');

// --- Resolução do canvas (devicePixelRatio) --- O(1) por canvas
function fitCanvasToDisplaySize(canvas) {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = canvas.clientWidth * dpr;
  canvas.height = canvas.clientHeight * dpr;
  canvas.getContext('2d').setTransform(dpr, 0, 0, dpr, 0, 0);
  return { width: canvas.clientWidth, height: canvas.clientHeight };
}

const allCanvases = [canvasGame, canvasNetwork, canvasMatrix, canvasChart];
let logicalSizes = {};
function resizeAllCanvases() {
  for (const c of allCanvases) logicalSizes[c.id] = fitCanvasToDisplaySize(c);
}
resizeAllCanvases();
window.addEventListener('resize', () => {
  resizeAllCanvases();
  engine.canvasWidth = logicalSizes['canvas-game'].width;
  engine.canvasHeight = logicalSizes['canvas-game'].height;
});

// --- Núcleo ---
const engine = new GameEngine(logicalSizes['canvas-game'].width, logicalSizes['canvas-game'].height);
const telemetry = new Telemetry();
const chart = new LossChart();

const LR_SGD = 0.15;
const LR_ADAM = 0.01;
const BATCH_SIZE = 8;
const SWEEP_SPEED = 1.2; // camadas por segundo da onda reversa

// Duas redes com os MESMOS pesos iniciais, treinadas nos MESMOS batches
let activation = 'sigmoid';
let netSGD, netAdam, optSGD, optAdam;
function initNetworks() {
  netSGD = new NeuralNetwork([4, 6, 4], { hidden: activation });
  netAdam = netSGD.clone();
  optSGD = new SGD(LR_SGD);
  optAdam = new Adam(netAdam, LR_ADAM);
}
initNetworks();

let activeName = 'sgd';
let phase = 'expert';
let viewMode = 'jacobian'; // 'jacobian' | 'equations'
const emptyInfo = () => ({
  sgd:  { deltas: null, gradW: null, probe: null, loss: 0 },
  adam: { deltas: null, gradW: null, probe: null, loss: 0 },
});
let info = emptyInfo();
let gradCheckResult = null;

const activeNet = () => (activeName === 'sgd' ? netSGD : netAdam);

// --- Teclado ---
const keysHeld = new Set();
window.addEventListener('keydown', (e) => keysHeld.add(e.code));
window.addEventListener('keyup', (e) => keysHeld.delete(e.code));

// "jump" só conta no chão (o frame em que o pulo realmente acontece);
// no ar, as setas continuam valendo. Complexidade: O(1)
function currentHumanAction() {
  const left = keysHeld.has('ArrowLeft');
  const right = keysHeld.has('ArrowRight');
  const wantsJump = keysHeld.has('Space') || keysHeld.has('ArrowUp');

  if (wantsJump && engine.player.onGround) return 'jump';
  if (left && !right) return 'left';
  if (right && !left) return 'right';
  return 'none';
}

function applyAction(action) {
  if (action === 'jump') engine.player.jump();
  else if (action === 'left') engine.player.moveLeft();
  else if (action === 'right') engine.player.moveRight();
  else engine.player.stopMoving();
}

function aiAction(state) {
  const { activations } = activeNet().forward(state);
  const out = activations[activations.length - 1].map(r => r[0]);
  return ACTIONS[out.indexOf(Math.max(...out))];
}

// --- Botões ---
btnRecord.addEventListener('click', () => {
  phase = 'expert';
  telemetry.start();
  btnRecord.classList.add('active');
  btnTrain.classList.remove('active');
  phaseLabel.textContent = 'Fase: Especialista (gravando)';
});

btnTrain.addEventListener('click', () => {
  if (telemetry.size() < 10) {
    alert('Grave (ou importe) pelo menos ~10 ações antes de treinar.');
    return;
  }
  phase = 'training';
  telemetry.stop();
  btnTrain.classList.add('active');
  btnRecord.classList.remove('active');
  phaseLabel.textContent = 'Fase: Aprendiz (SGD e Adam treinando em paralelo)';
});

btnOptimizer.addEventListener('click', () => {
  activeName = activeName === 'sgd' ? 'adam' : 'sgd';
  optimizerLabel.textContent = activeName.toUpperCase();
});

// Troca a ativação oculta: reinicia as redes e a curva, MANTÉM o dataset
btnActivation.addEventListener('click', () => {
  activation = activation === 'sigmoid' ? 'relu' : 'sigmoid';
  activationLabel.textContent = activation.toUpperCase();
  initNetworks();
  chart.reset();
  info = emptyInfo();
  gradCheckResult = null;
});

btnView.addEventListener('click', () => {
  viewMode = viewMode === 'jacobian' ? 'equations' : 'jacobian';
  viewLabel.textContent = viewMode === 'jacobian' ? 'JACOBIANA' : 'EQUAÇÕES';
  panelTitle.textContent = viewMode === 'jacobian' ? 'Jacobiana + Gradiente' : 'Equações vivas (a cada frame)';
});

btnGradCheck.addEventListener('click', () => {
  let x, y;
  if (telemetry.size() > 0) {
    const s = telemetry.dataset[Math.floor(Math.random() * telemetry.size())];
    x = s.state; y = s.action;
  } else {
    x = [Math.random() * 2 - 1, Math.random() * 2 - 1, Math.random(), Math.random()];
    y = [1, 0, 0, 0];
  }
  gradCheckResult = gradientCheck(activeNet(), x, y);
  console.log('Gradient check:', gradCheckResult);
});

// Exportar: baixa o dataset como JSON. Complexidade: O(n)
btnExport.addEventListener('click', () => {
  if (telemetry.size() === 0) {
    alert('O dataset está vazio. Grave algumas ações primeiro.');
    return;
  }
  const blob = new Blob([telemetry.toJSON()], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `dataset-${telemetry.size()}-amostras.json`;
  a.click();
  URL.revokeObjectURL(url);
});

// Importar: lê o arquivo, valida e só então substitui o dataset. Complexidade: O(n)
btnImport.addEventListener('click', () => fileImport.click());
fileImport.addEventListener('change', () => {
  const file = fileImport.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    try {
      telemetry.load(JSON.parse(reader.result));
      alert(`Dataset importado: ${telemetry.size()} amostras. Já pode clicar em Treinar IA.`);
    } catch (err) {
      alert('Arquivo inválido: ' + err.message);
    }
    fileImport.value = ''; // permite importar o mesmo arquivo de novo
  };
  reader.readAsText(file);
});

btnReset.addEventListener('click', () => {
  engine.reset();
  telemetry.clear();
  chart.reset();
  initNetworks();
  info = emptyInfo();
  gradCheckResult = null;
  phase = 'expert';
  phaseLabel.textContent = 'Fase: Especialista';
  btnRecord.classList.remove('active');
  btnTrain.classList.remove('active');
});

// --- Treino de UMA rede em UM mini-batch (gradiente médio) ---
// Também devolve uma "sonda": o peso da camada de saída com maior |gradiente|,
// antes e depois do passo, para mostrar a equação de atualização com números reais.
// Complexidade: O(B * soma n_l*n_{l+1}) + O(soma n_l*n_{l+1}) da média + O(n_saída*n_oculta) da sonda
function trainNetwork(net, opt, batch) {
  let accGradW = null, accGradB = null, loss = 0, lastDelta = null;

  for (const sample of batch) {
    const { gradW, gradB, activations } = net.backward(sample.state, sample.action);
    const yPred = activations[activations.length - 1].map(r => r[0]);
    loss += LossChart.crossEntropy(sample.action, yPred);

    if (!accGradW) {
      accGradW = gradW;
      accGradB = gradB;
    } else {
      for (let l = 0; l < gradW.length; l++) {
        for (let i = 0; i < gradW[l].length; i++) {
          for (let j = 0; j < gradW[l][i].length; j++) accGradW[l][i][j] += gradW[l][i][j];
          accGradB[l][i][0] += gradB[l][i][0];
        }
      }
    }
    lastDelta = gradB;
  }

  const n = batch.length;
  for (let l = 0; l < accGradW.length; l++) {
    for (let i = 0; i < accGradW[l].length; i++) {
      for (let j = 0; j < accGradW[l][i].length; j++) accGradW[l][i][j] /= n;
      accGradB[l][i][0] /= n;
    }
  }

  // Sonda: peso de saída com maior |gradiente| (o mais "ativo" na tela)
  const L = accGradW.length - 1;
  let bi = 0, bj = 0, bg = 0;
  for (let i = 0; i < accGradW[L].length; i++) {
    for (let j = 0; j < accGradW[L][i].length; j++) {
      if (Math.abs(accGradW[L][i][j]) > Math.abs(bg)) { bi = i; bj = j; bg = accGradW[L][i][j]; }
    }
  }
  const wBefore = net.weights[L][bi][bj];

  opt.step(net, accGradW, accGradB);

  const probe = { l: L, i: bi, j: bj, g: bg, wBefore, wAfter: net.weights[L][bi][bj] };
  if (opt.mW) { // só o Adam tem momentos; usa o t já incrementado pelo step
    probe.mHat = opt.mW[L][bi][bj] / (1 - opt.beta1 ** opt.t);
    probe.vHat = opt.vW[L][bi][bj] / (1 - opt.beta2 ** opt.t);
  }

  return { gradW: accGradW, deltas: lastDelta, loss: loss / n, probe };
}

// Complexidade: O(2 * B * soma n_l*n_{l+1})
function trainStep() {
  const batch = telemetry.sampleBalancedBatch(BATCH_SIZE);
  if (batch.length === 0) return;

  const rS = trainNetwork(netSGD, optSGD, batch);
  const rA = trainNetwork(netAdam, optAdam, batch);

  info.sgd = { deltas: rS.deltas, gradW: rS.gradW, probe: rS.probe, loss: rS.loss };
  info.adam = { deltas: rA.deltas, gradW: rA.gradW, probe: rA.probe, loss: rA.loss };
  chart.push('sgd', rS.loss);
  chart.push('adam', rA.loss);
}

// Norma L2 do gradiente por camada e total. Complexidade: O(nº de pesos)
function gradNorms(gradW) {
  const perLayer = gradW.map(layer => {
    let s = 0;
    for (const row of layer) for (const v of row) s += v * v;
    return Math.sqrt(s);
  });
  return { perLayer, total: Math.sqrt(perLayer.reduce((a, b) => a + b * b, 0)) };
}

// --- Visão 1: Jacobiana + Gradiente ---
function drawJacobianPanel(state) {
  const size = logicalSizes['canvas-matrix'];
  ctxMatrix.clearRect(0, 0, size.width, size.height);

  const out = activeNet().forward(state).activations.at(-1); // 4x1
  const J = softmaxJacobian(out);                            // 4x4

  const cell = 40, x0 = 70, y0 = 50;
  drawMatrixPanel(ctxMatrix, J, 'Jacobiana J = d(sigma)/dz', x0, y0,
    { cellSize: cell, rowLabels: ACTIONS, colLabels: ACTIONS });
  drawMatrixPanel(ctxMatrix, out, 'sigma(z)', x0 + 4 * cell + 60, y0,
    { cellSize: cell, rowLabels: ACTIONS });

  const tx = x0 + 4 * cell + 130;
  ctxMatrix.textAlign = 'left';
  ctxMatrix.font = '11px monospace';
  ctxMatrix.fillStyle = '#e2e8f0';
  ctxMatrix.fillText(`Rede: ${activeName.toUpperCase()} | oculta: ${activation.toUpperCase()}`, tx, y0);
  ctxMatrix.fillText(`eta = ${activeName === 'sgd' ? LR_SGD : LR_ADAM}`, tx, y0 + 16);

  ctxMatrix.fillStyle = '#94a3b8';
  const gW = info[activeName].gradW;
  if (gW) {
    const { perLayer, total } = gradNorms(gW);
    ctxMatrix.fillText('Gradiente (norma L2):', tx, y0 + 40);
    perLayer.forEach((v, l) => ctxMatrix.fillText(`|dL/dW${l + 1}| = ${v.toFixed(4)}`, tx, y0 + 58 + l * 16));
    ctxMatrix.fillStyle = '#f472b6';
    ctxMatrix.fillText(`|grad L| = ${total.toFixed(4)}`, tx, y0 + 58 + perLayer.length * 16 + 4);
  } else {
    ctxMatrix.fillText('Treine a IA para ver o gradiente', tx, y0 + 40);
  }

  if (gradCheckResult) {
    ctxMatrix.fillStyle = gradCheckResult.ok ? '#4ade80' : '#f87171';
    ctxMatrix.fillText(
      `Grad check: ${gradCheckResult.maxErr.toExponential(2)} ${gradCheckResult.ok ? 'OK' : 'FALHOU'}`,
      tx, y0 + 125);
    ctxMatrix.fillStyle = '#94a3b8';
    ctxMatrix.fillText(`${gradCheckResult.count} params, pior: ${gradCheckResult.where}`, tx, y0 + 141);
  }
}

// --- Visão 2: equações vivas com valores reais do último mini-batch ---
// Complexidade: O(1), número fixo de linhas
function drawEquationsPanel() {
  const size = logicalSizes['canvas-matrix'];
  ctxMatrix.clearRect(0, 0, size.width, size.height);
  ctxMatrix.textAlign = 'left';
  ctxMatrix.font = '12px monospace';

  const inf = info[activeName];
  const lr = activeName === 'sgd' ? LR_SGD : LR_ADAM;
  const WHITE = '#e2e8f0', DIM = '#94a3b8', PINK = '#f472b6', GREEN = '#4ade80';
  const lines = [];

  lines.push([`Rede ${activeName.toUpperCase()} | ocultas ${activation.toUpperCase()} | η = ${lr}`, WHITE]);

  if (!inf.probe) {
    lines.push(['', DIM]);
    lines.push(['Treine a IA para ver as equações com valores reais.', DIM]);
  } else {
    const p = inf.probe;
    const dOut = inf.deltas[inf.deltas.length - 1].map(r => r[0].toFixed(3)).join(', ');
    const nHid = Math.sqrt(inf.deltas[0].reduce((s, r) => s + r[0] * r[0], 0));

    lines.push([`Perda     L = −Σ yi·log(ŷi) = ${inf.loss.toFixed(4)}`, DIM]);
    lines.push([`Saída     δ(L) = ŷ − y = [${dOut}]`, DIM]);
    lines.push([`Oculta    δ(l) = (W(l+1))^T δ(l+1) ⊙ f'(z)   ‖δ(1)‖ = ${nHid.toFixed(4)}`, DIM]);
    lines.push([`Gradiente ∂L/∂W${p.l + 1}[${p.i}][${p.j}] = δ·a^T = ${p.g.toFixed(5)}`, GREEN]);
    lines.push(['', DIM]);

    if (p.mHat === undefined) {
      lines.push(['SGD       w ← w − η·g', PINK]);
      lines.push([`          ${p.wBefore.toFixed(5)} − ${lr}·(${p.g.toFixed(5)}) = ${p.wAfter.toFixed(5)}`, WHITE]);
    } else {
      lines.push([`m̂ = m/(1−β1^t) = ${p.mHat.toFixed(5)}     v̂ = v/(1−β2^t) = ${p.vHat.toFixed(6)}`, DIM]);
      lines.push(['Adam      w ← w − η·m̂ / (√v̂ + ε)', PINK]);
      lines.push([`          ${p.wBefore.toFixed(5)} − ${lr}·${p.mHat.toFixed(5)}/(${Math.sqrt(p.vHat).toFixed(5)} + ε) = ${p.wAfter.toFixed(5)}`, WHITE]);
    }
  }

  lines.forEach(([text, color], k) => {
    ctxMatrix.fillStyle = color;
    ctxMatrix.fillText(text, 16, 26 + k * 17);
  });
}

// --- Loop principal ---
engine.onFrame = (snapshot) => {
  const action = phase === 'expert' ? currentHumanAction() : aiAction(snapshot.state);
  applyAction(action);

  if (phase === 'expert') telemetry.record(snapshot.state, action);
  else trainStep();

  if (snapshot.gameOver) engine.reset();

  scoreValue.textContent = snapshot.score.toFixed(1);
   const c = telemetry.counts();
  datasetSize.textContent = `${telemetry.size()} · J${c[0]} L${c[1]} R${c[2]} N${c[3]}`;

  const net = activeNet();
  const sweepPhase = phase === 'training'
    ? ((performance.now() / 1000) * SWEEP_SPEED) % (net.weights.length + 1)
    : null;

  drawGame(snapshot, logicalSizes['canvas-game']);
  drawNetworkDiagram(ctxNetwork, net, info[activeName].deltas,
    logicalSizes['canvas-network'].width, logicalSizes['canvas-network'].height, sweepPhase);
  if (viewMode === 'jacobian') drawJacobianPanel(snapshot.state);
  else drawEquationsPanel();
  chart.draw(ctxChart, logicalSizes['canvas-chart'].width, logicalSizes['canvas-chart'].height);
};

function drawGame(snapshot, size) {
  ctxGame.clearRect(0, 0, size.width, size.height);
  ctxGame.fillStyle = '#63b3ed';
  const hit = snapshot.player.getHitbox();
  ctxGame.fillRect(hit.left, hit.top, hit.right - hit.left, hit.bottom - hit.top);
  ctxGame.fillStyle = '#f87171';
  for (const obs of snapshot.obstacles) ctxGame.fillRect(obs.x, obs.y, obs.width, obs.height);
}

engine.start();