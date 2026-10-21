// src/ui/visualizer.js
const NEURON_RADIUS = 12;

// Complexidade: O(N)
function computeLayout(layerSizes, width, height) {
  const positions = [];
  const layerGapX = width / (layerSizes.length + 1);

  layerSizes.forEach((size, l) => {
    const x = layerGapX * (l + 1);
    const gapY = height / (size + 1);
    const layerPositions = [];
    for (let i = 0; i < size; i++) layerPositions.push({ x, y: gapY * (i + 1) });
    positions.push(layerPositions);
  });
  return positions;
}

// sweepPhase: posição da onda reversa em [0, L+1), ou null (sem animação).
// A camada de pesos mais próxima da saída (l = L-1) acende primeiro (waveIdx = 0).
// Intensidade proporcional a |delta|, normalizada pelo máximo GLOBAL
// (assim o vanishing gradient aparece como camadas mais fracas).
// Complexidade: O(N + E)
export function drawNetworkDiagram(ctx, network, deltas, width, height, sweepPhase = null) {
  ctx.clearRect(0, 0, width, height);
  const positions = computeLayout(network.layerSizes, width, height);
  const L = network.weights.length;

  let globalMax = 1e-6;
  if (deltas) {
    for (const d of deltas) for (const row of d) globalMax = Math.max(globalMax, Math.abs(row[0]));
  }

  for (let l = 0; l < L; l++) {
    const W = network.weights[l];
    const fromLayer = positions[l];
    const toLayer = positions[l + 1];

    // Pulso triangular da onda nesta camada: 1 quando a onda está sobre ela, 0 a 1 unidade de distância
    const waveIdx = L - 1 - l;
    const pulse = sweepPhase === null ? 1 : Math.max(0, 1 - Math.abs(sweepPhase - waveIdx));
    const boost = sweepPhase === null ? 1 : 0.25 + 0.75 * pulse;

    for (let i = 0; i < W.length; i++) {
      for (let j = 0; j < W[i].length; j++) {
        const from = fromLayer[j];
        const to = toLayer[i];
        const weight = W[i][j];

        let alpha = Math.min(1, Math.abs(weight) / 2);
        let color = '99, 179, 237';

        if (deltas) {
          const intensity = Math.abs(deltas[l][i][0]) / globalMax;
          alpha = Math.min(1, 0.08 + 0.92 * intensity * boost);
          color = '244, 114, 182';
        }

        ctx.strokeStyle = `rgba(${color}, ${alpha})`;
        ctx.lineWidth = weight >= 0 ? 1.5 : 1;
        ctx.beginPath();
        ctx.moveTo(from.x, from.y);
        ctx.lineTo(to.x, to.y);
        ctx.stroke();
      }
    }
  }

  positions.forEach(layer => {
    layer.forEach(({ x, y }) => {
      ctx.beginPath();
      ctx.arc(x, y, NEURON_RADIUS, 0, Math.PI * 2);
      ctx.fillStyle = '#1e293b';
      ctx.fill();
      ctx.strokeStyle = '#94a3b8';
      ctx.stroke();
    });
  });
}

// Painel de matriz com rótulos opcionais.
// opts: { cellSize, decimals, rowLabels, colLabels }
// Complexidade: O(linhas * colunas)
export function drawMatrixPanel(ctx, matrix, label, x, y, opts = {}) {
  const { cellSize = 38, decimals = 3, rowLabels = null, colLabels = null } = opts;

  ctx.textAlign = 'left';
  ctx.font = '11px monospace';
  ctx.fillStyle = '#e2e8f0';
  ctx.fillText(label, x, y - (colLabels ? 22 : 8));

  if (colLabels) {
    ctx.font = '9px monospace';
    ctx.fillStyle = '#94a3b8';
    colLabels.forEach((name, j) => ctx.fillText(name, x + j * cellSize + 3, y - 6));
  }

  for (let i = 0; i < matrix.length; i++) {
    if (rowLabels) {
      ctx.font = '9px monospace';
      ctx.fillStyle = '#94a3b8';
      ctx.textAlign = 'right';
      ctx.fillText(rowLabels[i], x - 6, y + i * cellSize + cellSize / 2 + 3);
      ctx.textAlign = 'left';
    }
    for (let j = 0; j < matrix[i].length; j++) {
      const cx = x + j * cellSize;
      const cy = y + i * cellSize;
      ctx.strokeStyle = '#334155';
      ctx.strokeRect(cx, cy, cellSize, cellSize);

      const value = matrix[i][j];
      ctx.fillStyle = value >= 0 ? '#4ade80' : '#f87171';
      ctx.font = '9px monospace';
      ctx.fillText(value.toFixed(decimals), cx + 3, cy + cellSize / 2 + 3);
    }
  }
}