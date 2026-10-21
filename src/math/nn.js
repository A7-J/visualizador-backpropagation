// src/math/nn.js
import { zeros, dot, transpose, add, hadamard, toColumn } from './matrix.js';
import { sigmoid, sigmoidDerivative, relu, reluDerivative, softmax } from './activations.js';

// Ativação das camadas ocultas (a de saída é sempre softmax)
const ACT = {
  sigmoid: { f: sigmoid, df: sigmoidDerivative },
  relu:    { f: relu,    df: reluDerivative },
};

export class NeuralNetwork {
  constructor(layerSizes, { hidden = 'sigmoid' } = {}) {
    this.layerSizes = layerSizes;
    this.hidden = hidden;
    this.weights = [];
    this.biases = [];

    for (let l = 0; l < layerSizes.length - 1; l++) {
      const rows = layerSizes[l + 1], cols = layerSizes[l];
      // ReLU: He uniform (+-sqrt(6/entradas)) evita neurônios mortos/explosão; sigmoid: +-0.5
      const scale = hidden === 'relu' ? Math.sqrt(6 / cols) : 0.5;
      this.weights.push(
        Array.from({ length: rows }, () =>
          Array.from({ length: cols }, () => (Math.random() * 2 - 1) * scale)
        )
      );
      this.biases.push(zeros(rows, 1));
    }
  }

  // Cópia profunda: mesmos pesos iniciais e mesma ativação. O(P)
  clone() {
    const c = new NeuralNetwork(this.layerSizes, { hidden: this.hidden });
    c.weights = this.weights.map(L => L.map(r => [...r]));
    c.biases = this.biases.map(L => L.map(r => [...r]));
    return c;
  }

  // Complexidade: O(soma n_l * n_{l+1})
  forward(xVec) {
    const act = ACT[this.hidden];
    const activations = [toColumn(xVec)];
    const zs = [];

    for (let l = 0; l < this.weights.length; l++) {
      const z = add(dot(this.weights[l], activations[l]), this.biases[l]);
      zs.push(z);
      const isLast = l === this.weights.length - 1;
      activations.push(isLast ? softmax(z) : act.f(z));
    }
    return { activations, zs };
  }

  // Complexidade: O(soma n_l * n_{l+1})
  backward(xVec, yTarget) {
    const act = ACT[this.hidden];
    const { activations, zs } = this.forward(xVec);
    const L = this.weights.length - 1;
    const yCol = toColumn(yTarget);

    const gradW = new Array(this.weights.length);
    const gradB = new Array(this.weights.length);
    const deltas = new Array(this.weights.length);

    // delta(L) = y_hat - y  (softmax + cross-entropy)
    deltas[L] = activations[L + 1].map((row, i) => [row[0] - yCol[i][0]]);
    gradW[L] = dot(deltas[L], transpose(activations[L]));
    gradB[L] = deltas[L];

    // delta(l) = (W(l+1))^T delta(l+1) hadamard f'(z(l))
    for (let l = L - 1; l >= 0; l--) {
      const propagated = dot(transpose(this.weights[l + 1]), deltas[l + 1]);
      deltas[l] = hadamard(propagated, act.df(zs[l]));
      gradW[l] = dot(deltas[l], transpose(activations[l]));
      gradB[l] = deltas[l];
    }

    return { gradW, gradB, activations };
  }
}