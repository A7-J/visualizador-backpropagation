// math/activations.js
export function sigmoid(z) {
  return z.map(row => row.map(x => 1 / (1 + Math.exp(-x))));
}

// σ'(z) = σ(z)·(1-σ(z)) — derivada local, O(m·n)
export function sigmoidDerivative(z) {
  const s = sigmoid(z);
  return s.map(row => row.map(x => x * (1 - x)));
}

export function relu(z) {
  return z.map(row => row.map(x => Math.max(0, x)));
}

export function reluDerivative(z) {
  return z.map(row => row.map(x => (x > 0 ? 1 : 0)));
}

// Softmax σ: R^n -> (0,1)^n, aplicado a um vetor-coluna (matriz n×1)
// Subtrai o máximo antes do exp por estabilidade numérica (evita overflow)
export function softmax(zCol) {
  const flat = zCol.map(row => row[0]);
  const maxZ = Math.max(...flat);
  const exps = flat.map(z => Math.exp(z - maxZ));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => [e / sum]);
}

// Matriz Jacobiana do softmax: J_ij = σ_i(δ_ij - σ_j)
// Complexidade: O(n²) — é uma matriz n×n, diferente das outras ativações
// (que são "diagonais", elemento a elemento)
export function softmaxJacobian(softmaxOut) {
  const s = softmaxOut.map(row => row[0]); // achata pra facilitar
  const n = s.length;
  const J = [];
  for (let i = 0; i < n; i++) {
    J.push([]);
    for (let j = 0; j < n; j++) {
      const deltaIJ = i === j ? 1 : 0;
      J[i].push(s[i] * (deltaIJ - s[j]));
    }
  }
  return J;
}