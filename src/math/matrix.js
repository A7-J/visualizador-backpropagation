// math/matrix.js
// Matrizes representadas como Array<Array<number>> (row-major).
// Vetores como Array<number> (tratados como coluna quando multiplicados).

export function zeros(rows, cols) {
  return Array.from({ length: rows }, () => new Array(cols).fill(0));
}

// Multiplicação matricial A(m×n) · B(n×p) = C(m×p)
// Complexidade: O(m·n·p)
export function dot(A, B) {
  const m = A.length, n = A[0].length, p = B[0].length;
  if (B.length !== n) throw new Error(`dot: dimensões incompatíveis (${m}x${n}) · (${B.length}x${p})`);
  const C = zeros(m, p);
  for (let i = 0; i < m; i++) {
    for (let k = 0; k < n; k++) {
      const a = A[i][k];
      if (a === 0) continue; // pequena poda, não muda o pior caso
      for (let j = 0; j < p; j++) {
        C[i][j] += a * B[k][j];
      }
    }
  }
  return C;
}

// Transposta A(m×n) -> A^T(n×m). Complexidade: O(m·n)
export function transpose(A) {
  const m = A.length, n = A[0].length;
  const T = zeros(n, m);
  for (let i = 0; i < m; i++)
    for (let j = 0; j < n; j++)
      T[j][i] = A[i][j];
  return T;
}

// Soma elemento a elemento. O(m·n)
export function add(A, B) {
  return A.map((row, i) => row.map((v, j) => v + B[i][j]));
}

// Produto de Hadamard (elemento a elemento) — usado em δ = ∇L ⊙ σ'(z). O(m·n)
export function hadamard(A, B) {
  return A.map((row, i) => row.map((v, j) => v * B[i][j]));
}

// Escala por escalar. O(m·n)
export function scale(A, k) {
  return A.map(row => row.map(v => v * k));
}

// Vetor coluna [n] -> matriz n×1, útil para uniformizar operações com dot()
export function toColumn(vec) {
  return vec.map(v => [v]);
}

export function toFlat(mat) {
  return mat.map(row => row[0]);
}