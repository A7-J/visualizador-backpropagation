// src/math/gradcheck.js
// Verificação numérica do gradiente: compara dL/dw do backward() (analítico)
// com (L(w+h) - L(w-h)) / 2h (diferença central, erro O(h^2)).

function loss(net, x, y) {
  const out = net.forward(x).activations.at(-1);
  let l = 0;
  for (let i = 0; i < y.length; i++) l -= y[i] * Math.log(Math.max(out[i][0], 1e-12));
  return l;
}

function relErr(a, n) {
  return Math.abs(a - n) / Math.max(Math.abs(a) + Math.abs(n), 1e-6);
}

// Complexidade: O(P * soma n_l*n_{l+1}), P = nº de parâmetros (~O(P^2)). Só roda ao clicar.
export function gradientCheck(net, x, y, h = 1e-5) {
  const { gradW, gradB } = net.backward(x, y);
  let maxErr = 0, where = '', count = 0;

  const test = (params, analytic, l, i, j, tag) => {
    const orig = params[i][j];
    params[i][j] = orig + h; const lp = loss(net, x, y);
    params[i][j] = orig - h; const lm = loss(net, x, y);
    params[i][j] = orig; // restaura SEMPRE
    const err = relErr(analytic, (lp - lm) / (2 * h));
    count++;
    if (err > maxErr) { maxErr = err; where = `${tag}${l + 1}[${i}][${j}]`; }
  };

  for (let l = 0; l < net.weights.length; l++) {
    for (let i = 0; i < net.weights[l].length; i++) {
      for (let j = 0; j < net.weights[l][i].length; j++) test(net.weights[l], gradW[l][i][j], l, i, j, 'W');
      test(net.biases[l], gradB[l][i][0], l, i, 0, 'b');
    }
  }
  return { maxErr, where, count, ok: maxErr < 1e-5 };
}
