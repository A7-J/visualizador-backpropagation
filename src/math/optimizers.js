// src/math/optimizers.js
// Ambos recebem a rede (para acessar/atualizar weights e biases) e os
// gradientes calculados por backward(). Mantêm estado próprio entre chamadas
// (Adam precisa "lembrar" dos momentos m e v de uma iteração pra outra).

export class SGD {
  constructor(learningRate = 0.1) {
    this.lr = learningRate;
  }

  // w_{t+1} = w_t - η∇L(w_t)  — sem memória, olha só o gradiente atual
  // Complexidade: O(Σ nₗ·nₗ₊₁) — percorre cada peso/bias uma vez
  step(network, gradW, gradB) {
    for (let l = 0; l < network.weights.length; l++) {
      for (let i = 0; i < network.weights[l].length; i++) {
        for (let j = 0; j < network.weights[l][i].length; j++) {
          network.weights[l][i][j] -= this.lr * gradW[l][i][j];
        }
        network.biases[l][i][0] -= this.lr * gradB[l][i][0];
      }
    }
  }
}

export class Adam {
  constructor(network, learningRate = 0.01, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8) {
    this.lr = learningRate;
    this.beta1 = beta1;
    this.beta2 = beta2;
    this.epsilon = epsilon;
    this.t = 0; // contador de iterações, usado no bias-correction

    // Momentos m (média móvel do gradiente) e v (média móvel do gradiente²)
    // — mesma forma que weights/biases, inicializados em zero
    this.mW = network.weights.map(layer => layer.map(row => row.map(() => 0)));
    this.vW = network.weights.map(layer => layer.map(row => row.map(() => 0)));
    this.mB = network.biases.map(layer => layer.map(() => [0]));
    this.vB = network.biases.map(layer => layer.map(() => [0]));
  }

  // Complexidade: O(Σ nₗ·nₗ₊₁), igual ao SGD — mais estado, mas mesma ordem de tempo
  step(network, gradW, gradB) {
    this.t += 1;
    const { beta1, beta2, epsilon, lr, t } = this;
    const biasCorr1 = 1 - beta1 ** t;
    const biasCorr2 = 1 - beta2 ** t;

    for (let l = 0; l < network.weights.length; l++) {
      for (let i = 0; i < network.weights[l].length; i++) {
        for (let j = 0; j < network.weights[l][i].length; j++) {
          const g = gradW[l][i][j];

          // m_t = β1·m_{t-1} + (1-β1)·g   (média móvel do gradiente — "pra onde venho indo")
          this.mW[l][i][j] = beta1 * this.mW[l][i][j] + (1 - beta1) * g;
          // v_t = β2·v_{t-1} + (1-β2)·g²  (média móvel do gradiente² — "quão irregular é o terreno")
          this.vW[l][i][j] = beta2 * this.vW[l][i][j] + (1 - beta2) * g * g;

          const mHat = this.mW[l][i][j] / biasCorr1;
          const vHat = this.vW[l][i][j] / biasCorr2;

          network.weights[l][i][j] -= lr * mHat / (Math.sqrt(vHat) + epsilon);
        }

        const gB = gradB[l][i][0];
        this.mB[l][i][0] = beta1 * this.mB[l][i][0] + (1 - beta1) * gB;
        this.vB[l][i][0] = beta2 * this.vB[l][i][0] + (1 - beta2) * gB * gB;

        const mHatB = this.mB[l][i][0] / biasCorr1;
        const vHatB = this.vB[l][i][0] / biasCorr2;

        network.biases[l][i][0] -= lr * mHatB / (Math.sqrt(vHatB) + epsilon);
      }
    }
  }
}