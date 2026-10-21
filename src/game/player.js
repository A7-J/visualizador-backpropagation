// src/game/player.js
const GRAVITY = 1800;       // px/s²
const JUMP_FORCE = -650;    // px/s
const MOVE_SPEED = 300;     // px/s

export class Player {
  constructor(x = 80, groundY = 320) {
    this.x = x;
    this.groundY = groundY;
    this.y = groundY;
    this.vx = 0;
    this.vy = 0;
    this.width = 24;
    this.height = 32;
    this.onGround = true;
  }

  jump() {
    if (!this.onGround) return;
    this.vy = JUMP_FORCE;
    this.onGround = false;
  }

  moveLeft()  { this.vx = -MOVE_SPEED; }
  moveRight() { this.vx = MOVE_SPEED; }
  stopMoving() { this.vx = 0; }

  // Complexidade: O(1)
  update(dt, minX = 0, maxX = Infinity) {
    this.vy += GRAVITY * dt;
    this.x += this.vx * dt;
    this.y += this.vy * dt;

    if (this.x < minX) { this.x = minX; this.vx = 0; }
    if (this.x > maxX) { this.x = maxX; this.vx = 0; }

    if (this.y >= this.groundY) {
      this.y = this.groundY;
      this.vy = 0;
      this.onGround = true;
    }
  }

  getHitbox() {
    return { left: this.x, right: this.x + this.width, top: this.y - this.height, bottom: this.y };
  }

  // Estado com 4 entradas, todas normalizadas:
  // [altura, velocidade vertical, posição horizontal, distância ao obstáculo]
  // A posição horizontal é a peça que faltava: sem ela, a rede não sabia
  // se estava na parede, nem como interpretar a distância ao obstáculo.
  // Complexidade: O(1)
  getState(nearestObstacle, canvasWidth) {
    const normY = (this.y - this.groundY) / this.groundY;
    const normVy = Math.max(-1, Math.min(1, this.vy / 1000));
    const normX = Math.max(0, Math.min(1, this.x / canvasWidth));
    const obstacleDist = nearestObstacle
      ? Math.max(0, Math.min(1, (nearestObstacle.x - this.x) / canvasWidth))
      : 1;

    return [normY, normVy, normX, obstacleDist];
  }
}