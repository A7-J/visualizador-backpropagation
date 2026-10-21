// src/game/engine.js
import { Player } from './player.js';

const OBSTACLE_SPEED = 250;
const SPAWN_INTERVAL = 1.4;
const OBSTACLE_WIDTH = 30;
const OBSTACLE_HEIGHT = 40;

export class GameEngine {
  constructor(canvasWidth, canvasHeight) {
    this.canvasWidth = canvasWidth;
    this.canvasHeight = canvasHeight;
    this.groundY = canvasHeight - 20;

    this.player = new Player(80, this.groundY);
    this.obstacles = [];
    this.spawnTimer = 0;
    this.score = 0;
    this.gameOver = false;

    this.lastTimestamp = null;
    this.onFrame = null;
  }

  start() {
    this.lastTimestamp = null;
    requestAnimationFrame(this._loop.bind(this));
  }

  _loop(timestamp) {
    if (this.lastTimestamp === null) this.lastTimestamp = timestamp;
    let dt = (timestamp - this.lastTimestamp) / 1000;
    dt = Math.min(dt, 0.05);
    this.lastTimestamp = timestamp;

    if (!this.gameOver) this.update(dt);

    if (this.onFrame) this.onFrame(this.getSnapshot());
    requestAnimationFrame(this._loop.bind(this));
  }

  // Complexidade: O(n), n = obstáculos ativos
  update(dt) {
    this.player.update(dt, 0, this.canvasWidth - this.player.width);

    this.spawnTimer += dt;
    if (this.spawnTimer >= SPAWN_INTERVAL) {
      this.spawnTimer = 0;
      this._spawnObstacle();
    }

    for (const obs of this.obstacles) obs.x -= OBSTACLE_SPEED * dt;
    this.obstacles = this.obstacles.filter(o => o.x + o.width > 0);

    if (this._checkCollision()) {
      this.gameOver = true;
      return;
    }
    this.score += dt;
  }

  _spawnObstacle() {
    this.obstacles.push({
      x: this.canvasWidth,
      y: this.groundY - OBSTACLE_HEIGHT,
      width: OBSTACLE_WIDTH,
      height: OBSTACLE_HEIGHT,
    });
  }

  // AABB, O(n)
  _checkCollision() {
    const hit = this.player.getHitbox();
    for (const obs of this.obstacles) {
      const overlapX = hit.left < obs.x + obs.width && hit.right > obs.x;
      const overlapY = hit.bottom > obs.y && hit.top < obs.y + obs.height;
      if (overlapX && overlapY) return true;
    }
    return false;
  }

  // O(n)
  getNearestObstacle() {
    let nearest = null;
    let minDist = Infinity;
    for (const obs of this.obstacles) {
      const dist = obs.x - this.player.x;
      if (dist >= 0 && dist < minDist) {
        minDist = dist;
        nearest = obs;
      }
    }
    return nearest;
  }

  getSnapshot() {
    const nearestObstacle = this.getNearestObstacle();
    return {
      player: this.player,
      obstacles: this.obstacles,
      state: this.player.getState(nearestObstacle, this.canvasWidth),
      score: this.score,
      gameOver: this.gameOver,
    };
  }

  reset() {
    this.player = new Player(80, this.groundY);
    this.obstacles = [];
    this.spawnTimer = 0;
    this.score = 0;
    this.gameOver = false;
  }
}