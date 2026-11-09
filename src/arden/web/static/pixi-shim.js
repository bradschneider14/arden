/**
 * PixiJS v7 provider with a lightweight Canvas2D scene-graph fallback.
 *
 * When loaded in a browser with network and WebGL, `window.PIXI` from the
 * pinned CDN (`pixi.js@7.4.2`) is used directly. If offline or running in a
 * headless environment without WebGL, `PixiCanvasShim` provides a compatible
 * `Application`, `Container`, `Graphics`, `Text`, and `TextStyle` implementation
 * backed by standard HTML5 2D Canvas.
 */

function hexToRgba(hex, alpha = 1.0) {
  const num = typeof hex === "number" ? hex : parseInt(String(hex).replace(/^#/, ""), 16) || 0;
  const r = (num >> 16) & 0xff;
  const g = (num >> 8) & 0xff;
  const b = num & 0xff;
  const a = Math.max(0, Math.min(1, Number.isFinite(alpha) ? alpha : 1));
  return `rgba(${r}, ${g}, ${b}, ${a})`;
}

class PointLike {
  constructor(x = 0, y = 0, onChange = null) {
    this._x = x;
    this._y = y;
    this._onChange = onChange;
  }
  get x() {
    return this._x;
  }
  set x(v) {
    this._x = Number(v) || 0;
    if (this._onChange) this._onChange(this._x, this._y);
  }
  get y() {
    return this._y;
  }
  set y(v) {
    this._y = Number(v) || 0;
    if (this._onChange) this._onChange(this._x, this._y);
  }
  set(x = 0, y = x) {
    this._x = Number(x) || 0;
    this._y = Number(y) || 0;
    if (this._onChange) this._onChange(this._x, this._y);
  }
}

export class ShimContainer {
  constructor() {
    this.children = [];
    this.parent = null;
    this.visible = true;
    this.alpha = 1.0;
    this.rotation = 0;
    this.tint = 0xffffff;
    this.position = new PointLike(0, 0);
    this.scale = new PointLike(1, 1);
    this.pivot = new PointLike(0, 0);
  }

  get x() {
    return this.position.x;
  }
  set x(v) {
    this.position.x = v;
  }

  get y() {
    return this.position.y;
  }
  set y(v) {
    this.position.y = v;
  }

  addChild(...items) {
    for (const item of items) {
      if (!item) continue;
      if (item.parent && item.parent !== this) {
        item.parent.removeChild(item);
      }
      item.parent = this;
      if (!this.children.includes(item)) {
        this.children.push(item);
      }
    }
    return items[0];
  }

  removeChild(item) {
    const idx = this.children.indexOf(item);
    if (idx !== -1) {
      this.children.splice(idx, 1);
      item.parent = null;
    }
    return item;
  }

  removeChildren() {
    const removed = this.children.slice();
    for (const child of removed) {
      child.parent = null;
    }
    this.children.length = 0;
    return removed;
  }

  destroy() {
    this.removeChildren();
  }

  _renderCanvas(ctx) {
    if (!this.visible || this.alpha <= 0) return;
    ctx.save();
    ctx.translate(this.position.x, this.position.y);
    if (this.rotation) ctx.rotate(this.rotation);
    if (this.scale.x !== 1 || this.scale.y !== 1) {
      ctx.scale(this.scale.x, this.scale.y);
    }
    if (this.pivot.x !== 0 || this.pivot.y !== 0) {
      ctx.translate(-this.pivot.x, -this.pivot.y);
    }
    const prevAlpha = ctx.globalAlpha;
    ctx.globalAlpha = prevAlpha * Math.max(0, Math.min(1, this.alpha));

    this._drawSelf(ctx);

    for (const child of this.children) {
      if (child && typeof child._renderCanvas === "function") {
        child._renderCanvas(ctx);
      }
    }
    ctx.globalAlpha = prevAlpha;
    ctx.restore();
  }

  _drawSelf(_ctx) {}
}

export class ShimGraphics extends ShimContainer {
  constructor() {
    super();
    this._ops = [];
  }

  clear() {
    this._ops.length = 0;
    return this;
  }

  beginFill(color = 0x000000, alpha = 1.0) {
    this._ops.push({ type: "beginFill", color, alpha });
    return this;
  }

  endFill() {
    this._ops.push({ type: "endFill" });
    return this;
  }

  lineStyle(width = 0, color = 0x000000, alpha = 1.0) {
    this._ops.push({ type: "lineStyle", width, color, alpha });
    return this;
  }

  drawRect(x, y, width, height) {
    this._ops.push({ type: "rect", x, y, width, height });
    return this;
  }

  drawRoundedRect(x, y, width, height, radius = 4) {
    this._ops.push({ type: "roundedRect", x, y, width, height, radius });
    return this;
  }

  drawCircle(x, y, radius) {
    this._ops.push({ type: "circle", x, y, radius: Math.max(0.1, radius) });
    return this;
  }

  drawEllipse(x, y, halfWidth, halfHeight) {
    this._ops.push({
      type: "ellipse",
      x,
      y,
      rx: Math.max(0.1, halfWidth),
      ry: Math.max(0.1, halfHeight),
    });
    return this;
  }

  drawPolygon(points) {
    const flat = Array.isArray(points) ? points.slice() : [];
    this._ops.push({ type: "polygon", points: flat });
    return this;
  }

  moveTo(x, y) {
    this._ops.push({ type: "moveTo", x, y });
    return this;
  }

  lineTo(x, y) {
    this._ops.push({ type: "lineTo", x, y });
    return this;
  }

  _drawSelf(ctx) {
    let fillActive = false;
    let fillStyle = "rgba(0,0,0,0)";
    let lineWidth = 0;
    let strokeStyle = "rgba(0,0,0,0)";
    let hasOpenPath = false;

    const flushShape = () => {
      if (fillActive) ctx.fill();
      if (lineWidth > 0) {
        ctx.lineWidth = lineWidth;
        ctx.strokeStyle = strokeStyle;
        ctx.stroke();
      }
    };

    for (const op of this._ops) {
      switch (op.type) {
        case "beginFill":
          fillActive = op.alpha > 0;
          fillStyle = hexToRgba(op.color, op.alpha * this.alpha);
          ctx.fillStyle = fillStyle;
          break;
        case "endFill":
          if (hasOpenPath) {
            flushShape();
            hasOpenPath = false;
          }
          fillActive = false;
          break;
        case "lineStyle":
          if (hasOpenPath && lineWidth > 0) {
            ctx.lineWidth = lineWidth;
            ctx.strokeStyle = strokeStyle;
            ctx.stroke();
            hasOpenPath = false;
          }
          lineWidth = op.width || 0;
          strokeStyle = hexToRgba(op.color, (op.alpha ?? 1) * this.alpha);
          break;
        case "rect":
          ctx.beginPath();
          ctx.rect(op.x, op.y, op.width, op.height);
          flushShape();
          hasOpenPath = false;
          break;
        case "roundedRect": {
          const r = Math.min(op.radius, op.width / 2, op.height / 2);
          ctx.beginPath();
          if (typeof ctx.roundRect === "function") {
            ctx.roundRect(op.x, op.y, op.width, op.height, r);
          } else {
            ctx.rect(op.x, op.y, op.width, op.height);
          }
          flushShape();
          hasOpenPath = false;
          break;
        }
        case "circle":
          ctx.beginPath();
          ctx.arc(op.x, op.y, op.radius, 0, Math.PI * 2);
          flushShape();
          hasOpenPath = false;
          break;
        case "ellipse":
          ctx.beginPath();
          ctx.ellipse(op.x, op.y, op.rx, op.ry, 0, 0, Math.PI * 2);
          flushShape();
          hasOpenPath = false;
          break;
        case "polygon": {
          const pts = op.points;
          if (pts.length >= 4) {
            ctx.beginPath();
            ctx.moveTo(pts[0], pts[1]);
            for (let i = 2; i < pts.length; i += 2) {
              ctx.lineTo(pts[i], pts[i + 1]);
            }
            ctx.closePath();
            flushShape();
            hasOpenPath = false;
          }
          break;
        }
        case "moveTo":
          if (!hasOpenPath) {
            ctx.beginPath();
            hasOpenPath = true;
          }
          ctx.moveTo(op.x, op.y);
          break;
        case "lineTo":
          if (!hasOpenPath) {
            ctx.beginPath();
            hasOpenPath = true;
          }
          ctx.lineTo(op.x, op.y);
          if (!fillActive && lineWidth > 0) {
            ctx.lineWidth = lineWidth;
            ctx.strokeStyle = strokeStyle;
            ctx.stroke();
            hasOpenPath = false;
          }
          break;
      }
    }
  }
}

export class ShimTextStyle {
  constructor(opts = {}) {
    Object.assign(this, opts);
  }
}

export class ShimText extends ShimContainer {
  constructor(text = "", style = {}) {
    super();
    this.text = String(text ?? "");
    this.style = style instanceof ShimTextStyle ? style : new ShimTextStyle(style);
    this.anchor = new PointLike(0, 0);
  }

  _drawSelf(ctx) {
    if (!this.text) return;
    const fontSize = this.style.fontSize || 12;
    const fontFamily = this.style.fontFamily || "system-ui, -apple-system, sans-serif";
    const fontWeight = this.style.fontWeight || "normal";
    ctx.font = `${fontWeight} ${typeof fontSize === "number" ? `${fontSize}px` : fontSize} ${fontFamily}`;

    const fill = this.style.fill ?? "#ffffff";
    ctx.fillStyle = typeof fill === "number" ? hexToRgba(fill, 1.0) : String(fill);

    const align =
      this.anchor.x >= 0.75 ? "right" : this.anchor.x >= 0.35 ? "center" : "left";
    const baseline =
      this.anchor.y >= 0.75 ? "bottom" : this.anchor.y >= 0.35 ? "middle" : "top";
    ctx.textAlign = align;
    ctx.textBaseline = baseline;
    ctx.fillText(this.text, 0, 0);
  }
}

export class ShimApplication {
  constructor(options = {}) {
    const width = options.width || 800;
    const height = options.height || 600;
    this.view = options.view || document.createElement("canvas");
    this.canvas = this.view;
    this._bgColor = options.backgroundColor ?? 0x161f16;
    this._resolution = options.resolution || 1;
    this.stage = new ShimContainer();
    this.screen = { x: 0, y: 0, width, height };
    this._ctx = this.view.getContext("2d");

    const self = this;
    this.renderer = {
      width,
      height,
      resize(w, h) {
        self.screen.width = w;
        self.screen.height = h;
        self.view.width = Math.round(w * self._resolution);
        self.view.height = Math.round(h * self._resolution);
        self.view.style.width = `${w}px`;
        self.view.style.height = `${h}px`;
      },
    };
    this.renderer.resize(width, height);

    this._callbacks = new Set();
    this._rafId = null;
    this._lastTime = performance.now();

    this.ticker = {
      add(fn) {
        self._callbacks.add(fn);
      },
      remove(fn) {
        self._callbacks.delete(fn);
      },
    };

    const step = (now) => {
      const dt = Math.min(4, Math.max(0.1, (now - self._lastTime) / (1000 / 60)));
      self._lastTime = now;
      for (const cb of self._callbacks) {
        try {
          cb(dt);
        } catch (_err) {
          // Ignore ticker callback errors to keep render loop alive
        }
      }
      self.render();
      self._rafId = requestAnimationFrame(step);
    };
    this._rafId = requestAnimationFrame(step);
  }

  render() {
    if (!this._ctx) return;
    const ctx = this._ctx;
    ctx.save();
    ctx.setTransform(this._resolution, 0, 0, this._resolution, 0, 0);
    ctx.fillStyle = hexToRgba(this._bgColor, 1.0);
    ctx.fillRect(0, 0, this.screen.width, this.screen.height);
    this.stage._renderCanvas(ctx);
    ctx.restore();
  }

  destroy() {
    if (this._rafId !== null) {
      cancelAnimationFrame(this._rafId);
      this._rafId = null;
    }
    this._callbacks.clear();
    this.stage.destroy();
  }
}

export const PixiCanvasShim = {
  Application: ShimApplication,
  Container: ShimContainer,
  Graphics: ShimGraphics,
  Text: ShimText,
  TextStyle: ShimTextStyle,
  isShim: true,
};

export function getPixi() {
  if (typeof globalThis !== "undefined" && globalThis.PIXI && globalThis.PIXI.Application) {
    return globalThis.PIXI;
  }
  if (typeof globalThis !== "undefined") {
    globalThis.PIXI = PixiCanvasShim;
  }
  return PixiCanvasShim;
}
