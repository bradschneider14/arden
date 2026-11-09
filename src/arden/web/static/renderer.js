/**
 * PixiJS top-down tile and object renderer for the Arden live world.
 */

import { getPixi, PixiCanvasShim } from "./pixi-shim.js";
import {
  DEFAULT_TILE_SIZE,
  DEFAULT_WORLD_HEIGHT,
  DEFAULT_WORLD_WIDTH,
  computeObjectElevation,
  computeObjectRadiusMeters,
  isHiddenInsideClosedContainer,
  resolveObjectPosition,
} from "./state.js";

export const BASE_PIXELS_PER_METER = 44;
export const MIN_ZOOM = 0.35;
export const MAX_ZOOM = 4.0;

const COLOR_MAP = {
  red: 0xd94848,
  brown: 0x8b5e3c,
  green: 0x3fa34d,
  gray: 0x8d959e,
  grey: 0x8d959e,
  blue: 0x3b82f6,
  white: 0xf3f4f6,
  yellow: 0xeab308,
  orange: 0xf97316,
  purple: 0xa855f7,
  black: 0x27272a,
  tan: 0xc2a878,
  gold: 0xd4af37,
};

/**
 * Resolve a color name or hex string into a 24-bit integer color.
 */
export function resolveColorHex(colorName, fallback = 0xcbd5e1) {
  if (typeof colorName === "number" && Number.isFinite(colorName)) {
    return colorName & 0xffffff;
  }
  if (!colorName || typeof colorName !== "string") {
    return fallback;
  }
  const key = colorName.trim().toLowerCase();
  if (key in COLOR_MAP) {
    return COLOR_MAP[key];
  }
  if (key.startsWith("#")) {
    const parsed = parseInt(key.slice(1), 16);
    if (Number.isFinite(parsed)) return parsed;
  }
  return fallback;
}

/**
 * Adjust a 24-bit RGB integer's brightness by `factor` (e.g. 0.7 darker, 1.25 lighter).
 */
export function shadeColor(hex, factor = 1.0) {
  const r = Math.max(0, Math.min(255, Math.round(((hex >> 16) & 0xff) * factor)));
  const g = Math.max(0, Math.min(255, Math.round(((hex >> 8) & 0xff) * factor)));
  const b = Math.max(0, Math.min(255, Math.round((hex & 0xff) * factor)));
  return (r << 16) | (g << 8) | b;
}

export class WorldRenderer {
  constructor(canvasElement, store, options = {}) {
    this.canvas = canvasElement;
    this.store = store;
    this.ppm = options.pixelsPerMeter || BASE_PIXELS_PER_METER;
    this.zoom = 1.0;
    this.panX = 0;
    this.panY = 0;
    this._hasCenteredInitially = false;
    this._animTime = 0;

    const width = canvasElement.clientWidth || options.width || 900;
    const height = canvasElement.clientHeight || options.height || 680;

    let PIXI = getPixi();
    let app;
    try {
      app = new PIXI.Application({
        view: canvasElement,
        width,
        height,
        backgroundColor: 0x161f16,
        antialias: true,
        resolution: (typeof window !== "undefined" && window.devicePixelRatio) || 1,
        autoDensity: true,
      });
    } catch (_webglErr) {
      // Fallback to Canvas2D PixiShim when WebGL is unavailable (e.g. headless browser)
      PIXI = PixiCanvasShim;
      app = new PIXI.Application({
        view: canvasElement,
        width,
        height,
        backgroundColor: 0x161f16,
        resolution: 1,
      });
    }

    this.PIXI = PIXI;
    this.app = app;

    // Scene graph layers
    this.worldContainer = new PIXI.Container();
    this.terrainGraphics = new PIXI.Graphics();
    this.shadowsGraphics = new PIXI.Graphics();
    this.objectsGraphics = new PIXI.Graphics();
    this.agentGraphics = new PIXI.Graphics();
    this.selectionGraphics = new PIXI.Graphics();

    this.worldContainer.addChild(
      this.terrainGraphics,
      this.shadowsGraphics,
      this.objectsGraphics,
      this.agentGraphics,
      this.selectionGraphics
    );

    // Screen-space overlays (night tint & weather particles/indicators)
    this.overlayContainer = new PIXI.Container();
    this.nightOverlayGraphics = new PIXI.Graphics();
    this.weatherOverlayGraphics = new PIXI.Graphics();
    this.windIndicatorGraphics = new PIXI.Graphics();
    this.overlayContainer.addChild(
      this.nightOverlayGraphics,
      this.weatherOverlayGraphics,
      this.windIndicatorGraphics
    );

    this.app.stage.addChild(this.worldContainer, this.overlayContainer);

    // Pre-seeded deterministic particle offsets for smooth rain/wind/cold animation
    this._particles = Array.from({ length: 75 }, (_, i) => ({
      u: ((i * 73 + 19) % 100) / 100,
      v: ((i * 41 + 7) % 100) / 100,
      speed: 0.75 + ((i * 29) % 50) / 100,
      len: 10 + (i % 12),
    }));

    this._onTick = (dt = 1) => {
      this._animTime += 0.025 * (typeof dt === "number" ? dt : 1);
      this.drawWeatherAndNightOverlay();
    };
    if (this.app.ticker && typeof this.app.ticker.add === "function") {
      this.app.ticker.add(this._onTick);
    }

    this.centerViewOnWorld();
    this.renderWorld();
  }

  /**
   * Resize the PixiJS renderer to match its container element dimensions.
   */
  resize(width, height) {
    const w = Math.max(200, Math.round(width || this.canvas.clientWidth || 800));
    const h = Math.max(200, Math.round(height || this.canvas.clientHeight || 600));
    if (this.app.renderer && typeof this.app.renderer.resize === "function") {
      this.app.renderer.resize(w, h);
    }
    if (!this._hasCenteredInitially) {
      this.centerViewOnWorld();
    }
    this.renderWorld();
  }

  /**
   * Center and fit the world grid inside the current viewport.
   */
  centerViewOnWorld() {
    const state = this.store.getState();
    const gridW = (state.terrain?.width || DEFAULT_WORLD_WIDTH) * (state.terrain?.tile_size || 1);
    const gridH = (state.terrain?.height || DEFAULT_WORLD_HEIGHT) * (state.terrain?.tile_size || 1);
    const worldPxW = gridW * this.ppm;
    const worldPxH = gridH * this.ppm;

    const screenW = this.app.screen?.width || this.canvas.clientWidth || 900;
    const screenH = this.app.screen?.height || this.canvas.clientHeight || 680;

    const fitScale = Math.min((screenW - 80) / worldPxW, (screenH - 80) / worldPxH);
    this.zoom = Math.max(MIN_ZOOM, Math.min(1.35, Number.isFinite(fitScale) && fitScale > 0 ? fitScale : 1.0));
    this.panX = Math.round((screenW - worldPxW * this.zoom) * 0.5);
    this.panY = Math.round((screenH - worldPxH * this.zoom) * 0.5);
    this._hasCenteredInitially = true;
    this._applyTransform();
  }

  resetView() {
    this.centerViewOnWorld();
    this.renderWorld();
  }

  _applyTransform() {
    this.worldContainer.position.set(this.panX, this.panY);
    this.worldContainer.scale.set(this.zoom, this.zoom);
  }

  /**
   * Pan the viewport by `(dx, dy)` screen pixels.
   */
  panBy(dx, dy) {
    this.panX += Number(dx) || 0;
    this.panY += Number(dy) || 0;
    this._applyTransform();
    if (this.PIXI.isShim) this.app.render();
  }

  /**
   * Zoom by `factor` anchored at screen pixel coordinates `(screenX, screenY)`.
   */
  zoomAt(factor, screenX, screenY) {
    const prevZoom = this.zoom;
    const nextZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, prevZoom * factor));
    if (Math.abs(nextZoom - prevZoom) < 1e-4) return;

    const sx = screenX ?? (this.app.screen?.width || 800) * 0.5;
    const sy = screenY ?? (this.app.screen?.height || 600) * 0.5;

    // Keep the world point under (sx, sy) stationary
    const worldX = (sx - this.panX) / prevZoom;
    const worldY = (sy - this.panY) / prevZoom;

    this.zoom = nextZoom;
    this.panX = sx - worldX * nextZoom;
    this.panY = sy - worldY * nextZoom;
    this._applyTransform();
    this.renderWorld();
  }

  /**
   * Convert screen pixel coordinates `(sx, sy)` relative to the canvas into world meters `(wx, wy)`.
   */
  screenToWorld(sx, sy) {
    const localPxX = (sx - this.panX) / this.zoom;
    const localPxY = (sy - this.panY) / this.zoom;
    return {
      x: localPxX / this.ppm,
      y: localPxY / this.ppm,
    };
  }

  /**
   * Convert world coordinates `(wx, wy)` in meters into screen pixel coordinates `(sx, sy)`.
   */
  worldToScreen(wx, wy) {
    const localPxX = wx * this.ppm;
    const localPxY = wy * this.ppm;
    return {
      x: this.panX + localPxX * this.zoom,
      y: this.panY + localPxY * this.zoom,
    };
  }

  /**
   * Full redraw of the world layers (terrain, shadows, objects, agent, selection, and overlays).
   */
  renderWorld() {
    this._applyTransform();
    this.drawTerrain();
    this.drawObjectsAndShadows();
    this.drawAgentAndHeldItem();
    this.drawSelectionAndHover();
    this.drawWeatherAndNightOverlay();
    if (this.PIXI.isShim) {
      this.app.render();
    }
  }

  /**
   * Draw the 2D terrain tile grid (`grass`, `water` with flow chevrons, `ledge` with raised bevel).
   */
  drawTerrain() {
    const g = this.terrainGraphics;
    g.clear();

    const state = this.store.getState();
    const terrain = state.terrain || {};
    const width = Number(terrain.width) || DEFAULT_WORLD_WIDTH;
    const height = Number(terrain.height) || DEFAULT_WORLD_HEIGHT;
    const tileSize = Number(terrain.tile_size) || DEFAULT_TILE_SIZE;
    const defaultType = terrain.default_type || "grass";
    const tiles = terrain.tiles || {};
    const tilePx = tileSize * this.ppm;

    // Outer world frame shadow
    g.beginFill(0x0d130d, 0.85);
    g.drawRoundedRect(-8, -8, width * tilePx + 16, height * tilePx + 16, 8);
    g.endFill();

    for (let tx = 0; tx < width; tx++) {
      for (let ty = 0; ty < height; ty++) {
        const key = `${tx},${ty}`;
        const tileEntry = tiles[key];
        const tileType =
          typeof tileEntry === "string"
            ? tileEntry
            : tileEntry?.type || defaultType;
        const px = tx * tilePx;
        const py = ty * tilePx;

        if (tileType === "water") {
          const baseWater = (tx + ty) % 2 === 0 ? 0x256394 : 0x296ca0;
          g.lineStyle(1, 0x1e4e78, 0.65);
          g.beginFill(baseWater, 1.0);
          g.drawRect(px, py, tilePx, tilePx);
          g.endFill();

          // Draw flow direction indicator chevrons inside water tile
          const flowX = Number(tileEntry?.flow?.x ?? 1.0);
          const flowY = Number(tileEntry?.flow?.y ?? 0.0);
          const flowLen = Math.hypot(flowX, flowY);
          if (flowLen > 0.01) {
            const ux = flowX / flowLen;
            const uy = flowY / flowLen;
            const cx = px + tilePx * 0.5;
            const cy = py + tilePx * 0.5;
            const tipX = cx + ux * (tilePx * 0.22);
            const tipY = cy + uy * (tilePx * 0.22);
            const tailX = cx - ux * (tilePx * 0.14);
            const tailY = cy - uy * (tilePx * 0.14);
            const perpX = -uy * (tilePx * 0.14);
            const perpY = ux * (tilePx * 0.14);

            g.lineStyle(1.8, 0x7dd3fc, 0.7);
            g.moveTo(tailX + perpX, tailY + perpY);
            g.lineTo(tipX, tipY);
            g.lineTo(tailX - perpX, tailY - perpY);
          }
        } else if (tileType === "ledge") {
          // Raised rock ledge (elevation = 1.0m): draw drop shadow + raised bevel stone tile
          g.lineStyle(0);
          g.beginFill(0x141814, 0.45);
          g.drawRect(px + 4, py + 5, tilePx, tilePx);
          g.endFill();

          const rockColor = (tx + ty) % 2 === 0 ? 0x78716c : 0x827b75;
          g.lineStyle(2, 0x44403c, 0.95);
          g.beginFill(rockColor, 1.0);
          g.drawRect(px, py, tilePx, tilePx);
          g.endFill();

          // Top/left highlight edge to show 1.0m raised elevation
          g.lineStyle(2, 0xd6d3d1, 0.55);
          g.moveTo(px + 2, py + tilePx - 2);
          g.lineTo(px + 2, py + 2);
          g.lineTo(px + tilePx - 2, py + 2);
        } else {
          // Default grass tile
          const grassColor = (tx + ty) % 2 === 0 ? 0x3b6934 : 0x417239;
          g.lineStyle(1, 0x2f5629, 0.45);
          g.beginFill(grassColor, 1.0);
          g.drawRect(px, py, tilePx, tilePx);
          g.endFill();
        }
      }
    }
  }

  /**
   * Draw all world objects sorted by elevation and height, showing shape, size, color,
   * and height (via drop shadow offset, height-scaled shade, and outline thickness).
   */
  drawObjectsAndShadows() {
    const sg = this.shadowsGraphics;
    const og = this.objectsGraphics;
    sg.clear();
    og.clear();

    const state = this.store.getState();
    const objectsMap = state.objects || {};
    const holdingId = state.body?.holding || null;

    const renderList = [];
    for (const obj of Object.values(objectsMap)) {
      if (!obj || !obj.id) continue;
      if (obj.id === holdingId) continue; // Held item is drawn with the agent
      if (isHiddenInsideClosedContainer(obj, objectsMap)) continue;

      const elev = computeObjectElevation(obj, objectsMap);
      const height = Math.max(0.02, Number(obj.height) || 0.1);
      renderList.push({
        obj,
        elev,
        height,
        topZ: elev + height,
      });
    }

    // Sort bottom-to-top so taller/stacked objects render above lower ground objects
    renderList.sort((a, b) => {
      if (Math.abs(a.elev - b.elev) > 0.01) return a.elev - b.elev;
      if (Math.abs(a.topZ - b.topZ) > 0.01) return a.topZ - b.topZ;
      return a.obj.id.localeCompare(b.obj.id);
    });

    for (const item of renderList) {
      this._drawSingleObject(og, sg, item.obj, item.elev, item.height, objectsMap);
    }
  }

  _drawSingleObject(og, sg, obj, elev, height, objectsMap) {
    const pos = resolveObjectPosition(obj, objectsMap);
    const cx = pos.x * this.ppm;
    const cy = pos.y * this.ppm;

    const physRadius = computeObjectRadiusMeters(obj);
    // Ensure small items (like berries) have at least 6.5px radius and large trees scale nicely
    const radiusPx = Math.max(6.5, Math.min(38, physRadius * this.ppm * 0.95));

    const baseColor = resolveColorHex(obj.color, 0xcbd5e1);
    // Height affects shade (taller objects catch slightly brighter top light and cast deeper shadow)
    const heightShadeFactor = Math.max(0.78, Math.min(1.22, 0.92 + Math.min(height, 3.5) * 0.08));
    const fillColor = shadeColor(baseColor, heightShadeFactor);
    const outlineColor = shadeColor(baseColor, 0.42);
    // Outline thickness scales with object height (1.5px for flat/low up to 4.2px for tall trees)
    const outlineWidth = Math.max(1.5, Math.min(4.2, 1.4 + height * 0.85));

    // Shadow offset scales with height + base elevation
    const totalHeight = elev + height;
    const shadowOffset = Math.max(2, Math.min(14, 2 + totalHeight * 3.2));
    const shape = String(obj.shape || "sphere");

    // 1. Draw drop shadow on shadowsGraphics
    sg.lineStyle(0);
    sg.beginFill(0x080c08, Math.min(0.55, 0.25 + totalHeight * 0.08));
    if (shape === "cube") {
      const side = radiusPx * 1.65;
      sg.drawRoundedRect(cx - side * 0.5 + shadowOffset, cy - side * 0.5 + shadowOffset, side, side, 4);
    } else if (shape === "flat") {
      const w = radiusPx * 1.9;
      const h = radiusPx * 1.35;
      sg.drawRoundedRect(cx - w * 0.5 + shadowOffset, cy - h * 0.5 + shadowOffset, w, h, 6);
    } else if (shape === "long") {
      const len = Math.max(radiusPx * 2.2, Math.min(54, height * this.ppm * 0.65));
      const thick = Math.max(6, Math.min(14, radiusPx * 0.68));
      sg.drawRoundedRect(cx - len * 0.5 + shadowOffset, cy - thick * 0.5 + shadowOffset, len, thick, thick * 0.45);
    } else {
      sg.drawCircle(cx + shadowOffset, cy + shadowOffset, radiusPx);
    }
    sg.endFill();

    // 2. If elevated (`raised` or `on_top_of:`), draw an elevation base ring
    if (elev > 0.01) {
      og.lineStyle(1.5, 0xfef08a, 0.55);
      og.beginFill(0x000000, 0);
      og.drawCircle(cx, cy, radiusPx + 3.5);
      og.endFill();
    }

    // 3. Draw object body by shape
    og.lineStyle(outlineWidth, outlineColor, 0.95);
    og.beginFill(fillColor, 1.0);

    if (shape === "cube") {
      const side = radiusPx * 1.65;
      og.drawRoundedRect(cx - side * 0.5, cy - side * 0.5, side, side, 4);
      og.endFill();

      // Inner height contour frame for tall cubes (e.g. mailbox)
      if (height >= 0.5) {
        const inner = side * 0.62;
        og.lineStyle(1.2, shadeColor(fillColor, 1.25), 0.7);
        og.drawRect(cx - inner * 0.5, cy - inner * 0.5, inner, inner);
      }

      // If object has a lid (e.g. mailbox `state.lid`), show open vs closed indicator
      if (obj.state && typeof obj.state.lid === "string") {
        const isOpen = obj.state.lid === "open";
        const lidColor = isOpen ? 0x4ade80 : 0xfbbf24;
        og.lineStyle(1.5, 0x1e293b, 0.9);
        og.beginFill(lidColor, 0.95);
        og.drawRoundedRect(cx - side * 0.35, cy - side * 0.5 - 3, side * 0.7, 5, 2);
        og.endFill();
      }
    } else if (shape === "flat") {
      const w = radiusPx * 1.9;
      const h = radiusPx * 1.35;
      og.drawRoundedRect(cx - w * 0.5, cy - h * 0.5, w, h, 6);
      og.endFill();

      // Subtle inner slab bevel
      og.lineStyle(1, shadeColor(fillColor, 1.2), 0.5);
      og.drawRoundedRect(cx - w * 0.34, cy - h * 0.32, w * 0.68, h * 0.64, 4);
    } else if (shape === "long") {
      const len = Math.max(radiusPx * 2.2, Math.min(54, height * this.ppm * 0.65));
      const thick = Math.max(6, Math.min(14, radiusPx * 0.68));
      og.drawRoundedRect(cx - len * 0.5, cy - thick * 0.5, len, thick, thick * 0.45);
      og.endFill();

      if (obj.binding) {
        // Twisting vine stripe accent
        og.lineStyle(1.3, shadeColor(fillColor, 1.35), 0.75);
        og.moveTo(cx - len * 0.32, cy);
        og.lineTo(cx + len * 0.32, cy);
      }
    } else if (shape === "cylinder") {
      og.drawCircle(cx, cy, radiusPx);
      og.endFill();

      // Inner concentric height/ring contour for cylinders (trees, hollow logs)
      const innerR = radiusPx * (obj.hollow ? 0.52 : 0.62);
      if (obj.hollow) {
        og.lineStyle(1.5, outlineColor, 0.9);
        og.beginFill(0x1f1914, 0.88);
        og.drawCircle(cx, cy, innerR);
        og.endFill();
      } else {
        og.lineStyle(1.6, shadeColor(fillColor, 1.22), 0.65);
        og.beginFill(shadeColor(fillColor, 0.88), 0.5);
        og.drawCircle(cx, cy, innerR);
        og.endFill();
      }
    } else {
      // Sphere (default)
      og.drawCircle(cx, cy, radiusPx);
      og.endFill();

      // Top-left specular highlight showing spherical curvature and height
      const hlR = Math.max(2, radiusPx * 0.32);
      og.lineStyle(0);
      og.beginFill(0xffffff, 0.32);
      og.drawCircle(cx - radiusPx * 0.28, cy - radiusPx * 0.28, hlR);
      og.endFill();
    }

    // Hollow indicator for non-cylinder hollow objects
    if (obj.hollow && shape !== "cylinder") {
      og.lineStyle(1.2, 0x0f172a, 0.85);
      og.beginFill(0x0f172a, 0.55);
      og.drawCircle(cx, cy, Math.max(3, radiusPx * 0.35));
      og.endFill();
    }

    // Wetness indicator ring when object is wet
    if (obj.state?.wet) {
      og.lineStyle(1.2, 0x38bdf8, 0.7);
      og.beginFill(0x000000, 0);
      og.drawCircle(cx, cy, radiusPx + 2);
      og.endFill();
    }
  }

  /**
   * Draw the embodied agent at `body.position` and any held item (`body.holding`).
   */
  drawAgentAndHeldItem() {
    const ag = this.agentGraphics;
    ag.clear();

    const state = this.store.getState();
    const body = state.body;
    if (!body || !body.position) return;

    const ax = (Number(body.position.x) || 0) * this.ppm;
    const ay = (Number(body.position.y) || 0) * this.ppm;
    const agentRadius = 13;

    // Subtle sensing radius boundary circle (scaled by light level)
    const light = Number(state.time?.light_level ?? 1.0);
    const visFactor = 0.25 + 0.75 * light;
    const senseRadiusPx = 10.0 * visFactor * this.ppm;
    ag.lineStyle(1, 0x86efac, 0.16);
    ag.beginFill(0x86efac, 0.025);
    ag.drawCircle(ax, ay, senseRadiusPx);
    ag.endFill();

    // Agent drop shadow
    ag.lineStyle(0);
    ag.beginFill(0x050805, 0.48);
    ag.drawCircle(ax + 3, ay + 4, agentRadius + 1);
    ag.endFill();

    // Outer comfort/energy status halo
    const comfort = Math.max(0, Math.min(1, Number(body.comfort ?? 1.0)));
    const haloColor = comfort >= 0.5 ? 0x34d399 : comfort >= 0.25 ? 0xfbbf24 : 0xf87171;
    ag.lineStyle(2.2, haloColor, 0.85);
    ag.beginFill(0x000000, 0);
    ag.drawCircle(ax, ay, agentRadius + 3.5);
    ag.endFill();

    // Agent main body circle
    ag.lineStyle(2.5, 0x0f172a, 0.95);
    ag.beginFill(0xf59e0b, 1.0);
    ag.drawCircle(ax, ay, agentRadius);
    ag.endFill();

    // Inner head/eye highlight
    ag.lineStyle(0);
    ag.beginFill(0xfef3c7, 0.92);
    ag.drawCircle(ax, ay - 2, agentRadius * 0.45);
    ag.endFill();

    // Held item rendered in the agent's right hand
    const holdingId = body.holding;
    if (holdingId) {
      const heldObj = state.objects?.[holdingId];
      const hx = ax + agentRadius + 7;
      const hy = ay - 3;
      const heldColor = resolveColorHex(heldObj?.color, 0xe2e8f0);
      const heldShape = String(heldObj?.shape || "sphere");

      // Connector line from agent to held object
      ag.lineStyle(2, 0xfef08a, 0.9);
      ag.moveTo(ax + agentRadius * 0.7, ay - 1);
      ag.lineTo(hx, hy);

      ag.lineStyle(2, 0x0f172a, 0.95);
      ag.beginFill(heldColor, 1.0);
      if (heldShape === "long") {
        ag.drawRoundedRect(hx - 9, hy - 3.5, 18, 7, 3);
      } else if (heldShape === "cube" || heldShape === "flat") {
        ag.drawRoundedRect(hx - 6.5, hy - 6.5, 13, 13, 2.5);
      } else {
        ag.drawCircle(hx, hy, 6.5);
      }
      ag.endFill();
    }
  }

  /**
   * Draw selection and hover rings around the currently selected or hovered object/agent.
   */
  drawSelectionAndHover() {
    const selG = this.selectionGraphics;
    selG.clear();

    const state = this.store.getState();
    const objectsMap = state.objects || {};

    // Hovered target ring
    const hovered = state.hoveredTarget;
    if (hovered) {
      if (hovered.kind === "object" && objectsMap[hovered.id]) {
        const obj = objectsMap[hovered.id];
        const pos = resolveObjectPosition(obj, objectsMap);
        const rPx = Math.max(8, Math.min(40, computeObjectRadiusMeters(obj) * this.ppm * 0.95)) + 5;
        selG.lineStyle(2, 0xffffff, 0.85);
        selG.beginFill(0xffffff, 0.06);
        selG.drawCircle(pos.x * this.ppm, pos.y * this.ppm, rPx);
        selG.endFill();
      } else if (hovered.kind === "agent" && state.body?.position) {
        const ax = (Number(state.body.position.x) || 0) * this.ppm;
        const ay = (Number(state.body.position.y) || 0) * this.ppm;
        selG.lineStyle(2, 0xffffff, 0.85);
        selG.beginFill(0xffffff, 0.06);
        selG.drawCircle(ax, ay, 19);
        selG.endFill();
      }
    }

    // Selected object highlight ring
    const selectedId = state.selectedObjectId;
    if (selectedId && objectsMap[selectedId]) {
      const obj = objectsMap[selectedId];
      const pos = resolveObjectPosition(obj, objectsMap);
      const rPx = Math.max(9, Math.min(42, computeObjectRadiusMeters(obj) * this.ppm * 0.95)) + 8;
      selG.lineStyle(2.8, 0x38bdf8, 0.98);
      selG.beginFill(0x38bdf8, 0.1);
      selG.drawCircle(pos.x * this.ppm, pos.y * this.ppm, rPx);
      selG.endFill();
    }
  }

  /**
   * Draw screen-space time-of-day tint (darker at night, amber at dawn/dusk)
   * and current weather overlay (rain streaks, wind streaks + wind indicator, storm, cold).
   */
  drawWeatherAndNightOverlay() {
    const ng = this.nightOverlayGraphics;
    const wg = this.weatherOverlayGraphics;
    const indG = this.windIndicatorGraphics;
    ng.clear();
    wg.clear();
    indG.clear();

    const state = this.store.getState();
    const screenW = this.app.screen?.width || this.canvas.clientWidth || 900;
    const screenH = this.app.screen?.height || this.canvas.clientHeight || 680;

    // 1. Time of day scene tint (darken at night, warm glow at dawn/dusk)
    const light = Math.max(0, Math.min(1, Number(state.time?.light_level ?? 1.0)));
    const tod = ((Number(state.time?.time_of_day ?? 0.5) % 1) + 1) % 1;
    const darkness = Math.max(0, 1.0 - light);

    if (darkness > 0.02) {
      const nightAlpha = Math.min(0.68, darkness * 0.72);
      ng.lineStyle(0);
      ng.beginFill(0x070d1f, nightAlpha);
      ng.drawRect(0, 0, screenW, screenH);
      ng.endFill();
    }

    // Dawn (0.20..0.32) or Dusk (0.68..0.80) warm horizon tint
    const dawnDist = Math.abs(tod - 0.25);
    const duskDist = Math.abs(tod - 0.75);
    const twilightFactor = Math.max(
      0,
      1 - Math.min(dawnDist, duskDist) / 0.08
    );
    if (twilightFactor > 0.01) {
      ng.lineStyle(0);
      ng.beginFill(0xf59e0b, twilightFactor * 0.14);
      ng.drawRect(0, 0, screenW, screenH);
      ng.endFill();
    }

    // 2. Weather overlay (rain, wind, storm, cold)
    const weatherType = String(state.weather?.type || "clear").toLowerCase();
    const intensity = Math.max(0, Math.min(1, Number(state.weather?.intensity ?? 0.0)));
    const windDir = state.weather?.wind_direction || { x: 1.0, y: 0.0 };
    const windLen = Math.hypot(Number(windDir.x) || 0, Number(windDir.y) || 0) || 1;
    const wx = (Number(windDir.x) || 1) / windLen;
    const wy = (Number(windDir.y) || 0) / windLen;

    const isRain = weatherType === "rain" || weatherType === "storm";
    const isWind = weatherType === "wind" || weatherType === "storm";
    const isCold = weatherType === "cold";
    const effIntensity = intensity > 0 ? intensity : weatherType !== "clear" ? 0.5 : 0;

    if (weatherType === "storm") {
      // Storm overcast darkening
      wg.lineStyle(0);
      wg.beginFill(0x0f172a, 0.22 * effIntensity);
      wg.drawRect(0, 0, screenW, screenH);
      wg.endFill();
    } else if (isCold) {
      // Cold icy blue tint
      wg.lineStyle(0);
      wg.beginFill(0x7dd3fc, 0.14 * effIntensity);
      wg.drawRect(0, 0, screenW, screenH);
      wg.endFill();
    }

    if (isRain && effIntensity > 0) {
      const count = Math.max(12, Math.round(this._particles.length * effIntensity));
      const dropColor = weatherType === "storm" ? 0xbae6fd : 0x7dd3fc;
      wg.lineStyle(1.6, dropColor, 0.45 + 0.3 * effIntensity);
      const driftX = isWind ? wx * 8 : -4;
      const dropY = 18 + effIntensity * 10;

      for (let i = 0; i < count; i++) {
        const p = this._particles[i];
        const phase = (this._animTime * p.speed * 1.4 + p.v) % 1;
        const x = ((p.u + phase * 0.15) % 1) * screenW;
        const y = phase * screenH;
        wg.moveTo(x, y);
        wg.lineTo(x + driftX, y + dropY);
      }
    }

    if (isWind && effIntensity > 0) {
      const count = Math.max(10, Math.round(45 * effIntensity));
      wg.lineStyle(1.5, 0xe2e8f0, 0.35 + 0.25 * effIntensity);
      const streakLen = 26 + effIntensity * 28;

      for (let i = 0; i < count; i++) {
        const p = this._particles[i];
        const phase = (this._animTime * p.speed + p.u) % 1;
        const x = ((p.u + phase * wx + 1) % 1) * screenW;
        const y = ((p.v + phase * wy + 1) % 1) * screenH;
        wg.moveTo(x, y);
        wg.lineTo(x + wx * streakLen, y + wy * streakLen);
      }
    }

    if (isCold && effIntensity > 0) {
      const count = Math.max(10, Math.round(35 * effIntensity));
      wg.lineStyle(0);
      wg.beginFill(0xe0f2fe, 0.55);
      for (let i = 0; i < count; i++) {
        const p = this._particles[i];
        const phase = (this._animTime * 0.3 * p.speed + p.v) % 1;
        const x = p.u * screenW;
        const y = phase * screenH;
        wg.drawCircle(x, y, 2.2);
      }
      wg.endFill();
    }

    // 3. Draw on-canvas weather/wind compass indicator in bottom-left corner
    const badgeX = 42;
    const badgeY = screenH - 42;
    const badgeR = 24;

    indG.lineStyle(1.5, 0x334155, 0.9);
    indG.beginFill(0x0f172a, 0.78);
    indG.drawCircle(badgeX, badgeY, badgeR);
    indG.endFill();

    if (isWind && effIntensity > 0) {
      const arrowLen = badgeR * 0.62;
      const tipX = badgeX + wx * arrowLen;
      const tipY = badgeY + wy * arrowLen;
      const tailX = badgeX - wx * (arrowLen * 0.65);
      const tailY = badgeY - wy * (arrowLen * 0.65);
      const perpX = -wy * 5;
      const perpY = wx * 5;

      indG.lineStyle(2.2, 0x38bdf8, 0.95);
      indG.moveTo(tailX, tailY);
      indG.lineTo(tipX, tipY);
      indG.moveTo(tipX - wx * 6 + perpX, tipY - wy * 6 + perpY);
      indG.lineTo(tipX, tipY);
      indG.lineTo(tipX - wx * 6 - perpX, tipY - wy * 6 - perpY);
    } else if (isRain) {
      indG.lineStyle(2, 0x60a5fa, 0.95);
      for (const dx of [-6, 0, 6]) {
        indG.moveTo(badgeX + dx + 2, badgeY - 7);
        indG.lineTo(badgeX + dx - 2, badgeY + 7);
      }
    } else if (isCold) {
      indG.lineStyle(1.8, 0x7dd3fc, 0.95);
      indG.moveTo(badgeX - 8, badgeY);
      indG.lineTo(badgeX + 8, badgeY);
      indG.moveTo(badgeX, badgeY - 8);
      indG.lineTo(badgeX, badgeY + 8);
    } else {
      // Clear sky sun/moon dot
      const orbColor = light >= 0.45 ? 0xfacc15 : 0x93c5fd;
      indG.lineStyle(0);
      indG.beginFill(orbColor, 0.92);
      indG.drawCircle(badgeX, badgeY, 7);
      indG.endFill();
    }
  }

  destroy() {
    if (this.app.ticker && typeof this.app.ticker.remove === "function") {
      this.app.ticker.remove(this._onTick);
    }
    this.app.destroy();
  }
}
