/**
 * Solar ML 24-Hour Hourly Prediction Bar Chart Card (`custom:solar-ml-24h-card`)
 *
 * Supports both:
 *  1. `sensor.<name>_forecast_next_24h` from the `solar_ml_predictor` integration
 *     (reads `attributes.series_compact` or `attributes.hourly_forecast`)
 *  2. Standalone `input_text.*` helper storing `"<start_utc_epoch_hour>|<v0>,...,<v23>"`
 */
class SolarMl24hCard extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: "open" });
    this._config = {};
    this._hass = null;
    this._lastSignature = null;
    this._activeIdx = -1;
    this._slots = [];
    this._totalKwh = 0;
    this._width = 480;
    this._height = 175;
    this._ro = null;
  }

  static getStubConfig() {
    return {
      entity: "sensor.solar_ml_predictor_forecast_next_24h",
      title: "Solar Prediction (24h)",
    };
  }

  connectedCallback() {
    if (!this._ro && typeof ResizeObserver !== "undefined") {
      this._ro = new ResizeObserver((entries) => {
        for (const entry of entries) {
          const w = Math.round(entry.contentRect.width);
          const h = Math.round(entry.contentRect.height);
          if (
            w > 80 &&
            h > 60 &&
            (Math.abs(w - this._width) > 2 || Math.abs(h - this._height) > 2)
          ) {
            this._width = w;
            this._height = h;
            this._renderSvgOnly();
          }
        }
      });
    }
    this._observeWrap();
  }

  disconnectedCallback() {
    if (this._ro) this._ro.disconnect();
  }

  _observeWrap() {
    if (!this._ro || !this.shadowRoot) return;
    const wrap = this.shadowRoot.querySelector(".chart-wrap");
    if (wrap) {
      this._ro.disconnect();
      this._ro.observe(wrap);
    }
  }

  setConfig(config) {
    if (!config || !config.entity) {
      throw new Error("Please specify an entity (sensor.*_forecast_next_24h or input_text.*)");
    }
    this._config = {
      title: "Solar Prediction (24h)",
      navigation_path: null,
      ...config,
    };
    this._lastSignature = null;
    this._render();
  }

  getCardSize() {
    return 4;
  }

  getGridOptions() {
    return {
      columns: 12,
      min_columns: 6,
      rows: 4,
      min_rows: 3,
    };
  }

  set hass(hass) {
    this._hass = hass;
    const entityId = this._config.entity;
    const stateObj = hass?.states?.[entityId];
    const compact =
      stateObj?.attributes?.series_compact ||
      (typeof stateObj?.state === "string" && stateObj.state.includes("|")
        ? stateObj.state
        : "");
    const hourlyJson = stateObj?.attributes?.hourly_forecast
      ? JSON.stringify(stateObj.attributes.hourly_forecast)
      : "";
    const localeKey = `${hass?.locale?.language || ""}|${hass?.locale?.time_format || ""}|${hass?.config?.time_zone || ""}`;
    const sig = `${compact}::${hourlyJson}::${localeKey}`;

    if (sig !== this._lastSignature) {
      this._lastSignature = sig;
      this._parseData(compact, stateObj?.attributes?.hourly_forecast);
      this._render();
    }
  }

  _getFormatters() {
    const tz = this._hass?.config?.time_zone || Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC";
    const lang = this._hass?.locale?.language || "en";
    const tf = this._hass?.locale?.time_format;
    const use12h =
      tf === "12"
        ? true
        : tf === "24"
          ? false
          : new Intl.DateTimeFormat(lang, { hour: "numeric" }).resolvedOptions().hour12 === true;

    return {
      hour24Fmt: new Intl.DateTimeFormat("en-GB", {
        timeZone: tz,
        hour: "2-digit",
        hour12: false,
      }),
      dateShortFmt: new Intl.DateTimeFormat(lang, {
        timeZone: tz,
        month: "short",
        day: "numeric",
      }),
      timeTickFmt: new Intl.DateTimeFormat(lang, {
        timeZone: tz,
        hour: "numeric",
        minute: "2-digit",
        hour12: use12h,
      }),
    };
  }

  _parseData(compact, hourlyList) {
    this._slots = [];
    this._totalKwh = 0;
    const { hour24Fmt, dateShortFmt, timeTickFmt } = this._getFormatters();

    if (compact && compact.includes("|")) {
      const [ehStr, valsStr] = compact.split("|", 2);
      const startEh = parseInt(ehStr, 10);
      if (Number.isFinite(startEh) && valsStr) {
        const parts = valsStr.split(",").slice(0, 24);
        let sum = 0;
        for (let i = 0; i < parts.length; i++) {
          const val = Math.max(0, parseFloat(parts[i]) || 0);
          sum += val;
          const startMs = (startEh + i) * 3600 * 1000;
          const dt = new Date(startMs);
          const dtEnd = new Date(startMs + 3600 * 1000);
          const h24 = parseInt(hour24Fmt.format(dt), 10) % 24;
          this._slots.push({
            idx: i,
            val,
            h24,
            dateLabel: dateShortFmt.format(dt),
            tickLabel: h24 === 0 ? dateShortFmt.format(dt) : timeTickFmt.format(dt),
            rangeLabel: `${timeTickFmt.format(dt)} – ${timeTickFmt.format(dtEnd)}`,
          });
        }
        this._totalKwh = sum;
        return;
      }
    }

    if (Array.isArray(hourlyList) && hourlyList.length > 0) {
      let sum = 0;
      const items = hourlyList.slice(0, 24);
      for (let i = 0; i < items.length; i++) {
        const item = items[i];
        const val = Math.max(0, parseFloat(item.kwh) || 0);
        const dt = new Date(item.datetime);
        if (Number.isNaN(dt.getTime())) continue;
        sum += val;
        const dtEnd = new Date(dt.getTime() + 3600 * 1000);
        const h24 = parseInt(hour24Fmt.format(dt), 10) % 24;
        this._slots.push({
          idx: i,
          val,
          h24,
          dateLabel: dateShortFmt.format(dt),
          tickLabel: h24 === 0 ? dateShortFmt.format(dt) : timeTickFmt.format(dt),
          rangeLabel: `${timeTickFmt.format(dt)} – ${timeTickFmt.format(dtEnd)}`,
        });
      }
      this._totalKwh = sum;
    }
  }

  _chooseYScale(maxVal) {
    if (maxVal <= 2.0) return { yMax: 2.0, ticks: [0, 0.5, 1, 1.5, 2] };
    if (maxVal <= 3.0) return { yMax: 3.0, ticks: [0, 1, 2, 3] };
    if (maxVal <= 4.0) return { yMax: 4.0, ticks: [0, 1, 2, 3, 4] };
    if (maxVal <= 5.0) return { yMax: 5.0, ticks: [0, 1, 2, 3, 4, 5] };
    if (maxVal <= 6.0) return { yMax: 6.0, ticks: [0, 2, 4, 6] };
    if (maxVal <= 8.0) return { yMax: 8.0, ticks: [0, 2, 4, 6, 8] };
    return { yMax: 10.0, ticks: [0, 2, 4, 6, 8, 10] };
  }

  _barPath(x, yTop, w, yBottom, r) {
    const h = yBottom - yTop;
    if (h <= 0.5) return "";
    const rr = Math.min(r, w / 2, h);
    return `M ${x.toFixed(1)},${yBottom.toFixed(1)} L ${x.toFixed(1)},${(yTop + rr).toFixed(1)} Q ${x.toFixed(1)},${yTop.toFixed(1)} ${(x + rr).toFixed(1)},${yTop.toFixed(1)} L ${(x + w - rr).toFixed(1)},${yTop.toFixed(1)} Q ${(x + w).toFixed(1)},${yTop.toFixed(1)} ${(x + w).toFixed(1)},${(yTop + rr).toFixed(1)} L ${(x + w).toFixed(1)},${yBottom.toFixed(1)} Z`;
  }

  _navigate() {
    const path = this._config.navigation_path;
    if (!path) return;
    window.history.pushState(null, "", path);
    window.dispatchEvent(new CustomEvent("location-changed", { composed: true }));
  }

  _updateInteractive(idx) {
    this._activeIdx = idx;
    const badge = this.shadowRoot.querySelector(".total-pill");
    const tip = this.shadowRoot.querySelector(".hover-tip");
    const bars = this.shadowRoot.querySelectorAll(".bar-shape");
    bars.forEach((el) => {
      const bIdx = parseInt(el.getAttribute("data-idx"), 10);
      el.classList.toggle("active", bIdx === idx);
    });

    if (idx >= 0 && idx < this._slots.length) {
      const s = this._slots[idx];
      if (badge) {
        badge.textContent = `${s.val.toFixed(2)} kWh`;
        badge.classList.add("active");
      }
      if (tip) {
        tip.textContent = `${s.dateLabel}, ${s.rangeLabel}: ${s.val.toFixed(2)} kWh`;
        tip.classList.add("visible");
      }
    } else {
      if (badge) {
        badge.textContent = `${this._totalKwh.toFixed(1)} kWh`;
        badge.classList.remove("active");
      }
      if (tip) {
        tip.classList.remove("visible");
      }
    }
  }

  _buildSvgMarkup() {
    const title = this._config.title || "Solar Prediction (24h)";
    const slots = this._slots;
    if (!slots.length) {
      return `<div class="empty-msg">Loading 24h solar forecast…</div>`;
    }

    const W = Math.max(240, this._width || 480);
    const H = Math.max(130, this._height || 175);
    const padL = 28;
    const padR = 8;
    const padT = 18;
    const padB = 22;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;
    const yBottom = padT + plotH;

    const maxVal = Math.max(...slots.map((s) => s.val), 0.1);
    const { yMax, ticks } = this._chooseYScale(maxVal);
    const n = slots.length;
    const stepX = plotW / n;
    const barW = Math.max(5, stepX * 0.68);

    const yLines = ticks
      .map((t) => {
        const y = Math.round(yBottom - (t / yMax) * plotH) + 0.5;
        const lbl = Number.isInteger(t) ? `${t}` : `${t.toFixed(1)}`;
        return `
          <line x1="${padL}" y1="${y}" x2="${W - padR}" y2="${y}" class="grid-line" />
          <text x="${padL - 6}" y="${y + 3.5}" text-anchor="end" class="axis-lbl">${lbl}</text>
        `;
      })
      .join("");

    const xTicks = slots
      .map((s, i) => {
        if (s.h24 % 4 !== 0) return "";
        const cx = Math.round(padL + (i + 0.5) * stepX) + 0.5;
        const anchor = i <= 1 ? "start" : i >= n - 2 ? "end" : "middle";
        const lx = i <= 1 ? padL : i >= n - 2 ? W - padR : cx;
        const cls = s.h24 === 0 ? "axis-lbl midnight-lbl" : "axis-lbl";
        return `
          <line x1="${cx}" y1="${padT}" x2="${cx}" y2="${yBottom}" class="grid-line-v" />
          <text x="${lx}" y="${H - 4}" text-anchor="${anchor}" class="${cls}">${s.tickLabel}</text>
        `;
      })
      .join("");

    const barsSvg = slots
      .map((s, i) => {
        const cx = padL + (i + 0.5) * stepX;
        const bx = cx - barW / 2;
        const bh = (Math.min(s.val, yMax) / yMax) * plotH;
        const by = yBottom - bh;
        const d = s.val > 0.004 ? this._barPath(bx, by, barW, yBottom, 3.5) : "";
        const activeCls = this._activeIdx === i ? " active" : "";
        return `
          ${d ? `<path d="${d}" class="bar-shape${activeCls}" data-idx="${i}" />` : ""}
          <rect x="${(padL + i * stepX).toFixed(1)}" y="${padT}" width="${stepX.toFixed(1)}" height="${plotH}" class="hit-col" data-idx="${i}" />
        `;
      })
      .join("");

    return `
      <svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" class="chart-svg" role="img" aria-label="${title}">
        <text x="${padL}" y="11" class="unit-lbl">kWh</text>
        ${yLines}
        ${xTicks}
        ${barsSvg}
      </svg>
    `;
  }

  _bindSvgEvents() {
    const cols = this.shadowRoot.querySelectorAll(".hit-col");
    cols.forEach((col) => {
      const i = parseInt(col.getAttribute("data-idx"), 10);
      col.addEventListener("pointerenter", () => this._updateInteractive(i));
      col.addEventListener("click", (ev) => {
        ev.stopPropagation();
        this._updateInteractive(this._activeIdx === i ? -1 : i);
      });
    });
    const svg = this.shadowRoot.querySelector(".chart-svg");
    if (svg) {
      svg.addEventListener("pointerleave", () => this._updateInteractive(-1));
    }
  }

  _renderSvgOnly() {
    const wrap = this.shadowRoot?.querySelector(".chart-wrap");
    if (!wrap) return;
    wrap.innerHTML = this._buildSvgMarkup();
    this._bindSvgEvents();
  }

  _render() {
    if (!this.shadowRoot) return;
    const title = this._config.title || "Solar Prediction (24h)";
    const clickable = Boolean(this._config.navigation_path);

    this.shadowRoot.innerHTML = `
      <style>
        :host {
          display: block;
          height: 100%;
        }
        ha-card {
          height: 100%;
          box-sizing: border-box;
          padding: 14px 16px 10px 16px;
          display: flex;
          flex-direction: column;
          justify-content: space-between;
          overflow: hidden;
        }
        .header-row {
          display: flex;
          align-items: flex-start;
          justify-content: space-between;
          gap: 8px;
          margin-bottom: 2px;
          ${clickable ? "cursor: pointer;" : ""}
        }
        .title-wrap {
          display: flex;
          flex-direction: column;
          min-width: 0;
        }
        .card-title {
          font-size: 18px;
          font-weight: 400;
          line-height: 1.25;
          color: var(--primary-text-color);
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }
        .hover-tip {
          font-size: 11.5px;
          line-height: 1.2;
          height: 14px;
          color: #ffa726;
          opacity: 0;
          transition: opacity 120ms ease;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }
        .hover-tip.visible {
          opacity: 1;
        }
        .total-pill {
          flex-shrink: 0;
          border: 1px solid var(--divider-color, rgba(255, 255, 255, 0.16));
          border-radius: 8px;
          padding: 3px 10px;
          font-size: 13px;
          font-weight: 500;
          color: var(--primary-text-color);
          background: rgba(127, 127, 127, 0.06);
          transition: border-color 120ms ease, color 120ms ease;
        }
        .total-pill.active {
          border-color: #ff9800;
          color: #ffa726;
        }
        .chart-wrap {
          flex: 1;
          min-height: 155px;
          width: 100%;
          display: flex;
          align-items: center;
          justify-content: center;
        }
        .chart-svg {
          display: block;
          width: 100%;
          height: 100%;
          user-select: none;
          -webkit-user-select: none;
        }
        .unit-lbl {
          font-size: 11px;
          fill: var(--secondary-text-color);
          font-family: var(--paper-font-body1_-_font-family, inherit);
        }
        .axis-lbl {
          font-size: 11px;
          fill: var(--secondary-text-color);
          font-family: var(--paper-font-body1_-_font-family, inherit);
        }
        .midnight-lbl {
          font-weight: 600;
          fill: var(--primary-text-color);
        }
        .grid-line {
          stroke: var(--divider-color, rgba(255, 255, 255, 0.12));
          stroke-width: 1;
        }
        .grid-line-v {
          stroke: var(--divider-color, rgba(255, 255, 255, 0.07));
          stroke-width: 1;
        }
        .bar-shape {
          fill: rgba(255, 152, 0, 0.46);
          stroke: #ff9800;
          stroke-width: 1.4;
          transition: fill 100ms ease;
        }
        .bar-shape.active {
          fill: rgba(255, 167, 38, 0.78);
        }
        .hit-col {
          fill: transparent;
          cursor: crosshair;
        }
        .empty-msg {
          margin: auto;
          font-size: 13px;
          color: var(--secondary-text-color);
        }
      </style>
      <ha-card>
        <div class="header-row">
          <div class="title-wrap">
            <div class="card-title">${title}</div>
            <div class="hover-tip"></div>
          </div>
          <div class="total-pill">${this._totalKwh.toFixed(1)} kWh</div>
        </div>
        <div class="chart-wrap">
          ${this._buildSvgMarkup()}
        </div>
      </ha-card>
    `;

    if (clickable) {
      const hdr = this.shadowRoot.querySelector(".header-row");
      if (hdr) hdr.addEventListener("click", () => this._navigate());
    }
    this._bindSvgEvents();
    this._observeWrap();
  }
}

if (!customElements.get("solar-ml-24h-card")) {
  customElements.define("solar-ml-24h-card", SolarMl24hCard);
}
window.customCards = window.customCards || [];
if (!window.customCards.some((c) => c.type === "solar-ml-24h-card")) {
  window.customCards.push({
    type: "solar-ml-24h-card",
    name: "Solar ML 24h Prediction Graph",
    description: "Rolling 24-hour hourly solar ML production bar chart",
  });
}
