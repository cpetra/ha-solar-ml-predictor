"""Coordinator & Online Normalized LMS Learner for Solar ML Predictor."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging
import math
from typing import Any, Callable

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    EVENT_HOMEASSISTANT_STARTED,
    STATE_UNAVAILABLE,
    STATE_UNKNOWN,
)
from homeassistant.core import CoreState, Event, HomeAssistant, callback
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.event import (
    async_track_state_change_event,
    async_track_time_change,
)
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util

from .const import (
    CONF_BATTERY_SOC_ENTITY,
    CONF_CLIP_SOC_THRESHOLD,
    CONF_CLOUD_COVERAGE_ENTITY,
    CONF_DAILY_PV_ENTITY,
    CONF_HUMIDITY_ENTITY,
    CONF_INVERTER_AC_LIMIT_KW,
    CONF_MAX_TRAINING_SAMPLES,
    CONF_WEATHER_ENTITY,
    DEFAULT_CLIP_SOC_THRESHOLD,
    DEFAULT_INITIAL_SAMPLES,
    DEFAULT_INVERTER_AC_LIMIT_KW,
    DEFAULT_MAX_TRAINING_SAMPLES,
    DOMAIN,
    PANEL_GROUP_KEYS,
    PRIOR_WEIGHTS,
    REFERENCE_POWER_KW,
    STORAGE_VERSION,
)
from .ml_engine import (
    PanelGroupConfig,
    build_feature_vector,
    compute_multigroup_factors,
    compute_solar_position,
    online_nlms_update,
    predict_hourly_kwh,
    solve_ridge_4x4,
)

_LOGGER = logging.getLogger(__name__)


class SolarMlCoordinator:
    """Manages multi-group solar ML online learning, persistent weights, and 48h/24h forecasts."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the Solar ML coordinator."""
        self.hass = hass
        self.entry = entry
        self._store: Store[dict[str, Any]] = Store(
            hass, STORAGE_VERSION, f"{DOMAIN}.{entry.entry_id}"
        )
        self._listeners: list[Callable[[], None]] = []
        self._unsubs: list[Callable[[], None]] = []

        # Persisted online learner state
        self.weights: list[float] = list(PRIOR_WEIGHTS)
        self.samples: int = DEFAULT_INITIAL_SAMPLES
        self.pv0: float = 0.0
        self.s0: float = 50.0

        # Cached forecast outputs
        self.forecast_today: float = 0.0
        self.forecast_remaining_today: float = 0.0
        self.forecast_tomorrow: float = 0.0
        self.forecast_next_24h: float = 0.0
        self.series_compact_24h: str = ""
        self.hourly_24h: list[dict[str, Any]] = []
        self._cached_owm_slots: list[dict[str, Any]] = []
        self.last_updated: datetime | None = None

    @property
    def cfg(self) -> dict[str, Any]:
        """Merged configuration dictionary (options override initial data)."""
        return {**self.entry.data, **self.entry.options}

    @property
    def panel_groups(self) -> list[PanelGroupConfig]:
        """Return active solar panel groups (`peak_power_kw > 0`), up to 4 groups."""
        c = self.cfg
        groups: list[PanelGroupConfig] = []
        for idx, (name_k, peak_k, tilt_k, az_k) in enumerate(
            PANEL_GROUP_KEYS, start=1
        ):
            peak_kw = float(c.get(peak_k, 0.0) or 0.0)
            if peak_kw > 0.0:
                groups.append(
                    PanelGroupConfig(
                        name=str(c.get(name_k) or f"Group {idx}"),
                        peak_power_kw=round(peak_kw, 3),
                        tilt_deg=float(c.get(tilt_k, 45.0) or 0.0),
                        azimuth_deg=float(c.get(az_k, 180.0) or 0.0),
                    )
                )
        return groups

    @property
    def total_peak_power_kw(self) -> float:
        """Total installed DC peak power across all active panel groups."""
        return round(sum(g.peak_power_kw for g in self.panel_groups), 3)

    @property
    def inverter_ac_limit_kw(self) -> float:
        """Configured inverter AC/MPPT power limit (kW)."""
        return float(
            self.cfg.get(CONF_INVERTER_AC_LIMIT_KW, DEFAULT_INVERTER_AC_LIMIT_KW)
        )

    @property
    def clip_soc_threshold(self) -> float:
        """Configured battery SOC clipping threshold (%)."""
        return float(
            self.cfg.get(CONF_CLIP_SOC_THRESHOLD, DEFAULT_CLIP_SOC_THRESHOLD)
        )

    @property
    def max_training_samples(self) -> int:
        """Configured maximum rolling online sample count."""
        return int(
            self.cfg.get(CONF_MAX_TRAINING_SAMPLES, DEFAULT_MAX_TRAINING_SAMPLES)
        )

    @property
    def device_info(self) -> DeviceInfo:
        """Device registry metadata grouping all Solar ML entities."""
        return DeviceInfo(
            identifiers={(DOMAIN, self.entry.entry_id)},
            name=f"Solar ML Predictor ({self.total_peak_power_kw:.2f} kWp)",
            manufacturer="Solar ML Predictor",
            model=f"{len(self.panel_groups)}-Group Physics-Informed NLMS",
            entry_type=DeviceEntryType.SERVICE,
        )

    def add_listener(self, update_cb: Callable[[], None]) -> Callable[[], None]:
        """Register an entity state update callback."""
        self._listeners.append(update_cb)

        @callback
        def _remove() -> None:
            if update_cb in self._listeners:
                self._listeners.remove(update_cb)

        return _remove

    @callback
    def _notify_listeners(self) -> None:
        """Push updated values to all subscribed entities."""
        for cb in list(self._listeners):
            cb()

    async def async_load_state(self) -> None:
        """Load persisted model weights and baselines from Home Assistant storage."""
        stored = await self._store.async_load()
        if isinstance(stored, dict):
            w = stored.get("weights")
            if isinstance(w, list) and len(w) == 4:
                self.weights = [float(x) for x in w]
            self.samples = int(stored.get("samples", DEFAULT_INITIAL_SAMPLES))
            self.pv0 = float(stored.get("pv0", 0.0))
            self.s0 = float(stored.get("s0", 50.0))
            self.series_compact_24h = str(stored.get("series_compact_24h", ""))
        else:
            # Initialize baseline from current live PV/SOC if available
            pv_val = self._read_pv_kwh()
            soc_val = self._read_soc()
            self.pv0 = pv_val if pv_val is not None else 0.0
            self.s0 = soc_val if soc_val is not None else 50.0

    async def async_save_state(self) -> None:
        """Persist model weights and hourly baselines to storage."""
        await self._store.async_save(
            {
                "weights": [round(w, 6) for w in self.weights],
                "samples": int(self.samples),
                "pv0": round(self.pv0, 3),
                "s0": round(self.s0, 1),
                "series_compact_24h": self.series_compact_24h,
                "updated_at": dt_util.utcnow().isoformat(),
            }
        )

    def start(self) -> None:
        """Start event listeners and periodic `:02` hourly online learner."""
        c = self.cfg
        weather_eid = c[CONF_WEATHER_ENTITY]
        pv_eid = c[CONF_DAILY_PV_ENTITY]

        self._unsubs.append(
            async_track_time_change(
                self.hass, self._handle_hourly_tick, minute=2, second=0
            )
        )
        self._unsubs.append(
            async_track_state_change_event(
                self.hass, [weather_eid], self._handle_weather_change
            )
        )
        self._unsubs.append(
            async_track_state_change_event(
                self.hass, [pv_eid], self._handle_pv_change
            )
        )

        if self.hass.state is CoreState.running:
            self.hass.async_create_task(self.async_refresh_forecast())
        else:
            self.hass.bus.async_listen_once(
                EVENT_HOMEASSISTANT_STARTED, self._handle_ha_started
            )

    @callback
    def shutdown(self) -> None:
        """Unsubscribe all listeners on unload."""
        while self._unsubs:
            unsub = self._unsubs.pop()
            unsub()

    async def _handle_ha_started(self, _event: Event) -> None:
        """Run initial forecast evaluation once Home Assistant finishes starting."""
        await self.async_refresh_forecast()

    async def _handle_weather_change(self, _event: Event) -> None:
        """Re-evaluate forecasts when the weather entity updates."""
        await self.async_refresh_forecast()

    @callback
    def _handle_pv_change(self, _event: Event) -> None:
        """Recompute remaining/today totals immediately when daily PV increments."""
        if self._cached_owm_slots:
            self._evaluate_cached_slots()
            self._notify_listeners()

    async def _handle_hourly_tick(self, _now: datetime) -> None:
        """Execute `:02` online Normalized LMS learning step and refresh 48h/24h forecasts."""
        self._run_online_learning_step()
        await self.async_refresh_forecast(save_after=True)

    def _read_numeric_state(self, entity_id: str | None) -> float | None:
        """Return float state of an entity, or None if unavailable/non-numeric."""
        if not entity_id:
            return None
        st = self.hass.states.get(entity_id)
        if st is None or st.state in (STATE_UNKNOWN, STATE_UNAVAILABLE, ""):
            return None
        try:
            return float(st.state)
        except (ValueError, TypeError):
            return None

    def _read_pv_kwh(self) -> float | None:
        """Read daily PV generation converted to kWh (supports Wh, kWh, MWh)."""
        pv_eid = self.cfg.get(CONF_DAILY_PV_ENTITY)
        raw = self._read_numeric_state(pv_eid)
        if raw is None:
            return None
        st = self.hass.states.get(pv_eid)
        unit = (st.attributes.get("unit_of_measurement") or "kWh") if st else "kWh"
        if unit == "Wh":
            return raw / 1000.0
        if unit == "MWh":
            return raw * 1000.0
        return raw

    def _read_soc(self) -> float | None:
        """Read current battery SOC (%), or None if no battery sensor is configured."""
        return self._read_numeric_state(self.cfg.get(CONF_BATTERY_SOC_ENTITY))

    def _read_live_weather_factors(self) -> tuple[float, float] | None:
        """Return live `(cloud_coverage_pct, humidity_pct)` from sensors or weather attributes."""
        c = self.cfg
        weather_st = self.hass.states.get(c[CONF_WEATHER_ENTITY])
        clouds = self._read_numeric_state(c.get(CONF_CLOUD_COVERAGE_ENTITY))
        if clouds is None and weather_st is not None:
            raw_c = weather_st.attributes.get("cloud_coverage")
            if raw_c is not None:
                try:
                    clouds = float(raw_c)
                except (ValueError, TypeError):
                    clouds = None

        humidity = self._read_numeric_state(c.get(CONF_HUMIDITY_ENTITY))
        if humidity is None and weather_st is not None:
            raw_h = weather_st.attributes.get("humidity")
            if raw_h is not None:
                try:
                    humidity = float(raw_h)
                except (ValueError, TypeError):
                    humidity = None

        if clouds is None or humidity is None:
            return None
        return clouds, humidity

    def _run_online_learning_step(self) -> None:
        """Evaluate the just-completed daylight hour and update weights via Normalized LMS."""
        now_loc = dt_util.now()
        now_utc = dt_util.utcnow()
        pv_raw = self._read_pv_kwh()
        has_pv = pv_raw is not None
        pv_now = 0.0 if now_loc.hour == 0 else (pv_raw if pv_raw is not None else max(0.0, self.pv0))

        has_soc_cfg = bool(self.cfg.get(CONF_BATTERY_SOC_ENTITY))
        soc_val = self._read_soc()
        has_soc = (soc_val is not None) if has_soc_cfg else True
        soc_now = soc_val if soc_val is not None else 0.0

        wx = self._read_live_weather_factors()
        dt_prev = now_utc - timedelta(minutes=30)
        el_deg_p, az_deg_p, sin_el_p = compute_solar_position(
            dt_prev,
            self.hass.config.latitude,
            self.hass.config.longitude,
            midpoint_offset_hours=0.0,
        )

        y_act = pv_now - self.pv0
        clip_soc = self.clip_soc_threshold
        inv_cap = self.inverter_ac_limit_kw

        if (
            has_pv
            and has_soc
            and wx is not None
            and el_deg_p > 3.0
            and soc_now < clip_soc
            and self.s0 < clip_soc
            and self.pv0 >= 0.0
            and 0.0 <= y_act < max(0.3, inv_cap - 0.2)
        ):
            poa_p, sky_p = compute_multigroup_factors(
                el_deg_p, az_deg_p, self.panel_groups, REFERENCE_POWER_KW
            )
            feats = build_feature_vector(poa_p, sky_p, sin_el_p, wx[0], wx[1])
            self.weights, self.samples = online_nlms_update(
                weights=self.weights,
                n_samples=self.samples,
                max_samples=self.max_training_samples,
                features=feats,
                actual_kwh=y_act,
                inverter_ac_limit_kw=inv_cap,
            )

        self.pv0 = 0.0 if now_loc.hour == 0 else (pv_now if has_pv else -1.0)
        if has_soc:
            self.s0 = soc_now

    async def async_refresh_forecast(self, save_after: bool = False) -> None:
        """Fetch hourly weather forecast via `weather.get_forecasts` and evaluate predictions."""
        weather_eid = self.cfg[CONF_WEATHER_ENTITY]
        try:
            resp = await self.hass.services.async_call(
                "weather",
                "get_forecasts",
                {"entity_id": weather_eid, "type": "hourly"},
                blocking=True,
                return_response=True,
            )
            if isinstance(resp, dict):
                fc_list = resp.get(weather_eid, {}).get("forecast", [])
                if isinstance(fc_list, list) and fc_list:
                    self._cached_owm_slots = fc_list
        except Exception as err:  # noqa: BLE001
            _LOGGER.warning(
                "Solar ML Predictor could not fetch hourly forecast from %s: %s",
                weather_eid,
                err,
            )

        self._evaluate_cached_slots()
        if save_after:
            await self.async_save_state()
        self._notify_listeners()

    @callback
    def _evaluate_cached_slots(self) -> None:
        """Evaluate Today, Remaining Today, Tomorrow, and Next-24h series from cached weather slots."""
        now_loc = dt_util.now()
        now_utc = dt_util.utcnow()
        now_ts = now_utc.timestamp()

        pv_raw = self._read_pv_kwh()
        pv_now = (
            0.0
            if now_loc.hour == 0
            else (pv_raw if pv_raw is not None else max(0.0, self.pv0))
        )

        today_str = now_loc.strftime("%Y-%m-%d")
        tomorrow_str = (now_loc + timedelta(days=1)).strftime("%Y-%m-%d")
        lat = self.hass.config.latitude
        lon = self.hass.config.longitude
        groups = self.panel_groups
        inv_cap = self.inverter_ac_limit_kw

        rem_today = 0.0
        tomorrow_total = 0.0
        h24_vals: list[str] = []
        h24_objs: list[dict[str, Any]] = []
        start_eh: int | None = None

        for item in self._cached_owm_slots:
            dt_raw = item.get("datetime")
            if not dt_raw:
                continue
            dt_parsed = dt_util.parse_datetime(str(dt_raw))
            if dt_parsed is None:
                continue
            dt_utc = dt_parsed.astimezone(timezone.utc)
            slot_start = dt_utc.timestamp()
            slot_end = slot_start + 3600.0
            if slot_end <= now_ts:
                continue

            dt_loc = dt_util.as_local(dt_utc)
            d_str = dt_loc.strftime("%Y-%m-%d")
            need_h24 = len(h24_vals) < 24

            if not need_h24 and d_str not in (today_str, tomorrow_str):
                continue

            el_deg, az_deg, sin_el = compute_solar_position(
                dt_utc, lat, lon, midpoint_offset_hours=0.5
            )
            if el_deg > 0.0:
                poa_eff, sky_scale = compute_multigroup_factors(
                    el_deg, az_deg, groups, REFERENCE_POWER_KW
                )
                clouds = float(item.get("cloud_coverage", 50.0) or 0.0)
                humidity = float(item.get("humidity", 80.0) or 0.0)
                feats = build_feature_vector(
                    poa_eff, sky_scale, sin_el, clouds, humidity
                )
                pred_h = predict_hourly_kwh(
                    self.weights, feats, el_deg, inv_cap
                )
            else:
                pred_h = 0.0

            if need_h24:
                if start_eh is None:
                    start_eh = int(slot_start // 3600)
                v_fmt = (
                    "0"
                    if pred_h < 0.005
                    else f"{pred_h:.2f}".rstrip("0").rstrip(".")
                )
                h24_vals.append(v_fmt)
                h24_objs.append(
                    {
                        "datetime": dt_loc.isoformat(),
                        "kwh": round(pred_h, 2),
                    }
                )

            if pred_h > 0.0:
                if d_str == today_str:
                    frac_rem = min(1.0, (slot_end - now_ts) / 3600.0)
                    rem_today += pred_h * frac_rem
                elif d_str == tomorrow_str:
                    tomorrow_total += pred_h

        self.forecast_remaining_today = round(rem_today, 2)
        self.forecast_today = round(pv_now + rem_today, 2)
        self.forecast_tomorrow = round(tomorrow_total, 2)
        if h24_objs and start_eh is not None:
            self.hourly_24h = h24_objs
            self.forecast_next_24h = round(
                sum(x["kwh"] for x in h24_objs), 2
            )
            self.series_compact_24h = f"{start_eh}|{','.join(h24_vals)}"
        self.last_updated = now_utc

    async def async_set_model_state(
        self, weights: list[float], samples: int
    ) -> None:
        """Manually seed or import model weights and sample count."""
        if len(weights) != 4:
            raise ValueError("weights must contain exactly 4 floats [w0, w1, w2, w3].")
        self.weights = [round(float(w), 6) for w in weights]
        self.samples = max(1, min(self.max_training_samples, int(samples)))
        self._evaluate_cached_slots()
        await self.async_save_state()
        self._notify_listeners()

    async def async_retrain_from_recorder(self, days: int = 30) -> int:
        """Batch-fit Ridge weights from Home Assistant Recorder hourly statistics."""
        try:
            from homeassistant.components.recorder import get_instance
            from homeassistant.components.recorder.statistics import (
                statistics_during_period,
            )
        except ImportError as err:
            _LOGGER.warning("Recorder component unavailable for batch retrain: %s", err)
            return 0

        c = self.cfg
        pv_eid = c[CONF_DAILY_PV_ENTITY]
        soc_eid = c.get(CONF_BATTERY_SOC_ENTITY)
        cld_eid = c.get(CONF_CLOUD_COVERAGE_ENTITY)
        hum_eid = c.get(CONF_HUMIDITY_ENTITY)

        stat_ids = [eid for eid in (pv_eid, soc_eid, cld_eid, hum_eid) if eid]
        end_utc = dt_util.utcnow()
        start_utc = end_utc - timedelta(days=max(3, int(days)))

        stats = await get_instance(self.hass).async_add_executor_job(
            statistics_during_period,
            self.hass,
            start_utc,
            end_utc,
            set(stat_ids),
            "hour",
            None,
            {"change", "mean", "max"},
        )

        by_ts: dict[float, dict[str, dict[str, Any]]] = {}
        for eid, rows in stats.items():
            for row in rows:
                st_ts = row.get("start")
                if st_ts is not None:
                    ts_sec = (
                        st_ts.timestamp()
                        if isinstance(st_ts, datetime)
                        else (float(st_ts) / 1000.0 if float(st_ts) > 1e11 else float(st_ts))
                    )
                    by_ts.setdefault(ts_sec, {})[eid] = row

        pv_st = self.hass.states.get(pv_eid)
        pv_unit = (pv_st.attributes.get("unit_of_measurement") or "kWh") if pv_st else "kWh"
        unit_mult = 0.001 if pv_unit == "Wh" else (1000.0 if pv_unit == "MWh" else 1.0)

        lat = self.hass.config.latitude
        lon = self.hass.config.longitude
        groups = self.panel_groups
        clip_soc = self.clip_soc_threshold
        ac_clip = max(0.3, self.inverter_ac_limit_kw - 0.2)

        X_rows: list[tuple[float, float, float, float]] = []
        y_vals: list[float] = []

        for ts_sec in sorted(by_ts.keys()):
            dt_utc = datetime.fromtimestamp(ts_sec, tz=timezone.utc)
            el_deg, az_deg, sin_el = compute_solar_position(
                dt_utc, lat, lon, midpoint_offset_hours=0.5
            )
            if el_deg <= 0.0:
                continue

            slot = by_ts[ts_sec]
            pv_row = slot.get(pv_eid)
            if not pv_row or pv_row.get("change") is None:
                continue
            pv_kwh = max(0.0, float(pv_row["change"]) * unit_mult)
            if pv_kwh >= ac_clip:
                continue

            if soc_eid and soc_eid in slot:
                s_row = slot[soc_eid]
                s_max = s_row.get("max") if s_row.get("max") is not None else s_row.get("mean")
                if s_max is not None and float(s_max) >= clip_soc:
                    continue

            if cld_eid and cld_eid in slot and slot[cld_eid].get("mean") is not None:
                clouds = float(slot[cld_eid]["mean"])
            else:
                continue

            humidity = (
                float(slot[hum_eid]["mean"])
                if (hum_eid and hum_eid in slot and slot[hum_eid].get("mean") is not None)
                else 75.0
            )

            poa_eff, sky_scale = compute_multigroup_factors(
                el_deg, az_deg, groups, REFERENCE_POWER_KW
            )
            X_rows.append(
                build_feature_vector(poa_eff, sky_scale, sin_el, clouds, humidity)
            )
            y_vals.append(pv_kwh)

        if len(X_rows) < 12:
            _LOGGER.warning(
                "Solar ML Predictor found only %d clean Recorder hours (need >= 12); keeping current weights.",
                len(X_rows),
            )
            return len(X_rows)

        self.weights = solve_ridge_4x4(X_rows, y_vals, l2_alpha=0.15)
        self.samples = min(self.max_training_samples, max(self.samples, len(X_rows)))
        self._evaluate_cached_slots()
        await self.async_save_state()
        self._notify_listeners()
        return len(X_rows)

    def l2_drift_from_prior(self) -> float:
        """Return Euclidean distance between live weights and default prior weights."""
        return round(
            math.sqrt(
                sum((w - wp) ** 2 for w, wp in zip(self.weights, PRIOR_WEIGHTS))
            ),
            4,
        )
