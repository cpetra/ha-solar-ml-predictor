"""Offline Training, Cross-Validation & Recorder Audit Utility for Solar ML Predictor.

Supports 1 to 4 solar panel groups (roof planes / MPPT strings) normalized to
a canonical `8.50 kWp` reference scale (`REFERENCE_POWER_KW = 8.50`) so learned
weights transfer seamlessly across installation sizes and hardware expansions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

REFERENCE_POWER_KW = 8.50
REF_TILT_COS_FACTOR = 1.0 + math.cos(math.radians(45.0))
MAX_PANEL_GROUPS = 4


@dataclass(frozen=True)
class SolarPlaneConfig:
    """Single PV roof plane / panel group geometry and optional commissioning date."""

    name: str
    tilt_deg: float
    azimuth_deg: float
    peak_power_kw: float
    commissioned_date: str | None = None  # YYYY-MM-DD local date when active (None = always active)


DEFAULT_PLANES: tuple[SolarPlaneConfig, ...] = (
    SolarPlaneConfig(
        name="Group 1",
        tilt_deg=45.0,
        azimuth_deg=180.0,
        peak_power_kw=REFERENCE_POWER_KW,
        commissioned_date=None,
    ),
)


@dataclass(frozen=True)
class SolarSiteConfig:
    """Geographic and multi-group physical configuration (up to 4 solar panel groups)."""

    latitude: float = 45.0
    longitude: float = 0.0
    reference_power_kw: float = REFERENCE_POWER_KW
    inverter_ac_limit_kw: float = 10.0
    clip_soc_threshold: float = 98.0
    planes: tuple[SolarPlaneConfig, ...] = DEFAULT_PLANES

    def __post_init__(self) -> None:
        if len(self.planes) > MAX_PANEL_GROUPS:
            raise ValueError(
                f"Up to {MAX_PANEL_GROUPS} solar panel groups are supported (got {len(self.planes)})."
            )

    @property
    def total_peak_power_kw(self) -> float:
        """Total installed DC peak power across all configured panel groups (kWp)."""
        return round(sum(p.peak_power_kw for p in self.planes if p.peak_power_kw > 0), 3)

    def active_planes(self, day_local: str | None = None) -> tuple[SolarPlaneConfig, ...]:
        """Return active panel groups (`peak_power_kw > 0`) on `day_local` ('YYYY-MM-DD')."""
        return tuple(
            p
            for p in self.planes
            if p.peak_power_kw > 0.0
            and (
                day_local is None
                or p.commissioned_date is None
                or day_local >= p.commissioned_date
            )
        )


@dataclass
class HourlySample:
    """Single daylight hourly observation combining multi-group solar geometry, weather, and PV."""

    dt_local: str
    day: str
    hour: int
    elevation_deg: float
    azimuth_deg: float
    sin_el: float
    poa_cos: float
    clouds_frac: float
    humidity_frac: float
    temperature_c: float
    pressure_hpa: float
    soc_max: float
    pv_kwh: float = 0.0
    sky_scale: float = 1.0

    def feature_vector(self) -> list[float]:
        """Physics-informed feature vector consistent across Recorder history and hourly forecasts."""
        clear = 1.0 - self.clouds_frac
        dry = max(0.0, 1.0 - self.humidity_frac)
        return [
            self.poa_cos * (clear ** 2),                           # w0: Capacity-weighted direct beam on planes
            self.sky_scale * self.sin_el * (clear ** 2),           # w1: Clear-sky horizontal / IAM term
            self.poa_cos * dry,                                    # w2: Low-humidity clear-sky plane factor
            self.sky_scale * self.sin_el * (0.25 + 0.75 * clear),  # w3: Sky diffuse + cloud-transmitted radiation
        ]


def compute_multiplane_geometry(
    dt_utc: datetime,
    site: SolarSiteConfig,
    day_local: str | None = None,
) -> tuple[float, float, float, float]:
    """Return `(elevation_deg, azimuth_deg, effective_poa, sky_scale)` at midpoint of the UTC hour.

    `effective_poa` and `sky_scale` are normalized by `site.reference_power_kw` (`8.50 kWp`)
    across up to 4 active solar panel groups.
    """
    doy = dt_utc.timetuple().tm_yday
    hour_utc = dt_utc.hour + dt_utc.minute / 60.0 + 0.5
    gamma = 2.0 * math.pi / 365.0 * (doy - 1 + (hour_utc - 12.0) / 24.0)

    eqtime = 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2.0 * gamma)
        - 0.040849 * math.sin(2.0 * gamma)
    )
    decl = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2.0 * gamma)
        + 0.000907 * math.sin(2.0 * gamma)
        - 0.002697 * math.cos(3.0 * gamma)
        + 0.001480 * math.sin(3.0 * gamma)
    )

    tst = hour_utc * 60.0 + eqtime + 4.0 * site.longitude
    ha = math.radians((tst / 4.0) - 180.0)
    lat_rad = math.radians(site.latitude)

    sin_el = (
        math.sin(lat_rad) * math.sin(decl)
        + math.cos(lat_rad) * math.cos(decl) * math.cos(ha)
    )
    el_rad = math.asin(max(-1.0, min(1.0, sin_el)))
    el_deg = math.degrees(el_rad)

    cos_el = math.cos(el_rad)
    if abs(cos_el) > 1e-6:
        sin_az = -math.cos(decl) * math.sin(ha) / cos_el
        cos_az = (math.sin(decl) - math.sin(lat_rad) * sin_el) / (
            math.cos(lat_rad) * cos_el
        )
        az_deg = (math.degrees(math.atan2(sin_az, cos_az)) + 360.0) % 360.0
    else:
        az_deg = 180.0

    planes = site.active_planes(day_local)
    if el_deg <= 0.0 or not planes:
        return el_deg, az_deg, 0.0, 0.0

    poa_eff = 0.0
    sky_scale = 0.0
    for plane in planes:
        scale_p = plane.peak_power_kw / site.reference_power_kw
        tilt_rad = math.radians(plane.tilt_deg)
        cos_aoi = (
            math.sin(el_rad) * math.cos(tilt_rad)
            + cos_el
            * math.sin(tilt_rad)
            * math.cos(math.radians(az_deg - plane.azimuth_deg))
        )
        poa_eff += scale_p * max(0.0, cos_aoi)
        sky_view = (1.0 + math.cos(tilt_rad)) / REF_TILT_COS_FACTOR
        sky_scale += scale_p * sky_view

    return el_deg, az_deg, poa_eff, sky_scale


def compute_solar_geometry(
    dt_utc: datetime,
    site: SolarSiteConfig,
    day_local: str | None = None,
) -> tuple[float, float, float]:
    """Return `(elevation_deg, azimuth_deg, effective_poa)` at midpoint of the UTC hour."""
    el_deg, az_deg, poa_eff, _ = compute_multiplane_geometry(dt_utc, site, day_local)
    return el_deg, az_deg, poa_eff


class SolarMLPredictor:
    """Physics-informed Multi-Group Ridge Regression predictor."""

    def __init__(self, site: SolarSiteConfig | None = None, l2_alpha: float = 0.15):
        self.site = site or SolarSiteConfig()
        self.l2_alpha = l2_alpha
        self.weights: np.ndarray = np.array([1.6201, -0.4649, 1.8363, 4.4914])

    def fit(self, samples: Iterable[HourlySample]) -> np.ndarray:
        """Fit model weights on unclipped daylight hours."""
        ac_clip = max(0.3, self.site.inverter_ac_limit_kw - 0.2)
        clean = [
            s
            for s in samples
            if s.elevation_deg > 0
            and s.soc_max < self.site.clip_soc_threshold
            and s.pv_kwh < ac_clip
        ]
        if not clean:
            raise ValueError("No unclipped daylight samples available for training.")
        X = np.array([s.feature_vector() for s in clean], dtype=np.float64)
        y = np.array([s.pv_kwh for s in clean], dtype=np.float64)
        reg = self.l2_alpha * np.eye(X.shape[1])
        self.weights = np.linalg.solve(X.T @ X + reg, X.T @ y)
        return self.weights

    def predict_hour(self, sample: HourlySample) -> float:
        """Predict single-hour solar generation (kWh), bounded by inverter AC rating."""
        if sample.elevation_deg <= 0.0:
            return 0.0
        x = np.array(sample.feature_vector(), dtype=np.float64)
        raw_kwh = max(0.0, float(x @ self.weights))
        return min(self.site.inverter_ac_limit_kw, raw_kwh)

    def evaluate_leave_one_day_out(
        self, samples: list[HourlySample], exclude_days: set[str] | None = None
    ) -> dict:
        """Run leave-one-day-out cross-validation and return daily & aggregate metrics."""
        exclude_days = exclude_days or set()
        by_day: dict[str, list[HourlySample]] = {}
        for s in samples:
            by_day.setdefault(s.day, []).append(s)

        results = []
        abs_errors = []
        actual_totals = []

        for test_day in sorted(by_day.keys()):
            train_samples = [
                s
                for d, hrs in by_day.items()
                if d != test_day and d not in exclude_days
                for s in hrs
            ]
            model = SolarMLPredictor(site=self.site, l2_alpha=self.l2_alpha)
            model.fit(train_samples)

            pred_kwh = sum(model.predict_hour(s) for s in by_day[test_day])
            actual_kwh = sum(s.pv_kwh for s in by_day[test_day])
            max_soc = max(s.soc_max for s in by_day[test_day])
            is_clipped = max_soc >= self.site.clip_soc_threshold or test_day in exclude_days

            results.append(
                {
                    "day": test_day,
                    "actual_kwh": round(actual_kwh, 2),
                    "predicted_kwh": round(pred_kwh, 2),
                    "error_kwh": round(pred_kwh - actual_kwh, 2),
                    "max_soc": round(max_soc, 1),
                    "clipped": is_clipped,
                }
            )
            if not is_clipped:
                abs_errors.append(abs(pred_kwh - actual_kwh))
                actual_totals.append(actual_kwh)

        mae = float(np.mean(abs_errors)) if abs_errors else 0.0
        mape = (100.0 * sum(abs_errors) / sum(actual_totals)) if sum(actual_totals) > 0 else 0.0
        return {"days": results, "mae_kwh": round(mae, 2), "wape_pct": round(mape, 1)}


def load_samples_from_ha_stats(
    json_files: list[str | Path],
    site: SolarSiteConfig | None = None,
    pv_entity_id: str = "sensor.pv_daily_energy",
    soc_entity_id: str | None = "sensor.battery_soc",
    cloud_entity_id: str = "sensor.weather_cloud_coverage",
    humidity_entity_id: str = "sensor.weather_humidity",
    temperature_entity_id: str | None = None,
    pressure_entity_id: str | None = None,
) -> list[HourlySample]:
    """Parse exported Home Assistant Recorder hourly statistics JSON files."""
    site = site or SolarSiteConfig()
    by_ts: dict[int, dict[str, dict]] = {}
    for path in json_files:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        entities = (
            raw["data"]["entities"]
            if "data" in raw and isinstance(raw["data"], dict)
            else raw["entities"]
        )
        for entity in entities:
            eid = entity["entity_id"]
            for stat in entity["statistics"]:
                by_ts.setdefault(stat["start"], {})[eid] = stat

    samples: list[HourlySample] = []
    for ts in sorted(by_ts.keys()):
        dt_utc = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc)
        dt_loc = dt_utc.astimezone()
        day_str = dt_loc.strftime("%Y-%m-%d")
        row = by_ts[ts]

        el_deg, az_deg, poa_eff, sky_scale = compute_multiplane_geometry(
            dt_utc, site, day_local=day_str
        )
        if el_deg <= 0.0:
            continue

        clouds = row.get(cloud_entity_id, {}).get("mean")
        if clouds is None:
            continue
        temp = (
            row.get(temperature_entity_id, {}).get("mean")
            if temperature_entity_id
            else 15.0
        )
        if temp is None:
            temp = 15.0

        pv_stat = row.get(pv_entity_id)
        if not pv_stat or pv_stat.get("change") is None:
            continue
        pv = max(0.0, float(pv_stat["change"]))

        soc_max = 0.0
        if soc_entity_id and soc_entity_id in row:
            soc_stat = row[soc_entity_id]
            soc_val = (
                soc_stat.get("max")
                if soc_stat.get("max") is not None
                else soc_stat.get("mean")
            )
            if soc_val is not None:
                soc_max = float(soc_val)

        humidity = row.get(humidity_entity_id, {}).get("mean") or 75.0
        pressure = (
            (row.get(pressure_entity_id, {}).get("mean") or 1013.0)
            if pressure_entity_id
            else 1013.0
        )

        samples.append(
            HourlySample(
                dt_local=dt_loc.strftime("%Y-%m-%d %H:%M"),
                day=day_str,
                hour=dt_loc.hour,
                elevation_deg=el_deg,
                azimuth_deg=az_deg,
                sin_el=math.sin(math.radians(el_deg)),
                poa_cos=poa_eff,
                clouds_frac=clouds / 100.0,
                humidity_frac=humidity / 100.0,
                temperature_c=float(temp),
                pressure_hpa=float(pressure),
                soc_max=soc_max,
                pv_kwh=pv,
                sky_scale=sky_scale,
            )
        )
    return samples
