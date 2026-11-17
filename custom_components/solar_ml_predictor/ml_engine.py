"""Physics-informed Multi-Group Solar ML Engine (zero external dependencies)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math

from .const import (
    PRIOR_ANCHOR_STRENGTH,
    PRIOR_WEIGHTS,
    REFERENCE_POWER_KW,
)

REF_TILT_SKY_FACTOR = 1.0 + math.cos(math.radians(45.0))


@dataclass(frozen=True)
class PanelGroupConfig:
    """Configuration for a single solar panel group (roof plane / MPPT string)."""

    name: str
    peak_power_kw: float
    tilt_deg: float
    azimuth_deg: float


def compute_solar_position(
    dt_utc: datetime,
    latitude: float,
    longitude: float,
    midpoint_offset_hours: float = 0.5,
) -> tuple[float, float, float]:
    """Compute (elevation_deg, azimuth_deg, sin_elevation) at UTC timestamp + offset."""
    doy = dt_utc.timetuple().tm_yday
    hour_utc = dt_utc.hour + dt_utc.minute / 60.0 + midpoint_offset_hours
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

    tst = hour_utc * 60.0 + eqtime + 4.0 * longitude
    ha = math.radians((tst / 4.0) - 180.0)
    lat_rad = math.radians(latitude)

    sin_el = (
        math.sin(lat_rad) * math.sin(decl)
        + math.cos(lat_rad) * math.cos(decl) * math.cos(ha)
    )
    sin_el_clamped = max(-1.0, min(1.0, sin_el))
    el_rad = math.asin(sin_el_clamped)
    el_deg = math.degrees(el_rad)

    cos_el = math.cos(el_rad)
    if abs(cos_el) > 1e-6:
        sin_az = -math.cos(decl) * math.sin(ha) / cos_el
        cos_az = (math.sin(decl) - math.sin(lat_rad) * sin_el_clamped) / (
            math.cos(lat_rad) * cos_el
        )
        az_deg = (math.degrees(math.atan2(sin_az, cos_az)) + 360.0) % 360.0
    else:
        az_deg = 180.0

    return el_deg, az_deg, sin_el_clamped


def compute_multigroup_factors(
    el_deg: float,
    az_deg: float,
    groups: list[PanelGroupConfig] | tuple[PanelGroupConfig, ...],
    reference_power_kw: float = REFERENCE_POWER_KW,
) -> tuple[float, float]:
    """Compute capacity-weighted (effective_poa, sky_scale) across up to 4 panel groups."""
    if el_deg <= 0.0 or not groups or reference_power_kw <= 0.0:
        return 0.0, 0.0

    el_rad = math.radians(el_deg)
    sin_el = math.sin(el_rad)
    cos_el = math.cos(el_rad)

    poa_eff = 0.0
    sky_scale = 0.0
    for grp in groups:
        if grp.peak_power_kw <= 0.0:
            continue
        scale_g = grp.peak_power_kw / reference_power_kw
        tilt_rad = math.radians(grp.tilt_deg)
        cos_aoi = (
            sin_el * math.cos(tilt_rad)
            + cos_el
            * math.sin(tilt_rad)
            * math.cos(math.radians(az_deg - grp.azimuth_deg))
        )
        poa_eff += scale_g * max(0.0, cos_aoi)
        sky_view = (1.0 + math.cos(tilt_rad)) / REF_TILT_SKY_FACTOR
        sky_scale += scale_g * sky_view

    return poa_eff, sky_scale


def build_feature_vector(
    poa_eff: float,
    sky_scale: float,
    sin_el: float,
    clouds_pct: float,
    humidity_pct: float,
) -> tuple[float, float, float, float]:
    """Build 4-element physics-informed feature vector from geometry and weather."""
    c_frac = max(0.0, min(1.0, clouds_pct / 100.0))
    h_frac = max(0.0, min(1.0, humidity_pct / 100.0))
    clear = 1.0 - c_frac
    dry = max(0.0, 1.0 - h_frac)
    return (
        poa_eff * clear * clear,
        sky_scale * sin_el * clear * clear,
        poa_eff * dry,
        sky_scale * sin_el * (0.25 + 0.75 * clear),
    )


def predict_hourly_kwh(
    weights: list[float] | tuple[float, ...],
    features: tuple[float, float, float, float],
    el_deg: float,
    inverter_ac_limit_kw: float,
) -> float:
    """Predict 1-hour PV production (kWh), clamped to [0.0, inverter_ac_limit_kw]."""
    if el_deg <= 0.0:
        return 0.0
    raw = sum(w * x for w, x in zip(weights, features))
    return min(inverter_ac_limit_kw, max(0.0, raw))


def online_nlms_update(
    weights: list[float],
    n_samples: int,
    max_samples: int,
    features: tuple[float, float, float, float],
    actual_kwh: float,
    inverter_ac_limit_kw: float,
    prior_weights: tuple[float, float, float, float] = PRIOR_WEIGHTS,
) -> tuple[list[float], int]:
    """Perform one Normalized LMS online step with weak L2 prior anchoring (`round(6)`)."""
    y_hat = min(
        inverter_ac_limit_kw,
        max(0.0, sum(w * x for w, x in zip(weights, features))),
    )
    err = actual_kwh - y_hat
    n_new = min(max_samples, n_samples + 1)
    norm2 = 1.0 + sum(x * x for x in features)
    eta = 1.0 / (n_new * norm2)
    new_weights = [
        round(
            w + eta * err * x + PRIOR_ANCHOR_STRENGTH * eta * (w_p - w),
            6,
        )
        for w, x, w_p in zip(weights, features, prior_weights)
    ]
    return new_weights, n_new


def solve_ridge_4x4(
    X: list[tuple[float, float, float, float]],
    y: list[float],
    l2_alpha: float = 0.15,
    prior_weights: tuple[float, float, float, float] = PRIOR_WEIGHTS,
) -> list[float]:
    """Solve `(X^T X + alpha * I) w = X^T y + alpha * w_prior` in pure Python."""
    dim = 4
    a_mat = [[0.0] * dim for _ in range(dim)]
    b_vec = [l2_alpha * prior_weights[i] for i in range(dim)]

    for i in range(dim):
        a_mat[i][i] = l2_alpha

    for row, target in zip(X, y):
        for i in range(dim):
            b_vec[i] += row[i] * target
            for j in range(dim):
                a_mat[i][j] += row[i] * row[j]

    # Gaussian elimination with partial pivoting
    aug = [a_mat[i][:] + [b_vec[i]] for i in range(dim)]
    for col in range(dim):
        pivot = max(range(col, dim), key=lambda r: abs(aug[r][col]))
        aug[col], aug[pivot] = aug[pivot], aug[col]
        denom = aug[col][col]
        if abs(denom) < 1e-12:
            return list(prior_weights)
        for j in range(col, dim + 1):
            aug[col][j] /= denom
        for r in range(dim):
            if r != col and abs(aug[r][col]) > 0.0:
                factor = aug[r][col]
                for j in range(col, dim + 1):
                    aug[r][j] -= factor * aug[col][j]

    return [round(aug[i][dim], 6) for i in range(dim)]
