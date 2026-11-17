"""Constants for the Solar ML Predictor integration."""

from __future__ import annotations

from homeassistant.const import Platform

DOMAIN = "solar_ml_predictor"
PLATFORMS: list[Platform] = [Platform.SENSOR, Platform.BUTTON]

STORAGE_VERSION = 1
FRONTEND_SCRIPT_URL = f"/{DOMAIN}/solar-ml-24h-card.js"

MAX_PANEL_GROUPS = 4
REFERENCE_POWER_KW = 8.50

# Core entity & inverter configuration keys
CONF_WEATHER_ENTITY = "weather_entity"
CONF_DAILY_PV_ENTITY = "daily_pv_entity"
CONF_BATTERY_SOC_ENTITY = "battery_soc_entity"
CONF_CLOUD_COVERAGE_ENTITY = "cloud_coverage_entity"
CONF_HUMIDITY_ENTITY = "humidity_entity"
CONF_INVERTER_AC_LIMIT_KW = "inverter_ac_limit_kw"
CONF_CLIP_SOC_THRESHOLD = "clip_soc_threshold"
CONF_MAX_TRAINING_SAMPLES = "max_training_samples"

# Up to 4 solar panel group configuration keys
CONF_GROUP_1_NAME = "group_1_name"
CONF_GROUP_1_PEAK_KW = "group_1_peak_kw"
CONF_GROUP_1_TILT = "group_1_tilt"
CONF_GROUP_1_AZIMUTH = "group_1_azimuth"

CONF_GROUP_2_NAME = "group_2_name"
CONF_GROUP_2_PEAK_KW = "group_2_peak_kw"
CONF_GROUP_2_TILT = "group_2_tilt"
CONF_GROUP_2_AZIMUTH = "group_2_azimuth"

CONF_GROUP_3_NAME = "group_3_name"
CONF_GROUP_3_PEAK_KW = "group_3_peak_kw"
CONF_GROUP_3_TILT = "group_3_tilt"
CONF_GROUP_3_AZIMUTH = "group_3_azimuth"

CONF_GROUP_4_NAME = "group_4_name"
CONF_GROUP_4_PEAK_KW = "group_4_peak_kw"
CONF_GROUP_4_TILT = "group_4_tilt"
CONF_GROUP_4_AZIMUTH = "group_4_azimuth"

PANEL_GROUP_KEYS: tuple[tuple[str, str, str, str], ...] = (
    (CONF_GROUP_1_NAME, CONF_GROUP_1_PEAK_KW, CONF_GROUP_1_TILT, CONF_GROUP_1_AZIMUTH),
    (CONF_GROUP_2_NAME, CONF_GROUP_2_PEAK_KW, CONF_GROUP_2_TILT, CONF_GROUP_2_AZIMUTH),
    (CONF_GROUP_3_NAME, CONF_GROUP_3_PEAK_KW, CONF_GROUP_3_TILT, CONF_GROUP_3_AZIMUTH),
    (CONF_GROUP_4_NAME, CONF_GROUP_4_PEAK_KW, CONF_GROUP_4_TILT, CONF_GROUP_4_AZIMUTH),
)

DEFAULT_PANEL_GROUPS: tuple[dict[str, float | str], ...] = (
    {"name": "Group 1 (Primary)", "peak_kw": 5.0, "tilt": 45.0, "azimuth": 180.0},
    {"name": "Group 2", "peak_kw": 0.0, "tilt": 45.0, "azimuth": 90.0},
    {"name": "Group 3", "peak_kw": 0.0, "tilt": 45.0, "azimuth": 270.0},
    {"name": "Group 4", "peak_kw": 0.0, "tilt": 30.0, "azimuth": 180.0},
)

DEFAULT_INVERTER_AC_LIMIT_KW = 10.0
DEFAULT_CLIP_SOC_THRESHOLD = 98.0
DEFAULT_MAX_TRAINING_SAMPLES = 4400

# Physics-informed seed weights normalized to REFERENCE_POWER_KW (8.50 kWp)
PRIOR_WEIGHTS: tuple[float, float, float, float] = (1.6201, -0.4649, 1.8363, 4.4914)
DEFAULT_INITIAL_SAMPLES = 80
PRIOR_ANCHOR_STRENGTH = 0.01

# Services
SERVICE_REFRESH_FORECAST = "refresh_forecast"
SERVICE_RETRAIN_FROM_RECORDER = "retrain_from_recorder"
SERVICE_SET_MODEL_STATE = "set_model_state"
