"""Config flow for Solar ML Predictor (supports up to 4 solar panel groups)."""

from __future__ import annotations

from typing import Any
import voluptuous as vol

from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlow,
)
from homeassistant.core import callback
from homeassistant.helpers import selector

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
    DEFAULT_INVERTER_AC_LIMIT_KW,
    DEFAULT_MAX_TRAINING_SAMPLES,
    DEFAULT_PANEL_GROUPS,
    DOMAIN,
    PANEL_GROUP_KEYS,
)


def _optional_entity_key(
    defaults: dict[str, Any], key: str
) -> vol.Optional:
    """Create an Optional entity key that preserves clearing in OptionsFlow."""
    val = defaults.get(key)
    if val:
        return vol.Optional(key, description={"suggested_value": val})
    return vol.Optional(key)


def _build_schema(defaults: dict[str, Any]) -> vol.Schema:
    """Build config/options schema with weather/inverter entities and up to 4 solar panel groups."""
    schema_dict: dict[Any, Any] = {
        vol.Required(
            CONF_WEATHER_ENTITY,
            default=defaults.get(CONF_WEATHER_ENTITY),
        ): selector.EntitySelector(
            selector.EntitySelectorConfig(domain="weather")
        ),
        vol.Required(
            CONF_DAILY_PV_ENTITY,
            default=defaults.get(CONF_DAILY_PV_ENTITY),
        ): selector.EntitySelector(
            selector.EntitySelectorConfig(domain="sensor")
        ),
        _optional_entity_key(defaults, CONF_BATTERY_SOC_ENTITY): selector.EntitySelector(
            selector.EntitySelectorConfig(domain="sensor")
        ),
        _optional_entity_key(defaults, CONF_CLOUD_COVERAGE_ENTITY): selector.EntitySelector(
            selector.EntitySelectorConfig(domain="sensor")
        ),
        _optional_entity_key(defaults, CONF_HUMIDITY_ENTITY): selector.EntitySelector(
            selector.EntitySelectorConfig(domain="sensor")
        ),
        vol.Required(
            CONF_INVERTER_AC_LIMIT_KW,
            default=defaults.get(
                CONF_INVERTER_AC_LIMIT_KW, DEFAULT_INVERTER_AC_LIMIT_KW
            ),
        ): selector.NumberSelector(
            selector.NumberSelectorConfig(
                min=0.5,
                max=100.0,
                step=0.1,
                unit_of_measurement="kW",
                mode=selector.NumberSelectorMode.BOX,
            )
        ),
        vol.Required(
            CONF_CLIP_SOC_THRESHOLD,
            default=defaults.get(
                CONF_CLIP_SOC_THRESHOLD, DEFAULT_CLIP_SOC_THRESHOLD
            ),
        ): selector.NumberSelector(
            selector.NumberSelectorConfig(
                min=50.0,
                max=100.0,
                step=0.5,
                unit_of_measurement="%",
                mode=selector.NumberSelectorMode.BOX,
            )
        ),
        vol.Required(
            CONF_MAX_TRAINING_SAMPLES,
            default=defaults.get(
                CONF_MAX_TRAINING_SAMPLES, DEFAULT_MAX_TRAINING_SAMPLES
            ),
        ): selector.NumberSelector(
            selector.NumberSelectorConfig(
                min=100,
                max=8800,
                step=100,
                unit_of_measurement="samples",
                mode=selector.NumberSelectorMode.BOX,
            )
        ),
    }

    for idx, (name_k, peak_k, tilt_k, az_k) in enumerate(PANEL_GROUP_KEYS):
        grp_def = DEFAULT_PANEL_GROUPS[idx]
        min_kw = 0.1 if idx == 0 else 0.0
        schema_dict[
            vol.Required(name_k, default=defaults.get(name_k, grp_def["name"]))
        ] = selector.TextSelector()
        schema_dict[
            vol.Required(
                peak_k, default=defaults.get(peak_k, grp_def["peak_kw"])
            )
        ] = selector.NumberSelector(
            selector.NumberSelectorConfig(
                min=min_kw,
                max=100.0,
                step=0.01,
                unit_of_measurement="kWp",
                mode=selector.NumberSelectorMode.BOX,
            )
        )
        schema_dict[
            vol.Required(tilt_k, default=defaults.get(tilt_k, grp_def["tilt"]))
        ] = selector.NumberSelector(
            selector.NumberSelectorConfig(
                min=0.0,
                max=90.0,
                step=1.0,
                unit_of_measurement="°",
                mode=selector.NumberSelectorMode.BOX,
            )
        )
        schema_dict[
            vol.Required(az_k, default=defaults.get(az_k, grp_def["azimuth"]))
        ] = selector.NumberSelector(
            selector.NumberSelectorConfig(
                min=0.0,
                max=360.0,
                step=1.0,
                unit_of_measurement="°",
                mode=selector.NumberSelectorMode.BOX,
            )
        )

    return vol.Schema(schema_dict)


def _normalize_and_validate(
    user_input: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Normalize optional selectors and verify at least one panel group has peak_kw > 0."""
    cleaned = dict(user_input)
    for opt_key in (
        CONF_BATTERY_SOC_ENTITY,
        CONF_CLOUD_COVERAGE_ENTITY,
        CONF_HUMIDITY_ENTITY,
    ):
        if not cleaned.get(opt_key):
            cleaned[opt_key] = None

    total_kw = sum(
        float(cleaned.get(peak_k, 0.0) or 0.0)
        for _name_k, peak_k, _tilt_k, _az_k in PANEL_GROUP_KEYS
    )
    errors: dict[str, str] = {}
    if total_kw <= 0.0:
        errors["base"] = "no_active_groups"
    return cleaned, errors


class SolarMlPredictorConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Solar ML Predictor."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial UI configuration step."""
        errors: dict[str, str] = {}
        if user_input is not None:
            cleaned, errors = _normalize_and_validate(user_input)
            if not errors:
                await self.async_set_unique_id(cleaned[CONF_DAILY_PV_ENTITY])
                self._abort_if_unique_id_configured()
                total_kw = sum(
                    float(cleaned.get(peak_k, 0.0) or 0.0)
                    for _, peak_k, _, _ in PANEL_GROUP_KEYS
                )
                return self.async_create_entry(
                    title=f"Solar ML Predictor ({total_kw:.2f} kWp)",
                    data=cleaned,
                )

        return self.async_show_form(
            step_id="user",
            data_schema=_build_schema(user_input or {}),
            errors=errors,
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> OptionsFlow:
        """Return the options flow."""
        return SolarMlPredictorOptionsFlow(config_entry)


class SolarMlPredictorOptionsFlow(OptionsFlow):
    """Handle options flow for Solar ML Predictor."""

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialize options flow."""
        self._config_entry = config_entry

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage the integration options (including all 4 solar panel groups)."""
        errors: dict[str, str] = {}
        if user_input is not None:
            cleaned, errors = _normalize_and_validate(user_input)
            if not errors:
                return self.async_create_entry(title="", data=cleaned)

        merged = {**self._config_entry.data, **self._config_entry.options}
        return self.async_show_form(
            step_id="init",
            data_schema=_build_schema(user_input or merged),
            errors=errors,
        )
