"""Sensor entities for Solar ML Predictor."""

from __future__ import annotations

from typing import Any

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EntityCategory, UnitOfEnergy
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN, REFERENCE_POWER_KW
from .coordinator import SolarMlCoordinator


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Solar ML Predictor sensor entities."""
    coordinator: SolarMlCoordinator = hass.data[DOMAIN][entry.entry_id]
    async_add_entities(
        [
            SolarMlForecastTodaySensor(coordinator),
            SolarMlForecastRemainingSensor(coordinator),
            SolarMlForecastTomorrowSensor(coordinator),
            SolarMlForecastNext24hSensor(coordinator),
            SolarMlTrainingSamplesSensor(coordinator),
        ],
        True,
    )


class _BaseSolarMlSensor(SensorEntity):
    """Base push-updated sensor for Solar ML Predictor."""

    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, coordinator: SolarMlCoordinator) -> None:
        """Initialize base Solar ML sensor."""
        self._coordinator = coordinator
        self._attr_device_info = coordinator.device_info

    async def async_added_to_hass(self) -> None:
        """Subscribe to coordinator updates."""
        self._recalculate()
        self.async_on_remove(
            self._coordinator.add_listener(self._handle_update)
        )

    @callback
    def _handle_update(self) -> None:
        """Refresh sensor state and push to Home Assistant."""
        self._recalculate()
        self.async_write_ha_state()

    @callback
    def _recalculate(self) -> None:
        """Recalculate sensor value from coordinator."""


class SolarMlForecastTodaySensor(_BaseSolarMlSensor):
    """Full-day projected solar generation (`Produced so far + ML Remaining Today`)."""

    _attr_name = "Forecast Today"
    _attr_icon = "mdi:solar-power-variant"
    _attr_device_class = SensorDeviceClass.ENERGY
    _attr_state_class = SensorStateClass.TOTAL
    _attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
    _attr_suggested_display_precision = 2

    def __init__(self, coordinator: SolarMlCoordinator) -> None:
        """Initialize Forecast Today sensor."""
        super().__init__(coordinator)
        self._attr_unique_id = f"{coordinator.entry.entry_id}_forecast_today"

    @callback
    def _recalculate(self) -> None:
        """Update Forecast Today."""
        self._attr_native_value = self._coordinator.forecast_today


class SolarMlForecastRemainingSensor(_BaseSolarMlSensor):
    """Predicted solar generation for the remaining daylight hours of today."""

    _attr_name = "Forecast Remaining Today"
    _attr_icon = "mdi:solar-power-variant-outline"
    _attr_device_class = SensorDeviceClass.ENERGY
    _attr_state_class = SensorStateClass.TOTAL
    _attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
    _attr_suggested_display_precision = 2

    def __init__(self, coordinator: SolarMlCoordinator) -> None:
        """Initialize Forecast Remaining Today sensor."""
        super().__init__(coordinator)
        self._attr_unique_id = (
            f"{coordinator.entry.entry_id}_forecast_remaining_today"
        )

    @callback
    def _recalculate(self) -> None:
        """Update Forecast Remaining Today."""
        self._attr_native_value = self._coordinator.forecast_remaining_today


class SolarMlForecastTomorrowSensor(_BaseSolarMlSensor):
    """Predicted solar generation across all daylight hours of tomorrow."""

    _attr_name = "Forecast Tomorrow"
    _attr_icon = "mdi:weather-sunny-alert"
    _attr_device_class = SensorDeviceClass.ENERGY
    _attr_state_class = SensorStateClass.TOTAL
    _attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
    _attr_suggested_display_precision = 2

    def __init__(self, coordinator: SolarMlCoordinator) -> None:
        """Initialize Forecast Tomorrow sensor."""
        super().__init__(coordinator)
        self._attr_unique_id = f"{coordinator.entry.entry_id}_forecast_tomorrow"

    @callback
    def _recalculate(self) -> None:
        """Update Forecast Tomorrow."""
        self._attr_native_value = self._coordinator.forecast_tomorrow


class SolarMlForecastNext24hSensor(_BaseSolarMlSensor):
    """Rolling next-24-hours solar forecast total + hourly bar-chart series attributes."""

    _attr_name = "Forecast Next 24h"
    _attr_icon = "mdi:chart-bar"
    _attr_device_class = SensorDeviceClass.ENERGY
    _attr_state_class = SensorStateClass.TOTAL
    _attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
    _attr_suggested_display_precision = 2

    def __init__(self, coordinator: SolarMlCoordinator) -> None:
        """Initialize Forecast Next 24h sensor."""
        super().__init__(coordinator)
        self._attr_unique_id = f"{coordinator.entry.entry_id}_forecast_next_24h"
        self._attr_extra_state_attributes: dict[str, Any] = {}

    @callback
    def _recalculate(self) -> None:
        """Update 24h total and hourly series attributes for `custom:solar-ml-24h-card`."""
        self._attr_native_value = self._coordinator.forecast_next_24h
        self._attr_extra_state_attributes = {
            "series_compact": self._coordinator.series_compact_24h,
            "hourly_forecast": self._coordinator.hourly_24h,
            "active_groups_count": len(self._coordinator.panel_groups),
            "total_peak_power_kw": self._coordinator.total_peak_power_kw,
        }


class SolarMlTrainingSamplesSensor(_BaseSolarMlSensor):
    """Diagnostic sensor tracking learned daylight hours, weights, and panel group configuration."""

    _attr_name = "Training Samples"
    _attr_icon = "mdi:brain"
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_native_unit_of_measurement = "samples"
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_suggested_display_precision = 0

    def __init__(self, coordinator: SolarMlCoordinator) -> None:
        """Initialize Training Samples sensor."""
        super().__init__(coordinator)
        self._attr_unique_id = f"{coordinator.entry.entry_id}_training_samples"
        self._attr_extra_state_attributes: dict[str, Any] = {}

    @callback
    def _recalculate(self) -> None:
        """Update sample count and model/panel diagnostic attributes."""
        self._attr_native_value = self._coordinator.samples
        self._attr_extra_state_attributes = {
            "weights": [round(w, 6) for w in self._coordinator.weights],
            "l2_drift_from_prior": self._coordinator.l2_drift_from_prior(),
            "max_training_samples": self._coordinator.max_training_samples,
            "pv0_kwh": round(self._coordinator.pv0, 2),
            "s0_soc": round(self._coordinator.s0, 1),
            "reference_power_kw": REFERENCE_POWER_KW,
            "total_peak_power_kw": self._coordinator.total_peak_power_kw,
            "inverter_ac_limit_kw": self._coordinator.inverter_ac_limit_kw,
            "panel_groups": [
                {
                    "name": g.name,
                    "peak_power_kw": g.peak_power_kw,
                    "tilt_deg": g.tilt_deg,
                    "azimuth_deg": g.azimuth_deg,
                }
                for g in self._coordinator.panel_groups
            ],
        }
