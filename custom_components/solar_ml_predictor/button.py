"""Button entities for manual forecast refresh and Recorder batch retraining."""

from __future__ import annotations

from homeassistant.components.button import ButtonEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN
from .coordinator import SolarMlCoordinator


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Solar ML Predictor button entities."""
    coordinator: SolarMlCoordinator = hass.data[DOMAIN][entry.entry_id]
    async_add_entities(
        [
            SolarMlRefreshForecastButton(coordinator),
            SolarMlRetrainRecorderButton(coordinator),
        ]
    )


class SolarMlRefreshForecastButton(ButtonEntity):
    """Button to trigger an immediate hourly weather forecast re-evaluation."""

    _attr_has_entity_name = True
    _attr_name = "Refresh Forecast"
    _attr_icon = "mdi:refresh"
    _attr_entity_category = EntityCategory.CONFIG

    def __init__(self, coordinator: SolarMlCoordinator) -> None:
        """Initialize Refresh Forecast button."""
        self._coordinator = coordinator
        self._attr_unique_id = f"{coordinator.entry.entry_id}_refresh_forecast"
        self._attr_device_info = coordinator.device_info

    async def async_press(self) -> None:
        """Handle button press."""
        await self._coordinator.async_refresh_forecast()


class SolarMlRetrainRecorderButton(ButtonEntity):
    """Button to batch-fit Ridge weights from 30 days of Recorder hourly statistics."""

    _attr_has_entity_name = True
    _attr_name = "Retrain from Recorder"
    _attr_icon = "mdi:database-sync"
    _attr_entity_category = EntityCategory.CONFIG

    def __init__(self, coordinator: SolarMlCoordinator) -> None:
        """Initialize Retrain from Recorder button."""
        self._coordinator = coordinator
        self._attr_unique_id = (
            f"{coordinator.entry.entry_id}_retrain_from_recorder"
        )
        self._attr_device_info = coordinator.device_info

    async def async_press(self) -> None:
        """Handle button press."""
        await self._coordinator.async_retrain_from_recorder(days=30)
