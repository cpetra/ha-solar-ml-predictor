"""The Solar ML Predictor integration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
import voluptuous as vol

from homeassistant.components.http import StaticPathConfig
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.helpers import config_validation as cv

from .const import (
    DOMAIN,
    FRONTEND_SCRIPT_URL,
    PLATFORMS,
    SERVICE_REFRESH_FORECAST,
    SERVICE_RETRAIN_FROM_RECORDER,
    SERVICE_SET_MODEL_STATE,
)
from .coordinator import SolarMlCoordinator

_LOGGER = logging.getLogger(__name__)

DATA_STATIC_REGISTERED = "_static_registered"
DATA_SERVICES_REGISTERED = "_services_registered"


def _iter_coordinators(
    hass: HomeAssistant, entry_id: str | None
) -> list[SolarMlCoordinator]:
    """Return matching SolarMlCoordinator instances for a service call."""
    domain_data: dict[str, Any] = hass.data.get(DOMAIN, {})
    if entry_id:
        coord = domain_data.get(entry_id)
        return [coord] if isinstance(coord, SolarMlCoordinator) else []
    return [
        v for v in domain_data.values() if isinstance(v, SolarMlCoordinator)
    ]


async def _async_register_frontend(hass: HomeAssistant) -> None:
    """Serve the bundled `solar-ml-24h-card.js` custom Lovelace card."""
    domain_data = hass.data.setdefault(DOMAIN, {})
    if domain_data.get(DATA_STATIC_REGISTERED):
        return

    js_path = Path(__file__).parent / "frontend" / "solar-ml-24h-card.js"
    if js_path.exists() and hasattr(hass, "http") and hass.http is not None:
        await hass.http.async_register_static_paths(
            [
                StaticPathConfig(
                    FRONTEND_SCRIPT_URL,
                    str(js_path),
                    cache_headers=False,
                )
            ]
        )
        domain_data[DATA_STATIC_REGISTERED] = True


def _register_services(hass: HomeAssistant) -> None:
    """Register domain services once when the first config entry loads."""
    domain_data = hass.data.setdefault(DOMAIN, {})
    if domain_data.get(DATA_SERVICES_REGISTERED):
        return

    async def _handle_refresh(call: ServiceCall) -> None:
        for coord in _iter_coordinators(hass, call.data.get("entry_id")):
            await coord.async_refresh_forecast()

    async def _handle_retrain(call: ServiceCall) -> None:
        days = int(call.data.get("days", 30))
        for coord in _iter_coordinators(hass, call.data.get("entry_id")):
            await coord.async_retrain_from_recorder(days=days)

    async def _handle_set_state(call: ServiceCall) -> None:
        weights = [float(x) for x in call.data["weights"]]
        samples = int(call.data["samples"])
        for coord in _iter_coordinators(hass, call.data.get("entry_id")):
            await coord.async_set_model_state(weights=weights, samples=samples)

    hass.services.async_register(
        DOMAIN,
        SERVICE_REFRESH_FORECAST,
        _handle_refresh,
        schema=vol.Schema({vol.Optional("entry_id"): cv.string}),
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_RETRAIN_FROM_RECORDER,
        _handle_retrain,
        schema=vol.Schema(
            {
                vol.Optional("days", default=30): vol.All(
                    vol.Coerce(int), vol.Range(min=3, max=365)
                ),
                vol.Optional("entry_id"): cv.string,
            }
        ),
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_SET_MODEL_STATE,
        _handle_set_state,
        schema=vol.Schema(
            {
                vol.Required("weights"): vol.All(
                    list, vol.Length(min=4, max=4)
                ),
                vol.Required("samples"): vol.All(
                    vol.Coerce(int), vol.Range(min=1, max=8800)
                ),
                vol.Optional("entry_id"): cv.string,
            }
        ),
    )
    domain_data[DATA_SERVICES_REGISTERED] = True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Solar ML Predictor from a config entry."""
    hass.data.setdefault(DOMAIN, {})
    await _async_register_frontend(hass)
    _register_services(hass)

    coordinator = SolarMlCoordinator(hass, entry)
    await coordinator.async_load_state()
    hass.data[DOMAIN][entry.entry_id] = coordinator

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    coordinator.start()
    entry.async_on_unload(coordinator.shutdown)
    entry.async_on_unload(entry.add_update_listener(async_reload_entry))
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    unload_ok = await hass.config_entries.async_unload_platforms(
        entry, PLATFORMS
    )
    if unload_ok:
        domain_data = hass.data.get(DOMAIN, {})
        domain_data.pop(entry.entry_id, None)
        remaining = [
            v for v in domain_data.values() if isinstance(v, SolarMlCoordinator)
        ]
        if not remaining and domain_data.get(DATA_SERVICES_REGISTERED):
            hass.services.async_remove(DOMAIN, SERVICE_REFRESH_FORECAST)
            hass.services.async_remove(DOMAIN, SERVICE_RETRAIN_FROM_RECORDER)
            hass.services.async_remove(DOMAIN, SERVICE_SET_MODEL_STATE)
            domain_data[DATA_SERVICES_REGISTERED] = False
    return unload_ok


async def async_reload_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload config entry when options change."""
    await hass.config_entries.async_reload(entry.entry_id)
