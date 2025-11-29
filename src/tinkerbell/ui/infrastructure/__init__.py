# Copyright 2025 Tinkerbell contributors
# SPDX-License-Identifier: Apache-2.0

"""
Infrastructure Layer - Adapters for external systems and cross-cutting concerns.

This package provides adapters that connect the application/domain layers
to external infrastructure such as:

- Settings management (SettingsAdapter)
- Telemetry and analytics (TelemetryAdapter)
- Tool registration and wiring (ToolAdapter)
- Workspace bridge operations (BridgeAdapter)

All adapters follow the pattern of wrapping external systems while emitting
events to the EventBus for loose coupling with the rest of the application.
"""

from __future__ import annotations

from tinkerbell.ui.infrastructure.bridge_adapter import BridgeAdapter
from tinkerbell.ui.infrastructure.settings_adapter import SettingsAdapter
from tinkerbell.ui.infrastructure.telemetry_adapter import TelemetryAdapter
from tinkerbell.ui.infrastructure.tool_adapter import ToolAdapter

__all__: list[str] = [
    "BridgeAdapter",
    "SettingsAdapter",
    "TelemetryAdapter",
    "ToolAdapter",
]
