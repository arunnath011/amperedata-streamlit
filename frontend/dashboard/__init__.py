"""Dashboard builder framework for battery data analysis.

This module provides a comprehensive dashboard creation and management system
built on top of the visualization framework. It includes drag-and-drop interface,
widget library, templates, sharing, versioning, and embedding capabilities.
"""

from .builder import DashboardBuilder, DashboardRenderer, LayoutManager, WidgetManager
from .embedding import APIEndpointGenerator, EmbedManager, IFrameGenerator
from .exceptions import (
    DashboardError,
    EmbedError,
    LayoutError,
    PermissionError,
    VersionError,
    WidgetError,
)
from .models import (
    DashboardConfig,
    DashboardLayout,
    DashboardPermission,
    DashboardRole,
    DashboardSnapshot,
    DashboardTemplate,
    DashboardVersion,
    DashboardWidget,
    EmbedConfig,
    LayoutType,
    PermissionLevel,
    ScheduleConfig,
    WidgetType,
)
from .permissions import AccessController, DashboardPermissionManager, ShareManager
from .storage import DashboardStorage, SnapshotManager, VersionManager
from .templates import BatteryAnalysisTemplates, DashboardTemplateManager, RoleBasedTemplates
from .widgets import (
    AlertWidget,
    BaseWidget,
    ChartWidget,
    FilterWidget,
    GaugeWidget,
    ImageWidget,
    KPIWidget,
    MetricWidget,
    ProgressWidget,
    TableWidget,
    TextWidget,
    create_widget,
)

# Try to import Streamlit components (optional)
try:
    from .streamlit_dashboard import StreamlitDashboardApp

    STREAMLIT_AVAILABLE = True
except ImportError:
    StreamlitDashboardApp = None
    STREAMLIT_AVAILABLE = False

__all__ = [
    # Core models
    "DashboardLayout",
    "DashboardWidget",
    "DashboardConfig",
    "DashboardTemplate",
    "DashboardPermission",
    "DashboardVersion",
    "DashboardSnapshot",
    "WidgetType",
    "LayoutType",
    "PermissionLevel",
    "DashboardRole",
    "EmbedConfig",
    "ScheduleConfig",
    # Widget system
    "BaseWidget",
    "KPIWidget",
    "ChartWidget",
    "TableWidget",
    "TextWidget",
    "ImageWidget",
    "MetricWidget",
    "GaugeWidget",
    "ProgressWidget",
    "AlertWidget",
    "FilterWidget",
    "create_widget",
    # Builder components
    "DashboardBuilder",
    "LayoutManager",
    "WidgetManager",
    "DashboardRenderer",
    # Templates
    "DashboardTemplateManager",
    "BatteryAnalysisTemplates",
    "RoleBasedTemplates",
    # Permissions
    "DashboardPermissionManager",
    "ShareManager",
    "AccessController",
    # Storage
    "DashboardStorage",
    "VersionManager",
    "SnapshotManager",
    # Embedding
    "EmbedManager",
    "IFrameGenerator",
    "APIEndpointGenerator",
    # Exceptions
    "DashboardError",
    "LayoutError",
    "WidgetError",
    "PermissionError",
    "VersionError",
    "EmbedError",
]

# Add Streamlit components if available
if STREAMLIT_AVAILABLE:
    __all__.append("StreamlitDashboardApp")
