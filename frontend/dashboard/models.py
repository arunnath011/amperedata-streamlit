"""Dashboard models for configuration, layout, widgets, and permissions.

This module defines Pydantic models for dashboard components, including
layouts, widgets, permissions, versioning, and embedding configurations.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Union

from pydantic import BaseModel, Field

# Import visualization models
from frontend.visualization.models import ChartConfig, ExportFormat


class WidgetType(str, Enum):
    """Widget type enumeration."""

    KPI = "kpi"
    CHART = "chart"
    TABLE = "table"
    TEXT = "text"
    IMAGE = "image"
    METRIC = "metric"
    GAUGE = "gauge"
    PROGRESS = "progress"
    ALERT = "alert"
    FILTER = "filter"
    IFRAME = "iframe"
    CUSTOM = "custom"


class LayoutType(str, Enum):
    """Dashboard layout type enumeration."""

    GRID = "grid"
    TABS = "tabs"
    ACCORDION = "accordion"
    SIDEBAR = "sidebar"
    COLUMNS = "columns"
    ROWS = "rows"
    FLEXIBLE = "flexible"
    MASONRY = "masonry"


class PermissionLevel(str, Enum):
    """Permission level enumeration."""

    NONE = "none"
    VIEW = "view"
    COMMENT = "comment"
    EDIT = "edit"
    ADMIN = "admin"
    OWNER = "owner"


class DashboardRole(str, Enum):
    """Dashboard role enumeration."""

    RESEARCHER = "researcher"
    ENGINEER = "engineer"
    MANAGER = "manager"
    ANALYST = "analyst"
    VIEWER = "viewer"
    ADMIN = "admin"


class RefreshInterval(str, Enum):
    """Data refresh interval enumeration."""

    MANUAL = "manual"
    REAL_TIME = "real_time"
    SECONDS_30 = "30s"
    MINUTES_1 = "1m"
    MINUTES_5 = "5m"
    MINUTES_15 = "15m"
    MINUTES_30 = "30m"
    HOURS_1 = "1h"
    HOURS_6 = "6h"
    HOURS_12 = "12h"
    DAILY = "24h"


class EmbedType(str, Enum):
    """Embed type enumeration."""

    IFRAME = "iframe"
    JAVASCRIPT = "javascript"
    API = "api"
    SCREENSHOT = "screenshot"
    PDF = "pdf"


# Core widget models
class WidgetPosition(BaseModel):
    """Widget position in dashboard layout."""

    x: int = Field(description="X coordinate (grid units)")
    y: int = Field(description="Y coordinate (grid units)")
    width: int = Field(description="Widget width (grid units)")
    height: int = Field(description="Widget height (grid units)")
    z_index: Optional[int] = Field(None, description="Z-index for layering")


class WidgetStyle(BaseModel):
    """Widget styling configuration."""

    background_color: Optional[str] = Field(None, description="Background color")
    border_color: Optional[str] = Field(None, description="Border color")
    border_width: Optional[int] = Field(None, description="Border width in pixels")
    border_radius: Optional[int] = Field(None, description="Border radius in pixels")
    padding: Optional[int] = Field(None, description="Internal padding in pixels")
    margin: Optional[int] = Field(None, description="External margin in pixels")
    shadow: Optional[bool] = Field(None, description="Drop shadow enabled")
    opacity: Optional[float] = Field(None, ge=0.0, le=1.0, description="Widget opacity")
    font_family: Optional[str] = Field(None, description="Font family")
    font_size: Optional[int] = Field(None, description="Font size in pixels")
    font_color: Optional[str] = Field(None, description="Font color")
    text_align: Optional[str] = Field(None, description="Text alignment")


class DataSource(BaseModel):
    """Data source configuration for widgets."""

    type: str = Field(description="Data source type (database, api, file, etc.)")
    connection_string: Optional[str] = Field(None, description="Connection string")
    query: Optional[str] = Field(None, description="SQL query or API endpoint")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Query parameters")
    refresh_interval: RefreshInterval = Field(
        default=RefreshInterval.MANUAL, description="Data refresh interval"
    )
    cache_duration: Optional[int] = Field(None, description="Cache duration in seconds")
    timeout: Optional[int] = Field(None, description="Query timeout in seconds")


class WidgetConfig(BaseModel):
    """Base widget configuration."""

    title: Optional[str] = Field(None, description="Widget title")
    description: Optional[str] = Field(None, description="Widget description")
    data_source: Optional[DataSource] = Field(None, description="Data source configuration")
    refresh_interval: RefreshInterval = Field(
        default=RefreshInterval.MANUAL, description="Widget refresh interval"
    )
    show_title: bool = Field(default=True, description="Show widget title")
    show_border: bool = Field(default=True, description="Show widget border")
    interactive: bool = Field(default=True, description="Enable interactivity")
    custom_config: dict[str, Any] = Field(default_factory=dict, description="Custom configuration")


class KPIConfig(WidgetConfig):
    """KPI widget configuration."""

    value: Union[int, float, str] = Field(description="KPI value")
    unit: Optional[str] = Field(None, description="Value unit")
    trend: Optional[float] = Field(None, description="Trend percentage")
    target: Union[int, Optional[float]] = Field(None, description="Target value")
    format: Optional[str] = Field(None, description="Number format string")
    color_scheme: Optional[str] = Field(None, description="Color scheme")
    show_trend: bool = Field(default=True, description="Show trend indicator")
    show_target: bool = Field(default=False, description="Show target comparison")


class ChartWidgetConfig(WidgetConfig):
    """Chart widget configuration."""

    chart_config: ChartConfig = Field(description="Chart configuration")
    auto_refresh: bool = Field(default=False, description="Auto-refresh chart data")
    export_enabled: bool = Field(default=True, description="Enable chart export")
    fullscreen_enabled: bool = Field(default=True, description="Enable fullscreen mode")


class TableConfig(WidgetConfig):
    """Table widget configuration."""

    columns: list[dict[str, Any]] = Field(description="Table column definitions")
    sortable: bool = Field(default=True, description="Enable column sorting")
    filterable: bool = Field(default=True, description="Enable column filtering")
    searchable: bool = Field(default=True, description="Enable global search")
    pagination: bool = Field(default=True, description="Enable pagination")
    page_size: int = Field(default=25, description="Rows per page")
    export_enabled: bool = Field(default=True, description="Enable table export")
    row_selection: bool = Field(default=False, description="Enable row selection")


class TextConfig(WidgetConfig):
    """Text widget configuration."""

    content: str = Field(description="Text content (supports Markdown)")
    markdown_enabled: bool = Field(default=True, description="Enable Markdown rendering")
    html_enabled: bool = Field(default=False, description="Enable HTML rendering")
    auto_size: bool = Field(default=True, description="Auto-size text to fit widget")


class ImageConfig(WidgetConfig):
    """Image widget configuration."""

    image_url: Optional[str] = Field(None, description="Image URL")
    image_data: Optional[str] = Field(None, description="Base64 encoded image data")
    alt_text: Optional[str] = Field(None, description="Alt text for accessibility")
    fit_mode: str = Field(default="contain", description="Image fit mode")
    clickable: bool = Field(default=False, description="Make image clickable")
    click_url: Optional[str] = Field(None, description="URL to navigate on click")


class MetricConfig(WidgetConfig):
    """Metric widget configuration."""

    value: Union[int, float] = Field(description="Metric value")
    label: str = Field(description="Metric label")
    unit: Optional[str] = Field(None, description="Value unit")
    precision: int = Field(default=2, description="Decimal precision")
    color: Optional[str] = Field(None, description="Metric color")
    icon: Optional[str] = Field(None, description="Metric icon")
    trend_data: Optional[list[float]] = Field(None, description="Historical trend data")


class GaugeConfig(WidgetConfig):
    """Gauge widget configuration."""

    value: float = Field(description="Current gauge value")
    min_value: float = Field(default=0.0, description="Minimum gauge value")
    max_value: float = Field(default=100.0, description="Maximum gauge value")
    unit: Optional[str] = Field(None, description="Value unit")
    thresholds: Optional[list[dict[str, Any]]] = Field(None, description="Color thresholds")
    show_value: bool = Field(default=True, description="Show numeric value")
    show_needle: bool = Field(default=True, description="Show gauge needle")


class ProgressConfig(WidgetConfig):
    """Progress widget configuration."""

    value: float = Field(description="Current progress value")
    max_value: float = Field(default=100.0, description="Maximum progress value")
    label: Optional[str] = Field(None, description="Progress label")
    show_percentage: bool = Field(default=True, description="Show percentage")
    color: Optional[str] = Field(None, description="Progress bar color")
    animated: bool = Field(default=False, description="Animate progress bar")


class AlertConfig(WidgetConfig):
    """Alert widget configuration."""

    message: str = Field(description="Alert message")
    alert_type: str = Field(
        default="info", description="Alert type (info, warning, error, success)"
    )
    dismissible: bool = Field(default=True, description="Allow alert dismissal")
    auto_dismiss: Optional[int] = Field(None, description="Auto-dismiss after seconds")
    icon: Optional[str] = Field(None, description="Alert icon")


class FilterConfig(WidgetConfig):
    """Filter widget configuration."""

    filter_type: str = Field(description="Filter type (dropdown, slider, date, text)")
    options: Optional[list[dict[str, Any]]] = Field(None, description="Filter options")
    default_value: Optional[Any] = Field(None, description="Default filter value")
    multi_select: bool = Field(default=False, description="Allow multiple selections")
    target_widgets: list[str] = Field(description="Widget IDs to filter")


# Dashboard models
class DashboardWidget(BaseModel):
    """Dashboard widget model."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Widget ID")
    type: WidgetType = Field(description="Widget type")
    position: WidgetPosition = Field(description="Widget position")
    style: WidgetStyle = Field(default_factory=WidgetStyle, description="Widget styling")
    config: Union[
        KPIConfig,
        ChartWidgetConfig,
        TableConfig,
        TextConfig,
        ImageConfig,
        MetricConfig,
        GaugeConfig,
        ProgressConfig,
        AlertConfig,
        FilterConfig,
        WidgetConfig,
    ] = Field(description="Widget-specific configuration")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    created_by: Optional[str] = Field(None, description="Creator user ID")


class DashboardLayout(BaseModel):
    """Dashboard layout configuration."""

    type: LayoutType = Field(description="Layout type")
    grid_size: tuple[int, int] = Field(
        default=(12, 12), description="Grid dimensions (columns, rows)"
    )
    gap: int = Field(default=10, description="Gap between widgets in pixels")
    padding: int = Field(default=20, description="Layout padding in pixels")
    responsive: bool = Field(default=True, description="Enable responsive layout")
    breakpoints: dict[str, int] = Field(
        default_factory=lambda: {"sm": 576, "md": 768, "lg": 992, "xl": 1200},
        description="Responsive breakpoints",
    )
    custom_css: Optional[str] = Field(None, description="Custom CSS styles")


class DashboardConfig(BaseModel):
    """Main dashboard configuration."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Dashboard ID")
    name: str = Field(description="Dashboard name")
    description: Optional[str] = Field(None, description="Dashboard description")
    layout: DashboardLayout = Field(description="Dashboard layout configuration")
    widgets: list[DashboardWidget] = Field(default_factory=list, description="Dashboard widgets")
    theme: Optional[str] = Field(None, description="Dashboard theme")
    tags: list[str] = Field(default_factory=list, description="Dashboard tags")
    auto_refresh: bool = Field(default=False, description="Enable auto-refresh")
    refresh_interval: RefreshInterval = Field(
        default=RefreshInterval.MINUTES_5, description="Auto-refresh interval"
    )
    public: bool = Field(default=False, description="Public dashboard")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    created_by: str = Field(description="Creator user ID")
    version: int = Field(default=1, description="Dashboard version")


class DashboardPermission(BaseModel):
    """Dashboard permission model."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Permission ID")
    dashboard_id: str = Field(description="Dashboard ID")
    user_id: Optional[str] = Field(None, description="User ID (for user permissions)")
    role: Optional[DashboardRole] = Field(None, description="Role (for role-based permissions)")
    permission_level: PermissionLevel = Field(description="Permission level")
    granted_by: str = Field(description="User who granted permission")
    granted_at: datetime = Field(
        default_factory=datetime.now, description="Permission grant timestamp"
    )
    expires_at: Optional[datetime] = Field(None, description="Permission expiration")
    conditions: dict[str, Any] = Field(
        default_factory=dict, description="Additional permission conditions"
    )


class DashboardVersion(BaseModel):
    """Dashboard version model."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Version ID")
    dashboard_id: str = Field(description="Dashboard ID")
    version_number: int = Field(description="Version number")
    config: DashboardConfig = Field(description="Dashboard configuration at this version")
    change_summary: Optional[str] = Field(None, description="Summary of changes")
    created_by: str = Field(description="User who created this version")
    created_at: datetime = Field(
        default_factory=datetime.now, description="Version creation timestamp"
    )
    is_published: bool = Field(default=False, description="Is this version published")
    tags: list[str] = Field(default_factory=list, description="Version tags")


class DashboardSnapshot(BaseModel):
    """Dashboard snapshot model."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Snapshot ID")
    dashboard_id: str = Field(description="Dashboard ID")
    name: str = Field(description="Snapshot name")
    description: Optional[str] = Field(None, description="Snapshot description")
    format: ExportFormat = Field(description="Snapshot format")
    file_path: Optional[str] = Field(None, description="Snapshot file path")
    file_size: Optional[int] = Field(None, description="File size in bytes")
    created_by: str = Field(description="User who created snapshot")
    created_at: datetime = Field(
        default_factory=datetime.now, description="Snapshot creation timestamp"
    )
    scheduled: bool = Field(default=False, description="Is this a scheduled snapshot")
    schedule_config: Optional["ScheduleConfig"] = Field(None, description="Schedule configuration")


class ScheduleConfig(BaseModel):
    """Schedule configuration for snapshots."""

    enabled: bool = Field(default=True, description="Schedule enabled")
    frequency: str = Field(description="Schedule frequency (cron format)")
    timezone: str = Field(default="UTC", description="Schedule timezone")
    format: ExportFormat = Field(description="Export format")
    recipients: list[str] = Field(default_factory=list, description="Email recipients")
    webhook_url: Optional[str] = Field(None, description="Webhook URL for notifications")
    retention_days: Optional[int] = Field(None, description="Snapshot retention in days")


class EmbedConfig(BaseModel):
    """Dashboard embedding configuration."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Embed config ID")
    dashboard_id: str = Field(description="Dashboard ID")
    embed_type: EmbedType = Field(description="Embed type")
    public: bool = Field(default=False, description="Public embed")
    password_protected: bool = Field(default=False, description="Password protection")
    password: Optional[str] = Field(None, description="Embed password")
    allowed_domains: list[str] = Field(
        default_factory=list, description="Allowed embedding domains"
    )
    width: Optional[int] = Field(None, description="Embed width")
    height: Optional[int] = Field(None, description="Embed height")
    auto_refresh: bool = Field(default=True, description="Enable auto-refresh in embed")
    show_toolbar: bool = Field(default=False, description="Show dashboard toolbar")
    show_filters: bool = Field(default=True, description="Show filter widgets")
    expires_at: Optional[datetime] = Field(None, description="Embed expiration")
    created_by: str = Field(description="User who created embed")
    created_at: datetime = Field(
        default_factory=datetime.now, description="Embed creation timestamp"
    )


class DashboardTemplate(BaseModel):
    """Dashboard template model."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Template ID")
    name: str = Field(description="Template name")
    description: Optional[str] = Field(None, description="Template description")
    category: str = Field(description="Template category")
    role: Optional[DashboardRole] = Field(None, description="Target role")
    config_template: DashboardConfig = Field(description="Template configuration")
    preview_image: Optional[str] = Field(None, description="Template preview image URL")
    tags: list[str] = Field(default_factory=list, description="Template tags")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Template parameters")
    is_system: bool = Field(default=False, description="System template")
    created_by: str = Field(description="Template creator")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    usage_count: int = Field(default=0, description="Template usage count")


class DashboardShare(BaseModel):
    """Dashboard sharing model."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Share ID")
    dashboard_id: str = Field(description="Dashboard ID")
    share_token: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Share token")
    public: bool = Field(default=False, description="Public share")
    password_protected: bool = Field(default=False, description="Password protection")
    password: Optional[str] = Field(None, description="Share password")
    permission_level: PermissionLevel = Field(description="Share permission level")
    expires_at: Optional[datetime] = Field(None, description="Share expiration")
    max_views: Optional[int] = Field(None, description="Maximum view count")
    view_count: int = Field(default=0, description="Current view count")
    created_by: str = Field(description="User who created share")
    created_at: datetime = Field(
        default_factory=datetime.now, description="Share creation timestamp"
    )


# Update forward references (Pydantic v1 compatibility)
try:
    DashboardSnapshot.update_forward_refs()
except AttributeError:
    # Pydantic v2 compatibility
    try:
        DashboardSnapshot.model_rebuild()
    except AttributeError:
        pass  # No forward references to update
