"""Visualization models for chart configuration and data structures.

This module defines Pydantic models for chart configurations, data structures,
themes, templates, and visualization settings.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, List, Optional, Union

from pydantic import BaseModel, Field, validator


class ChartType(str, Enum):
    """Chart type enumeration."""

    LINE = "line"
    SCATTER = "scatter"
    BAR = "bar"
    HISTOGRAM = "histogram"
    BOX = "box"
    VIOLIN = "violin"
    HEATMAP = "heatmap"
    SURFACE = "surface"
    CONTOUR = "contour"
    CANDLESTICK = "candlestick"
    GAUGE = "gauge"
    TREEMAP = "treemap"
    SANKEY = "sankey"
    FUNNEL = "funnel"
    WATERFALL = "waterfall"
    RADAR = "radar"
    POLAR = "polar"
    SUNBURST = "sunburst"


class ExportFormat(str, Enum):
    """Export format enumeration."""

    PNG = "png"
    JPEG = "jpeg"
    SVG = "svg"
    PDF = "pdf"
    HTML = "html"
    JSON = "json"


class ColorScale(str, Enum):
    """Color scale enumeration."""

    VIRIDIS = "viridis"
    PLASMA = "plasma"
    INFERNO = "inferno"
    MAGMA = "magma"
    CIVIDIS = "cividis"
    BLUES = "blues"
    GREENS = "greens"
    REDS = "reds"
    RAINBOW = "rainbow"
    JET = "jet"
    HOT = "hot"
    COOL = "cool"


class InteractionMode(str, Enum):
    """Chart interaction mode enumeration."""

    ZOOM = "zoom"
    PAN = "pan"
    SELECT = "select"
    LASSO = "lasso"
    ORBIT = "orbit"
    TURNTABLE = "turntable"


class AnimationType(str, Enum):
    """Animation type enumeration."""

    NONE = "none"
    FADE_IN = "fade_in"
    SLIDE_IN = "slide_in"
    GROW = "grow"
    BOUNCE = "bounce"
    ELASTIC = "elastic"


# Base data models
class ChartData(BaseModel):
    """Chart data model."""

    x: List[Union[str, int, float, datetime]] = Field(description="X-axis data")
    y: List[Union[int, float]] = Field(description="Y-axis data")
    z: Optional[List[Union[int, float]]] = Field(None, description="Z-axis data (for 3D charts)")
    text: Optional[List[str]] = Field(None, description="Text labels")
    hover_text: Optional[List[str]] = Field(None, description="Hover text")
    color: Optional[List[Union[str, int, float]]] = Field(None, description="Color data")
    size: Optional[List[Union[int, float]]] = Field(None, description="Size data")
    error_x: Optional[list[float]] = Field(None, description="X-axis error bars")
    error_y: Optional[list[float]] = Field(None, description="Y-axis error bars")

    @validator("y", allow_reuse=True)
    def validate_y_length(cls, v, values):
        if "x" in values and len(v) != len(values["x"]):
            raise ValueError("Y data must have same length as X data")
        return v


class ChartStyle(BaseModel):
    """Chart style configuration."""

    color: Optional[str] = Field(None, description="Primary color")
    colors: Optional[list[str]] = Field(None, description="Color palette")
    color_scale: Optional[ColorScale] = Field(None, description="Color scale for continuous data")
    opacity: float = Field(default=1.0, ge=0.0, le=1.0, description="Chart opacity")
    line_width: float = Field(default=2.0, ge=0.1, description="Line width")
    marker_size: float = Field(default=6.0, ge=0.0, description="Marker size")
    marker_symbol: str = Field(default="circle", description="Marker symbol")
    fill: Optional[str] = Field(None, description="Fill style")
    dash: Optional[str] = Field(None, description="Line dash style")

    class Config:
        use_enum_values = True


class AxisConfig(BaseModel):
    """Axis configuration model."""

    title: Optional[str] = Field(None, description="Axis title")
    range: Optional[tuple[float, float]] = Field(None, description="Axis range")
    type: str = Field(default="linear", description="Axis type (linear, log, date, category)")
    tick_format: Optional[str] = Field(None, description="Tick format string")
    tick_angle: float = Field(default=0.0, description="Tick label angle")
    show_grid: bool = Field(default=True, description="Show grid lines")
    grid_color: str = Field(default="lightgray", description="Grid color")
    grid_width: float = Field(default=1.0, description="Grid line width")
    zero_line: bool = Field(default=True, description="Show zero line")
    mirror: bool = Field(default=False, description="Mirror axis on opposite side")

    @validator("range")
    def validate_range(cls, v):
        if v is not None and len(v) != 2:
            raise ValueError("Range must be a tuple of two values")
        if v is not None and v[0] >= v[1]:
            raise ValueError("Range start must be less than end")
        return v


class LayoutConfig(BaseModel):
    """Chart layout configuration."""

    title: Optional[str] = Field(None, description="Chart title")
    width: Optional[int] = Field(None, ge=100, description="Chart width in pixels")
    height: Optional[int] = Field(None, ge=100, description="Chart height in pixels")
    margin: dict[str, int] = Field(
        default_factory=lambda: {"l": 50, "r": 50, "t": 50, "b": 50},
        description="Chart margins",
    )
    background_color: str = Field(default="white", description="Background color")
    paper_color: str = Field(default="white", description="Paper color")
    font_family: str = Field(default="Arial", description="Font family")
    font_size: int = Field(default=12, ge=8, description="Font size")
    font_color: str = Field(default="black", description="Font color")
    show_legend: bool = Field(default=True, description="Show legend")
    legend_position: str = Field(default="right", description="Legend position")

    @validator("margin")
    def validate_margin(cls, v):
        required_keys = {"l", "r", "t", "b"}
        if not required_keys.issubset(v.keys()):
            raise ValueError(f"Margin must contain keys: {required_keys}")
        return v


class InteractionConfig(BaseModel):
    """Chart interaction configuration."""

    mode: list[InteractionMode] = Field(
        default_factory=lambda: [InteractionMode.ZOOM, InteractionMode.PAN],
        description="Interaction modes",
    )
    show_tips: bool = Field(default=True, description="Show hover tooltips")
    crossfilter: bool = Field(default=False, description="Enable crossfilter")
    brush: bool = Field(default=False, description="Enable brush selection")
    double_click: str = Field(default="reset", description="Double-click action")
    scroll_zoom: bool = Field(default=True, description="Enable scroll zoom")

    class Config:
        use_enum_values = True


class AnimationConfig(BaseModel):
    """Chart animation configuration."""

    type: AnimationType = Field(default=AnimationType.NONE, description="Animation type")
    duration: int = Field(default=500, ge=0, description="Animation duration in ms")
    easing: str = Field(default="cubic-in-out", description="Animation easing function")
    frame_duration: int = Field(default=50, ge=10, description="Frame duration for transitions")
    transition_duration: int = Field(default=500, ge=0, description="Transition duration")

    class Config:
        use_enum_values = True


class ChartConfig(BaseModel):
    """Complete chart configuration."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Chart ID")
    type: ChartType = Field(description="Chart type")
    title: Optional[str] = Field(None, description="Chart title")
    data: ChartData = Field(description="Chart data")
    style: ChartStyle = Field(default_factory=ChartStyle, description="Chart style")
    x_axis: AxisConfig = Field(default_factory=AxisConfig, description="X-axis configuration")
    y_axis: AxisConfig = Field(default_factory=AxisConfig, description="Y-axis configuration")
    z_axis: Optional[AxisConfig] = Field(None, description="Z-axis configuration (for 3D charts)")
    layout: LayoutConfig = Field(default_factory=LayoutConfig, description="Layout configuration")
    interaction: InteractionConfig = Field(
        default_factory=InteractionConfig, description="Interaction configuration"
    )
    animation: AnimationConfig = Field(
        default_factory=AnimationConfig, description="Animation configuration"
    )
    custom_config: dict[str, Any] = Field(
        default_factory=dict, description="Custom configuration options"
    )

    class Config:
        use_enum_values = True


class ChartTemplate(BaseModel):
    """Chart template model."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Template ID")
    name: str = Field(description="Template name")
    description: Optional[str] = Field(None, description="Template description")
    category: str = Field(description="Template category")
    chart_type: ChartType = Field(description="Chart type")
    config_template: dict[str, Any] = Field(description="Configuration template")
    data_requirements: dict[str, Any] = Field(description="Data requirements")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Template parameters")
    preview_image: Optional[str] = Field(None, description="Preview image URL")
    tags: list[str] = Field(default_factory=list, description="Template tags")
    created_by: str = Field(description="Template creator")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    is_system: bool = Field(default=False, description="System template flag")

    class Config:
        use_enum_values = True


class VisualizationTheme(BaseModel):
    """Visualization theme model."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Theme ID")
    name: str = Field(description="Theme name")
    description: Optional[str] = Field(None, description="Theme description")
    colors: dict[str, str] = Field(description="Color definitions")
    fonts: dict[str, str] = Field(description="Font definitions")
    layout_defaults: dict[str, Any] = Field(description="Default layout settings")
    chart_defaults: dict[str, Any] = Field(description="Default chart settings")
    is_dark: bool = Field(default=False, description="Dark theme flag")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")


class DashboardConfig(BaseModel):
    """Dashboard configuration model."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Dashboard ID")
    title: str = Field(description="Dashboard title")
    description: Optional[str] = Field(None, description="Dashboard description")
    layout: str = Field(default="grid", description="Dashboard layout type")
    columns: int = Field(default=2, ge=1, le=6, description="Number of columns")
    charts: list[str] = Field(description="List of chart IDs")
    chart_positions: dict[str, dict[str, int]] = Field(
        default_factory=dict, description="Chart positions and sizes"
    )
    theme_id: Optional[str] = Field(None, description="Theme ID")
    filters: dict[str, Any] = Field(default_factory=dict, description="Dashboard filters")
    auto_refresh: bool = Field(default=False, description="Auto-refresh enabled")
    refresh_interval: int = Field(default=30, ge=5, description="Refresh interval in seconds")
    created_by: str = Field(description="Dashboard creator")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")


# Battery-specific models
class BatteryDataPoint(BaseModel):
    """Battery data point model."""

    timestamp: datetime = Field(description="Measurement timestamp")
    cycle_number: Optional[int] = Field(None, description="Cycle number")
    step_number: Optional[int] = Field(None, description="Step number")
    voltage: float = Field(description="Voltage (V)")
    current: float = Field(description="Current (A)")
    capacity: Optional[float] = Field(None, description="Capacity (Ah)")
    energy: Optional[float] = Field(None, description="Energy (Wh)")
    temperature: Optional[float] = Field(None, description="Temperature (°C)")
    power: Optional[float] = Field(None, description="Power (W)")
    resistance: Optional[float] = Field(None, description="Resistance (Ω)")
    state_of_charge: Optional[float] = Field(None, description="State of charge (%)")


class BatteryAnalysisConfig(BaseModel):
    """Battery analysis configuration."""

    analysis_type: str = Field(description="Type of analysis")
    parameters: dict[str, Any] = Field(description="Analysis parameters")
    data_filters: dict[str, Any] = Field(default_factory=dict, description="Data filters")
    time_range: Optional[tuple[datetime, datetime]] = Field(None, description="Time range filter")
    cycle_range: Optional[tuple[int, int]] = Field(None, description="Cycle range filter")
    temperature_range: Optional[tuple[float, float]] = Field(
        None, description="Temperature range filter"
    )

    @validator("time_range")
    def validate_time_range(cls, v):
        if v is not None and v[0] >= v[1]:
            raise ValueError("Start time must be before end time")
        return v

    @validator("cycle_range")
    def validate_cycle_range(cls, v):
        if v is not None and v[0] >= v[1]:
            raise ValueError("Start cycle must be less than end cycle")
        return v


# Export and sharing models
class ExportConfig(BaseModel):
    """Chart export configuration."""

    format: ExportFormat = Field(description="Export format")
    width: int = Field(default=800, ge=100, description="Export width")
    height: int = Field(default=600, ge=100, description="Export height")
    scale: float = Field(default=1.0, ge=0.1, le=10.0, description="Export scale factor")
    filename: Optional[str] = Field(None, description="Export filename")
    include_plotlyjs: bool = Field(default=True, description="Include Plotly.js in HTML export")
    config: dict[str, Any] = Field(
        default_factory=dict, description="Format-specific configuration"
    )

    class Config:
        use_enum_values = True


class ShareConfig(BaseModel):
    """Chart sharing configuration."""

    public: bool = Field(default=False, description="Public sharing enabled")
    password_protected: bool = Field(default=False, description="Password protection")
    expiration_date: Optional[datetime] = Field(None, description="Share expiration date")
    allowed_users: list[str] = Field(default_factory=list, description="Allowed user IDs")
    permissions: list[str] = Field(default_factory=list, description="Share permissions")
    embed_enabled: bool = Field(default=False, description="Embed code enabled")
    download_enabled: bool = Field(default=True, description="Download enabled")


# Real-time update models
class DataStreamConfig(BaseModel):
    """Data stream configuration for real-time updates."""

    source_type: str = Field(description="Data source type")
    connection_config: dict[str, Any] = Field(description="Connection configuration")
    update_interval: int = Field(default=1000, ge=100, description="Update interval in ms")
    buffer_size: int = Field(default=1000, ge=10, description="Data buffer size")
    auto_scale: bool = Field(default=True, description="Auto-scale axes")
    max_points: Optional[int] = Field(None, ge=1, description="Maximum points to display")
    filters: dict[str, Any] = Field(default_factory=dict, description="Data filters")


class RealtimeConfig(BaseModel):
    """Real-time chart configuration."""

    enabled: bool = Field(default=False, description="Real-time updates enabled")
    stream_config: Optional[DataStreamConfig] = Field(None, description="Data stream configuration")
    animation_frame: int = Field(default=50, ge=10, description="Animation frame duration")
    transition_duration: int = Field(default=300, ge=0, description="Transition duration")
    pause_on_hover: bool = Field(default=True, description="Pause updates on hover")
    show_controls: bool = Field(default=True, description="Show playback controls")


# Validation and metadata models
class ChartValidation(BaseModel):
    """Chart validation result."""

    is_valid: bool = Field(description="Validation result")
    errors: list[str] = Field(default_factory=list, description="Validation errors")
    warnings: list[str] = Field(default_factory=list, description="Validation warnings")
    suggestions: list[str] = Field(default_factory=list, description="Improvement suggestions")
    performance_score: float = Field(ge=0.0, le=1.0, description="Performance score")
    accessibility_score: float = Field(ge=0.0, le=1.0, description="Accessibility score")


class ChartMetadata(BaseModel):
    """Chart metadata model."""

    chart_id: str = Field(description="Chart ID")
    created_by: str = Field(description="Chart creator")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    version: str = Field(default="1.0", description="Chart version")
    tags: list[str] = Field(default_factory=list, description="Chart tags")
    category: Optional[str] = Field(None, description="Chart category")
    data_source: Optional[str] = Field(None, description="Data source information")
    usage_count: int = Field(default=0, ge=0, description="Usage count")
    last_accessed: Optional[datetime] = Field(None, description="Last access timestamp")
    is_favorite: bool = Field(default=False, description="Favorite flag")
    notes: Optional[str] = Field(None, description="User notes")
