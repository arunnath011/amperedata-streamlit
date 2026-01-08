"""Low-code visualization framework for battery data analysis.

This module provides a comprehensive visualization system built on Streamlit and Plotly
for creating interactive charts, dashboards, and reports for battery testing data.
"""

from .components import (
    BarChart,
    BaseChart,
    BoxPlot,
    CandlestickChart,
    ContourPlot,
    GaugeChart,
    HeatmapChart,
    HistogramChart,
    LineChart,
    SankeyDiagram,
    ScatterPlot,
    SurfacePlot,
    TreemapChart,
    ViolinPlot,
)
from .config import ChartConfigManager, StyleManager, TemplateManager, ThemeManager
from .exceptions import (
    ChartRenderingError,
    ConfigurationError,
    DataFormatError,
    ExportError,
    TemplateError,
    VisualizationError,
)
from .models import (
    AnimationConfig,
    ChartConfig,
    ChartData,
    ChartStyle,
    ChartTemplate,
    ChartType,
    DashboardConfig,
    ExportFormat,
    InteractionConfig,
    VisualizationTheme,
)
from .templates import (
    BatteryAnalysisTemplates,
    CapacityFadeTemplate,
    ComparisonTemplate,
    CyclingAnalysisTemplate,
    EfficiencyAnalysisTemplate,
    ImpedanceAnalysisTemplate,
    TrendAnalysisTemplate,
    VoltageProfileTemplate,
)
from .utils import ChartExporter, ColorPalette, DataProcessor, LayoutManager, ValidationUtils

try:
    from .dashboard import DashboardBuilder, DashboardRenderer
    from .streamlit_app import StreamlitVisualizationApp

    STREAMLIT_AVAILABLE = True
except ImportError:
    StreamlitVisualizationApp = None
    DashboardBuilder = None
    DashboardRenderer = None
    STREAMLIT_AVAILABLE = False

__all__ = [
    # Models
    "ChartType",
    "ChartConfig",
    "ChartData",
    "ChartStyle",
    "ChartTemplate",
    "DashboardConfig",
    "VisualizationTheme",
    "ExportFormat",
    "InteractionConfig",
    "AnimationConfig",
    # Components
    "BaseChart",
    "LineChart",
    "ScatterPlot",
    "BarChart",
    "HeatmapChart",
    "HistogramChart",
    "BoxPlot",
    "ViolinPlot",
    "SurfacePlot",
    "ContourPlot",
    "CandlestickChart",
    "GaugeChart",
    "TreemapChart",
    "SankeyDiagram",
    # Templates
    "BatteryAnalysisTemplates",
    "CyclingAnalysisTemplate",
    "CapacityFadeTemplate",
    "ImpedanceAnalysisTemplate",
    "VoltageProfileTemplate",
    "EfficiencyAnalysisTemplate",
    "ComparisonTemplate",
    "TrendAnalysisTemplate",
    # Configuration
    "ChartConfigManager",
    "ThemeManager",
    "TemplateManager",
    "StyleManager",
    # Utilities
    "DataProcessor",
    "ChartExporter",
    "ColorPalette",
    "LayoutManager",
    "ValidationUtils",
    # Exceptions
    "VisualizationError",
    "ChartRenderingError",
    "DataFormatError",
    "ConfigurationError",
    "ExportError",
    "TemplateError",
]

# Add Streamlit components if available
if STREAMLIT_AVAILABLE:
    __all__.extend(
        [
            "StreamlitVisualizationApp",
            "DashboardBuilder",
            "DashboardRenderer",
        ]
    )
