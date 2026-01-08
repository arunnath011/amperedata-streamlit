"""Electrochemical visualization library for battery data analysis.

This module provides specialized visualizations for electrochemical data analysis
including voltage profiles, capacity fade, differential analysis, EIS, and more.
"""

from .exceptions import ComparisonError, DataProcessingError, ElectrochemicalError, PlottingError
from .models import (
    AgingData,
    AnalysisType,
    ComparisonData,
    CycleData,
    DifferentialData,
    EISData,
    ElectrochemicalConfig,
    ElectrochemicalData,
    PlotStyle,
    RateCapabilityData,
)
from .processors import (
    AgingAnalyzer,
    ComparisonAnalyzer,
    CycleAnalyzer,
    DifferentialAnalyzer,
    EISAnalyzer,
    ElectrochemicalProcessor,
    RateAnalyzer,
)
from .visualizations import (
    BatchComparisonPlot,
    BodePlot,
    CalendarAgingPlot,
    CapacityFadePlot,
    CycleLifePlot,
    DifferentialPlot,
    NyquistPlot,
    RateCapabilityPlot,
    VoltageCapacityPlot,
    VoltageProfilePlot,
    create_electrochemical_plot,
)

# Try to import Plotly (optional)
try:
    import plotly.express as px
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    go = None
    px = None
    PLOTLY_AVAILABLE = False

# Try to import scientific libraries (optional)
try:
    import numpy as np
    import scipy.optimize
    import scipy.signal

    SCIPY_AVAILABLE = True
except ImportError:
    np = None
    scipy = None
    SCIPY_AVAILABLE = False

__all__ = [
    # Core models
    "ElectrochemicalData",
    "CycleData",
    "EISData",
    "DifferentialData",
    "RateCapabilityData",
    "AgingData",
    "ComparisonData",
    "ElectrochemicalConfig",
    "PlotStyle",
    "AnalysisType",
    # Data processors
    "ElectrochemicalProcessor",
    "CycleAnalyzer",
    "DifferentialAnalyzer",
    "EISAnalyzer",
    "RateAnalyzer",
    "AgingAnalyzer",
    "ComparisonAnalyzer",
    # Visualization classes
    "VoltageCapacityPlot",
    "CycleLifePlot",
    "CapacityFadePlot",
    "DifferentialPlot",
    "NyquistPlot",
    "BodePlot",
    "VoltageProfilePlot",
    "RateCapabilityPlot",
    "CalendarAgingPlot",
    "BatchComparisonPlot",
    "create_electrochemical_plot",
    # Exceptions
    "ElectrochemicalError",
    "DataProcessingError",
    "PlottingError",
    "ComparisonError",
    # Availability flags
    "PLOTLY_AVAILABLE",
    "SCIPY_AVAILABLE",
]
