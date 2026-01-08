"""Electrochemical data models for battery analysis visualizations.

This module defines Pydantic models for various types of electrochemical data
and configuration options for specialized battery analysis plots.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Union

from pydantic import BaseModel, Field, validator


class AnalysisType(str, Enum):
    """Electrochemical analysis type enumeration."""

    VOLTAGE_CAPACITY = "voltage_capacity"
    CYCLE_LIFE = "cycle_life"
    CAPACITY_FADE = "capacity_fade"
    DIFFERENTIAL = "differential"
    EIS_NYQUIST = "eis_nyquist"
    EIS_BODE = "eis_bode"
    VOLTAGE_PROFILE = "voltage_profile"
    RATE_CAPABILITY = "rate_capability"
    CALENDAR_AGING = "calendar_aging"
    BATCH_COMPARISON = "batch_comparison"


class PlotStyle(str, Enum):
    """Plot style enumeration."""

    SCIENTIFIC = "scientific"
    PUBLICATION = "publication"
    PRESENTATION = "presentation"
    INTERACTIVE = "interactive"
    MINIMAL = "minimal"


class ChargeState(str, Enum):
    """Charge/discharge state enumeration."""

    CHARGE = "charge"
    DISCHARGE = "discharge"
    REST = "rest"
    UNKNOWN = "unknown"


class TestCondition(str, Enum):
    """Test condition enumeration."""

    CONSTANT_CURRENT = "constant_current"
    CONSTANT_VOLTAGE = "constant_voltage"
    CONSTANT_POWER = "constant_power"
    PULSE = "pulse"
    CYCLE = "cycle"
    REST = "rest"
    EIS = "eis"


# Core data models
class ElectrochemicalData(BaseModel):
    """Base electrochemical data model."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Data ID")
    name: str = Field(description="Data name/identifier")
    description: Optional[str] = Field(None, description="Data description")
    cell_id: Optional[str] = Field(None, description="Cell identifier")
    experiment_id: Optional[str] = Field(None, description="Experiment identifier")
    timestamp: datetime = Field(default_factory=datetime.now, description="Data timestamp")
    temperature: Optional[float] = Field(None, description="Test temperature (°C)")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class CycleData(ElectrochemicalData):
    """Cycle data for capacity and voltage analysis."""

    cycle_number: int = Field(description="Cycle number")
    time: list[float] = Field(description="Time data (seconds)")
    voltage: list[float] = Field(description="Voltage data (V)")
    current: list[float] = Field(description="Current data (A)")
    capacity: list[float] = Field(description="Capacity data (Ah)")
    energy: Optional[list[float]] = Field(None, description="Energy data (Wh)")
    charge_state: list[ChargeState] = Field(description="Charge/discharge state")
    test_condition: TestCondition = Field(description="Test condition")
    c_rate: Optional[float] = Field(None, description="C-rate")

    @validator("voltage", "current", "capacity")
    def validate_data_length(cls, v, values):
        """Validate that all data arrays have the same length."""
        if "time" in values and len(v) != len(values["time"]):
            raise ValueError("All data arrays must have the same length")
        return v

    @validator("charge_state")
    def validate_charge_state_length(cls, v, values):
        """Validate charge state array length."""
        if "time" in values and len(v) != len(values["time"]):
            raise ValueError("Charge state array must match time array length")
        return v


class EISData(ElectrochemicalData):
    """Electrochemical Impedance Spectroscopy data."""

    frequency: list[float] = Field(description="Frequency data (Hz)")
    impedance_real: list[float] = Field(description="Real impedance (Ohm)")
    impedance_imag: list[float] = Field(description="Imaginary impedance (Ohm)")
    impedance_magnitude: Optional[list[float]] = Field(
        None, description="Impedance magnitude (Ohm)"
    )
    phase_angle: Optional[list[float]] = Field(None, description="Phase angle (degrees)")
    soc: Optional[float] = Field(None, description="State of charge (%)")
    voltage: Optional[float] = Field(None, description="Measurement voltage (V)")

    @validator("impedance_real", "impedance_imag")
    def validate_impedance_length(cls, v, values):
        """Validate impedance data length."""
        if "frequency" in values and len(v) != len(values["frequency"]):
            raise ValueError("Impedance arrays must match frequency array length")
        return v


class DifferentialData(ElectrochemicalData):
    """Differential analysis data (dQ/dV, dV/dQ)."""

    voltage: list[float] = Field(description="Voltage data (V)")
    capacity: list[float] = Field(description="Capacity data (Ah)")
    dq_dv: Optional[list[float]] = Field(None, description="dQ/dV data (Ah/V)")
    dv_dq: Optional[list[float]] = Field(None, description="dV/dQ data (V/Ah)")
    cycle_number: int = Field(description="Cycle number")
    charge_state: ChargeState = Field(description="Charge or discharge")
    smoothing_window: Optional[int] = Field(None, description="Smoothing window size")

    @validator("capacity")
    def validate_capacity_length(cls, v, values):
        """Validate capacity data length."""
        if "voltage" in values and len(v) != len(values["voltage"]):
            raise ValueError("Capacity array must match voltage array length")
        return v


class RateCapabilityData(ElectrochemicalData):
    """Rate capability test data."""

    c_rates: list[float] = Field(description="C-rates tested")
    discharge_capacity: list[float] = Field(description="Discharge capacity at each C-rate (Ah)")
    charge_capacity: Optional[list[float]] = Field(
        None, description="Charge capacity at each C-rate (Ah)"
    )
    energy_density: Optional[list[float]] = Field(None, description="Energy density (Wh/kg)")
    power_density: Optional[list[float]] = Field(None, description="Power density (W/kg)")
    efficiency: Optional[list[float]] = Field(None, description="Coulombic efficiency (%)")
    cycle_number: int = Field(description="Cycle number for rate test")

    @validator("discharge_capacity")
    def validate_capacity_length(cls, v, values):
        """Validate capacity data length."""
        if "c_rates" in values and len(v) != len(values["c_rates"]):
            raise ValueError("Capacity arrays must match C-rates array length")
        return v


class AgingData(ElectrochemicalData):
    """Calendar and cycle aging data."""

    time_days: list[float] = Field(description="Time in days")
    capacity_retention: list[float] = Field(description="Capacity retention (%)")
    resistance_increase: Optional[list[float]] = Field(None, description="Resistance increase (%)")
    cycle_count: Optional[list[int]] = Field(None, description="Cycle count at each point")
    aging_type: str = Field(description="Aging type (calendar, cycle, combined)")
    storage_temperature: Optional[float] = Field(None, description="Storage temperature (°C)")
    storage_soc: Optional[float] = Field(None, description="Storage SOC (%)")

    @validator("capacity_retention")
    def validate_retention_length(cls, v, values):
        """Validate retention data length."""
        if "time_days" in values and len(v) != len(values["time_days"]):
            raise ValueError("Capacity retention array must match time array length")
        return v


class ComparisonData(BaseModel):
    """Data for batch comparison analysis."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Comparison ID")
    name: str = Field(description="Comparison name")
    description: Optional[str] = Field(None, description="Comparison description")
    datasets: list[ElectrochemicalData] = Field(description="Datasets to compare")
    comparison_type: AnalysisType = Field(description="Type of comparison")
    grouping_variable: Optional[str] = Field(None, description="Variable to group by")
    normalization: Optional[str] = Field(None, description="Normalization method")
    statistical_analysis: bool = Field(default=False, description="Include statistical analysis")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")


# Configuration models
class ElectrochemicalConfig(BaseModel):
    """Configuration for electrochemical plots."""

    plot_style: PlotStyle = Field(default=PlotStyle.SCIENTIFIC, description="Plot style")
    color_scheme: str = Field(default="viridis", description="Color scheme")
    show_grid: bool = Field(default=True, description="Show grid lines")
    show_legend: bool = Field(default=True, description="Show legend")
    interactive: bool = Field(default=True, description="Enable interactivity")
    export_format: str = Field(default="png", description="Export format")
    dpi: int = Field(default=300, description="Export DPI")
    width: int = Field(default=800, description="Plot width (pixels)")
    height: int = Field(default=600, description="Plot height (pixels)")
    font_size: int = Field(default=12, description="Base font size")
    line_width: float = Field(default=2.0, description="Line width")
    marker_size: float = Field(default=6.0, description="Marker size")
    custom_style: dict[str, Any] = Field(default_factory=dict, description="Custom style options")


class VoltageCapacityConfig(ElectrochemicalConfig):
    """Configuration for voltage vs capacity plots."""

    show_charge: bool = Field(default=True, description="Show charge curve")
    show_discharge: bool = Field(default=True, description="Show discharge curve")
    highlight_plateaus: bool = Field(default=False, description="Highlight voltage plateaus")
    capacity_units: str = Field(default="Ah", description="Capacity units")
    voltage_range: Optional[tuple] = Field(None, description="Voltage range (V)")
    capacity_range: Optional[tuple] = Field(None, description="Capacity range")


class CycleLifeConfig(ElectrochemicalConfig):
    """Configuration for cycle life plots."""

    show_capacity: bool = Field(default=True, description="Show capacity retention")
    show_resistance: bool = Field(default=False, description="Show resistance increase")
    show_efficiency: bool = Field(default=False, description="Show coulombic efficiency")
    fit_fade_model: bool = Field(default=False, description="Fit capacity fade model")
    fade_model_type: str = Field(default="linear", description="Fade model type")
    eol_threshold: float = Field(default=80.0, description="End-of-life threshold (%)")


class DifferentialConfig(ElectrochemicalConfig):
    """Configuration for differential analysis plots."""

    analysis_type: str = Field(default="dq_dv", description="Analysis type (dq_dv, dv_dq)")
    smoothing_method: str = Field(default="savgol", description="Smoothing method")
    smoothing_window: int = Field(default=51, description="Smoothing window size")
    polynomial_order: int = Field(default=3, description="Polynomial order for smoothing")
    peak_detection: bool = Field(default=False, description="Enable peak detection")
    peak_threshold: float = Field(default=0.1, description="Peak detection threshold")


class EISConfig(ElectrochemicalConfig):
    """Configuration for EIS plots."""

    plot_type: str = Field(default="nyquist", description="Plot type (nyquist, bode)")
    frequency_range: Optional[tuple] = Field(None, description="Frequency range (Hz)")
    impedance_range: Optional[tuple] = Field(None, description="Impedance range (Ohm)")
    fit_equivalent_circuit: bool = Field(default=False, description="Fit equivalent circuit")
    circuit_model: Optional[str] = Field(None, description="Equivalent circuit model")
    show_frequency_labels: bool = Field(default=True, description="Show frequency labels")


class RateCapabilityConfig(ElectrochemicalConfig):
    """Configuration for rate capability plots."""

    show_capacity: bool = Field(default=True, description="Show capacity vs C-rate")
    show_energy: bool = Field(default=False, description="Show energy density")
    show_power: bool = Field(default=False, description="Show power density")
    show_efficiency: bool = Field(default=False, description="Show efficiency")
    normalize_capacity: bool = Field(default=False, description="Normalize to C/10 capacity")
    log_scale_rate: bool = Field(default=False, description="Use log scale for C-rate")


class AgingConfig(ElectrochemicalConfig):
    """Configuration for aging plots."""

    show_capacity: bool = Field(default=True, description="Show capacity retention")
    show_resistance: bool = Field(default=False, description="Show resistance increase")
    fit_aging_model: bool = Field(default=False, description="Fit aging model")
    aging_model_type: str = Field(default="sqrt_time", description="Aging model type")
    time_units: str = Field(default="days", description="Time units")
    extrapolate_years: Optional[int] = Field(None, description="Extrapolate to years")


class ComparisonConfig(ElectrochemicalConfig):
    """Configuration for batch comparison plots."""

    show_individual: bool = Field(default=True, description="Show individual curves")
    show_statistics: bool = Field(default=True, description="Show statistical summary")
    confidence_interval: float = Field(default=0.95, description="Confidence interval")
    grouping_colors: bool = Field(default=True, description="Use different colors for groups")
    normalize_data: bool = Field(default=False, description="Normalize data")
    statistical_test: Optional[str] = Field(None, description="Statistical test to perform")


# Data validation and processing models
class DataQuality(BaseModel):
    """Data quality assessment model."""

    completeness: float = Field(description="Data completeness (0-1)")
    consistency: float = Field(description="Data consistency (0-1)")
    accuracy: float = Field(description="Data accuracy (0-1)")
    outliers_detected: int = Field(description="Number of outliers detected")
    missing_points: int = Field(description="Number of missing data points")
    quality_score: float = Field(description="Overall quality score (0-1)")
    issues: list[str] = Field(default_factory=list, description="Data quality issues")
    recommendations: list[str] = Field(
        default_factory=list, description="Improvement recommendations"
    )


class ProcessingParameters(BaseModel):
    """Data processing parameters."""

    smoothing_enabled: bool = Field(default=False, description="Enable data smoothing")
    smoothing_method: str = Field(default="savgol", description="Smoothing method")
    smoothing_window: int = Field(default=51, description="Smoothing window size")
    outlier_removal: bool = Field(default=False, description="Remove outliers")
    outlier_threshold: float = Field(default=3.0, description="Outlier threshold (std devs)")
    interpolation_method: str = Field(default="linear", description="Interpolation method")
    resampling_rate: Optional[float] = Field(None, description="Resampling rate (Hz)")
    baseline_correction: bool = Field(default=False, description="Apply baseline correction")


class AnalysisResult(BaseModel):
    """Analysis result model."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Result ID")
    analysis_type: AnalysisType = Field(description="Type of analysis")
    input_data_id: str = Field(description="Input data ID")
    parameters: ProcessingParameters = Field(description="Processing parameters used")
    results: dict[str, Any] = Field(description="Analysis results")
    quality_assessment: DataQuality = Field(description="Data quality assessment")
    processing_time: float = Field(description="Processing time (seconds)")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


# Specialized result models
class CapacityFadeResult(AnalysisResult):
    """Capacity fade analysis result."""

    fade_rate_per_cycle: float = Field(description="Fade rate per cycle (%)")
    fade_rate_per_day: Optional[float] = Field(None, description="Fade rate per day (%)")
    eol_cycle: Optional[int] = Field(None, description="Predicted end-of-life cycle")
    eol_days: Optional[float] = Field(None, description="Predicted end-of-life days")
    model_r_squared: float = Field(description="Model R-squared value")
    model_parameters: dict[str, float] = Field(description="Model parameters")


class DifferentialResult(AnalysisResult):
    """Differential analysis result."""

    peaks_detected: list[dict[str, float]] = Field(description="Detected peaks")
    peak_positions: list[float] = Field(description="Peak voltage positions (V)")
    peak_intensities: list[float] = Field(description="Peak intensities")
    phase_transitions: list[str] = Field(description="Identified phase transitions")
    smoothing_quality: float = Field(description="Smoothing quality score")


class EISResult(AnalysisResult):
    """EIS analysis result."""

    equivalent_circuit: Optional[str] = Field(None, description="Fitted equivalent circuit")
    circuit_parameters: dict[str, float] = Field(
        default_factory=dict, description="Circuit parameters"
    )
    fit_quality: dict[str, float] = Field(default_factory=dict, description="Fit quality metrics")
    characteristic_frequencies: list[float] = Field(
        default_factory=list, description="Characteristic frequencies"
    )
    resistance_values: dict[str, float] = Field(
        default_factory=dict, description="Resistance values"
    )


class RateCapabilityResult(AnalysisResult):
    """Rate capability analysis result."""

    max_c_rate: float = Field(description="Maximum achievable C-rate")
    capacity_retention_at_1c: float = Field(description="Capacity retention at 1C (%)")
    capacity_retention_at_5c: Optional[float] = Field(
        None, description="Capacity retention at 5C (%)"
    )
    power_density_max: float = Field(description="Maximum power density (W/kg)")
    energy_density_at_1c: float = Field(description="Energy density at 1C (Wh/kg)")
    rate_capability_score: float = Field(description="Overall rate capability score (0-1)")


class AgingResult(AnalysisResult):
    """Aging analysis result."""

    aging_rate: float = Field(description="Aging rate (%/day or %/cycle)")
    calendar_life_prediction: Optional[float] = Field(
        None, description="Predicted calendar life (years)"
    )
    cycle_life_prediction: Optional[float] = Field(
        None, description="Predicted cycle life (cycles)"
    )
    dominant_aging_mechanism: str = Field(description="Dominant aging mechanism")
    aging_acceleration_factors: dict[str, float] = Field(
        default_factory=dict, description="Acceleration factors"
    )
    model_confidence: float = Field(description="Model confidence (0-1)")


class ComparisonResult(AnalysisResult):
    """Batch comparison analysis result."""

    statistical_summary: dict[str, dict[str, float]] = Field(
        description="Statistical summary by group"
    )
    significant_differences: list[dict[str, Any]] = Field(
        description="Statistically significant differences"
    )
    best_performing_sample: str = Field(description="Best performing sample ID")
    worst_performing_sample: str = Field(description="Worst performing sample ID")
    variability_metrics: dict[str, float] = Field(description="Variability metrics")
    correlation_matrix: Optional[list[list[float]]] = Field(None, description="Correlation matrix")
