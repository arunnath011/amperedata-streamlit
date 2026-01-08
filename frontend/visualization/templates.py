"""Battery analysis chart templates for common electrochemical visualizations.

This module provides specialized chart templates for battery testing data analysis,
including cycling analysis, capacity fade, impedance spectroscopy, and more.
"""

import logging
from typing import Optional

import pandas as pd

from .config import TemplateManager
from .exceptions import DataFormatError, TemplateError
from .models import (
    AxisConfig,
    ChartConfig,
    ChartData,
    ChartStyle,
    ChartTemplate,
    ChartType,
    LayoutConfig,
)

logger = logging.getLogger(__name__)


class BatteryAnalysisTemplates:
    """Collection of battery analysis chart templates."""

    def __init__(self):
        """Initialize battery analysis templates."""
        self.template_manager = TemplateManager()
        self._register_battery_templates()

    def _register_battery_templates(self) -> None:
        """Register all battery-specific templates."""
        templates = [
            self._create_cycling_analysis_template(),
            self._create_capacity_fade_template(),
            self._create_voltage_profile_template(),
            self._create_impedance_analysis_template(),
            self._create_efficiency_analysis_template(),
            self._create_dqdv_analysis_template(),
            self._create_power_analysis_template(),
            self._create_temperature_analysis_template(),
            self._create_comparison_template(),
            self._create_trend_analysis_template(),
            self._create_cycle_statistics_template(),
            self._create_rate_capability_template(),
        ]

        for template in templates:
            self.template_manager.create_template(template)

    def _create_cycling_analysis_template(self) -> ChartTemplate:
        """Create cycling analysis template."""
        return ChartTemplate(
            name="Cycling Analysis",
            description="Comprehensive cycling analysis with voltage, current, and capacity",
            category="Battery Analysis",
            chart_type=ChartType.LINE,
            config_template={
                "type": "line",
                "style": {
                    "line_width": 1.5,
                    "marker_size": 0,
                    "opacity": 0.8,
                    "colors": ["#1f77b4", "#ff7f0e", "#2ca02c"],
                },
                "x_axis": {
                    "title": "Time (hours)",
                    "type": "linear",
                    "show_grid": True,
                    "grid_color": "#f0f0f0",
                },
                "y_axis": {
                    "title": "Voltage (V)",
                    "type": "linear",
                    "show_grid": True,
                    "grid_color": "#f0f0f0",
                },
                "layout": {
                    "title": "Battery Cycling Analysis",
                    "show_legend": True,
                    "legend_position": "right",
                    "height": 600,
                    "margin": {"l": 60, "r": 100, "t": 60, "b": 60},
                },
                "interaction": {
                    "mode": ["zoom", "pan"],
                    "show_tips": True,
                    "crossfilter": True,
                },
            },
            data_requirements={
                "time": {"type": "array", "description": "Time values in hours"},
                "voltage": {"type": "array", "description": "Voltage values in V"},
                "current": {
                    "type": "array",
                    "optional": True,
                    "description": "Current values in A",
                },
                "capacity": {
                    "type": "array",
                    "optional": True,
                    "description": "Capacity values in Ah",
                },
                "cycle_number": {
                    "type": "array",
                    "optional": True,
                    "description": "Cycle numbers",
                },
            },
            parameters={
                "show_current": {
                    "type": "boolean",
                    "default": True,
                    "description": "Show current trace",
                },
                "show_capacity": {
                    "type": "boolean",
                    "default": True,
                    "description": "Show capacity trace",
                },
                "voltage_range": {
                    "type": "range",
                    "default": [2.5, 4.2],
                    "description": "Voltage axis range",
                },
                "color_by_cycle": {
                    "type": "boolean",
                    "default": False,
                    "description": "Color by cycle number",
                },
                "smooth_data": {
                    "type": "boolean",
                    "default": False,
                    "description": "Apply data smoothing",
                },
            },
            tags=["cycling", "voltage", "current", "capacity", "time-series"],
            created_by="system",
            is_system=True,
        )

    def _create_capacity_fade_template(self) -> ChartTemplate:
        """Create capacity fade analysis template."""
        return ChartTemplate(
            name="Capacity Fade Analysis",
            description="Track capacity degradation over cycling",
            category="Battery Analysis",
            chart_type=ChartType.SCATTER,
            config_template={
                "type": "scatter",
                "style": {"marker_size": 6.0, "opacity": 0.7, "color": "#d62728"},
                "x_axis": {
                    "title": "Cycle Number",
                    "type": "linear",
                    "show_grid": True,
                },
                "y_axis": {
                    "title": "Discharge Capacity (Ah)",
                    "type": "linear",
                    "show_grid": True,
                },
                "layout": {
                    "title": "Capacity Fade Analysis",
                    "show_legend": False,
                    "height": 500,
                },
                "custom_config": {
                    "show_trendline": True,
                    "trendline_type": "exponential",
                },
            },
            data_requirements={
                "cycle_number": {"type": "array", "description": "Cycle numbers"},
                "discharge_capacity": {
                    "type": "array",
                    "description": "Discharge capacity values",
                },
                "charge_capacity": {
                    "type": "array",
                    "optional": True,
                    "description": "Charge capacity values",
                },
            },
            parameters={
                "show_charge_capacity": {"type": "boolean", "default": False},
                "fit_model": {
                    "type": "select",
                    "options": ["linear", "exponential", "polynomial"],
                    "default": "exponential",
                },
                "capacity_threshold": {
                    "type": "number",
                    "default": 0.8,
                    "description": "Capacity retention threshold",
                },
            },
            tags=["capacity", "fade", "degradation", "cycling", "lifetime"],
            created_by="system",
            is_system=True,
        )

    def _create_voltage_profile_template(self) -> ChartTemplate:
        """Create voltage profile template."""
        return ChartTemplate(
            name="Voltage Profile",
            description="Voltage vs capacity/SOC profiles",
            category="Battery Analysis",
            chart_type=ChartType.LINE,
            config_template={
                "type": "line",
                "style": {
                    "line_width": 2.0,
                    "marker_size": 0,
                    "opacity": 0.8,
                    "colors": ["#1f77b4", "#ff7f0e"],
                },
                "x_axis": {
                    "title": "Capacity (Ah)",
                    "type": "linear",
                    "show_grid": True,
                },
                "y_axis": {"title": "Voltage (V)", "type": "linear", "show_grid": True},
                "layout": {
                    "title": "Voltage Profile",
                    "show_legend": True,
                    "height": 500,
                },
            },
            data_requirements={
                "capacity": {"type": "array", "description": "Capacity values"},
                "voltage": {"type": "array", "description": "Voltage values"},
                "charge_discharge": {
                    "type": "array",
                    "optional": True,
                    "description": "Charge/discharge indicator",
                },
            },
            parameters={
                "separate_charge_discharge": {"type": "boolean", "default": True},
                "normalize_capacity": {"type": "boolean", "default": False},
                "voltage_range": {"type": "range", "default": [2.5, 4.2]},
            },
            tags=["voltage", "profile", "capacity", "soc", "charge", "discharge"],
            created_by="system",
            is_system=True,
        )

    def _create_impedance_analysis_template(self) -> ChartTemplate:
        """Create impedance analysis template."""
        return ChartTemplate(
            name="Impedance Analysis (Nyquist Plot)",
            description="Electrochemical impedance spectroscopy visualization",
            category="Battery Analysis",
            chart_type=ChartType.SCATTER,
            config_template={
                "type": "scatter",
                "style": {"marker_size": 8.0, "opacity": 0.8, "color": "#9467bd"},
                "x_axis": {
                    "title": "Real Impedance (Ω)",
                    "type": "linear",
                    "show_grid": True,
                },
                "y_axis": {
                    "title": "-Imaginary Impedance (Ω)",
                    "type": "linear",
                    "show_grid": True,
                },
                "layout": {
                    "title": "Nyquist Plot - Impedance Analysis",
                    "show_legend": False,
                    "height": 500,
                },
                "custom_config": {
                    "equal_aspect_ratio": True,
                    "show_frequency_labels": True,
                },
            },
            data_requirements={
                "real_impedance": {
                    "type": "array",
                    "description": "Real impedance values",
                },
                "imaginary_impedance": {
                    "type": "array",
                    "description": "Imaginary impedance values",
                },
                "frequency": {
                    "type": "array",
                    "optional": True,
                    "description": "Frequency values",
                },
            },
            parameters={
                "show_frequency_labels": {"type": "boolean", "default": True},
                "color_by_frequency": {"type": "boolean", "default": False},
                "impedance_unit": {
                    "type": "select",
                    "options": ["Ω", "mΩ", "kΩ"],
                    "default": "Ω",
                },
            },
            tags=["impedance", "eis", "nyquist", "frequency", "electrochemical"],
            created_by="system",
            is_system=True,
        )

    def _create_efficiency_analysis_template(self) -> ChartTemplate:
        """Create efficiency analysis template."""
        return ChartTemplate(
            name="Coulombic Efficiency",
            description="Track coulombic and energy efficiency over cycling",
            category="Battery Analysis",
            chart_type=ChartType.LINE,
            config_template={
                "type": "line",
                "style": {
                    "line_width": 2.0,
                    "marker_size": 4.0,
                    "opacity": 0.8,
                    "colors": ["#2ca02c", "#ff7f0e"],
                },
                "x_axis": {
                    "title": "Cycle Number",
                    "type": "linear",
                    "show_grid": True,
                },
                "y_axis": {
                    "title": "Efficiency (%)",
                    "type": "linear",
                    "show_grid": True,
                    "range": [90, 100],
                },
                "layout": {
                    "title": "Coulombic Efficiency Analysis",
                    "show_legend": True,
                    "height": 500,
                },
            },
            data_requirements={
                "cycle_number": {"type": "array", "description": "Cycle numbers"},
                "coulombic_efficiency": {
                    "type": "array",
                    "description": "Coulombic efficiency values",
                },
                "energy_efficiency": {
                    "type": "array",
                    "optional": True,
                    "description": "Energy efficiency values",
                },
            },
            parameters={
                "show_energy_efficiency": {"type": "boolean", "default": True},
                "efficiency_threshold": {"type": "number", "default": 99.0},
                "moving_average": {
                    "type": "number",
                    "default": 1,
                    "description": "Moving average window",
                },
            },
            tags=["efficiency", "coulombic", "energy", "cycling"],
            created_by="system",
            is_system=True,
        )

    def _create_dqdv_analysis_template(self) -> ChartTemplate:
        """Create dQ/dV analysis template."""
        return ChartTemplate(
            name="dQ/dV Analysis",
            description="Differential capacity analysis for phase transitions",
            category="Battery Analysis",
            chart_type=ChartType.LINE,
            config_template={
                "type": "line",
                "style": {
                    "line_width": 1.5,
                    "marker_size": 0,
                    "opacity": 0.8,
                    "colors": ["#1f77b4", "#ff7f0e"],
                },
                "x_axis": {"title": "Voltage (V)", "type": "linear", "show_grid": True},
                "y_axis": {
                    "title": "dQ/dV (Ah/V)",
                    "type": "linear",
                    "show_grid": True,
                },
                "layout": {
                    "title": "Differential Capacity Analysis (dQ/dV)",
                    "show_legend": True,
                    "height": 500,
                },
            },
            data_requirements={
                "voltage": {"type": "array", "description": "Voltage values"},
                "dqdv": {"type": "array", "description": "dQ/dV values"},
                "charge_discharge": {
                    "type": "array",
                    "optional": True,
                    "description": "Charge/discharge indicator",
                },
            },
            parameters={
                "separate_charge_discharge": {"type": "boolean", "default": True},
                "smooth_data": {"type": "boolean", "default": True},
                "smoothing_window": {"type": "number", "default": 5},
            },
            tags=["dqdv", "differential", "capacity", "phase", "transitions"],
            created_by="system",
            is_system=True,
        )

    def _create_power_analysis_template(self) -> ChartTemplate:
        """Create power analysis template."""
        return ChartTemplate(
            name="Power Analysis",
            description="Power vs time or capacity analysis",
            category="Battery Analysis",
            chart_type=ChartType.LINE,
            config_template={
                "type": "line",
                "style": {
                    "line_width": 2.0,
                    "marker_size": 0,
                    "opacity": 0.8,
                    "color": "#d62728",
                },
                "x_axis": {
                    "title": "Time (hours)",
                    "type": "linear",
                    "show_grid": True,
                },
                "y_axis": {"title": "Power (W)", "type": "linear", "show_grid": True},
                "layout": {
                    "title": "Power Analysis",
                    "show_legend": False,
                    "height": 500,
                },
            },
            data_requirements={
                "time": {"type": "array", "description": "Time values"},
                "power": {"type": "array", "description": "Power values"},
                "voltage": {
                    "type": "array",
                    "optional": True,
                    "description": "Voltage values",
                },
                "current": {
                    "type": "array",
                    "optional": True,
                    "description": "Current values",
                },
            },
            parameters={
                "x_axis_type": {
                    "type": "select",
                    "options": ["time", "capacity"],
                    "default": "time",
                },
                "show_voltage_current": {"type": "boolean", "default": False},
                "power_unit": {
                    "type": "select",
                    "options": ["W", "mW", "kW"],
                    "default": "W",
                },
            },
            tags=["power", "energy", "time", "capacity"],
            created_by="system",
            is_system=True,
        )

    def _create_temperature_analysis_template(self) -> ChartTemplate:
        """Create temperature analysis template."""
        return ChartTemplate(
            name="Temperature Analysis",
            description="Temperature monitoring during cycling",
            category="Battery Analysis",
            chart_type=ChartType.LINE,
            config_template={
                "type": "line",
                "style": {
                    "line_width": 2.0,
                    "marker_size": 0,
                    "opacity": 0.8,
                    "color": "#ff7f0e",
                },
                "x_axis": {
                    "title": "Time (hours)",
                    "type": "linear",
                    "show_grid": True,
                },
                "y_axis": {
                    "title": "Temperature (°C)",
                    "type": "linear",
                    "show_grid": True,
                },
                "layout": {
                    "title": "Temperature Analysis",
                    "show_legend": False,
                    "height": 500,
                },
            },
            data_requirements={
                "time": {"type": "array", "description": "Time values"},
                "temperature": {"type": "array", "description": "Temperature values"},
                "voltage": {
                    "type": "array",
                    "optional": True,
                    "description": "Voltage values for correlation",
                },
            },
            parameters={
                "temperature_unit": {
                    "type": "select",
                    "options": ["°C", "°F", "K"],
                    "default": "°C",
                },
                "show_voltage_correlation": {"type": "boolean", "default": False},
                "temperature_limits": {"type": "range", "default": [0, 60]},
            },
            tags=["temperature", "thermal", "monitoring", "cycling"],
            created_by="system",
            is_system=True,
        )

    def _create_comparison_template(self) -> ChartTemplate:
        """Create comparison template."""
        return ChartTemplate(
            name="Battery Comparison",
            description="Compare multiple batteries or conditions",
            category="Battery Analysis",
            chart_type=ChartType.LINE,
            config_template={
                "type": "line",
                "style": {
                    "line_width": 2.0,
                    "marker_size": 4.0,
                    "opacity": 0.8,
                    "colors": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
                },
                "x_axis": {
                    "title": "Cycle Number",
                    "type": "linear",
                    "show_grid": True,
                },
                "y_axis": {
                    "title": "Discharge Capacity (Ah)",
                    "type": "linear",
                    "show_grid": True,
                },
                "layout": {
                    "title": "Battery Comparison",
                    "show_legend": True,
                    "legend_position": "right",
                    "height": 500,
                },
            },
            data_requirements={
                "cycle_number": {"type": "array", "description": "Cycle numbers"},
                "capacity": {"type": "array", "description": "Capacity values"},
                "battery_id": {"type": "array", "description": "Battery identifiers"},
            },
            parameters={
                "comparison_metric": {
                    "type": "select",
                    "options": ["capacity", "efficiency", "voltage"],
                    "default": "capacity",
                },
                "normalize_data": {"type": "boolean", "default": False},
                "show_statistics": {"type": "boolean", "default": True},
            },
            tags=["comparison", "multiple", "batteries", "analysis"],
            created_by="system",
            is_system=True,
        )

    def _create_trend_analysis_template(self) -> ChartTemplate:
        """Create trend analysis template."""
        return ChartTemplate(
            name="Trend Analysis",
            description="Long-term trend analysis with statistical indicators",
            category="Battery Analysis",
            chart_type=ChartType.SCATTER,
            config_template={
                "type": "scatter",
                "style": {"marker_size": 6.0, "opacity": 0.7, "color": "#2ca02c"},
                "x_axis": {"title": "Time (days)", "type": "linear", "show_grid": True},
                "y_axis": {
                    "title": "Performance Metric",
                    "type": "linear",
                    "show_grid": True,
                },
                "layout": {
                    "title": "Long-term Trend Analysis",
                    "show_legend": True,
                    "height": 500,
                },
                "custom_config": {
                    "show_trendline": True,
                    "show_confidence_interval": True,
                },
            },
            data_requirements={
                "time": {"type": "array", "description": "Time values"},
                "metric": {"type": "array", "description": "Performance metric values"},
                "confidence_upper": {
                    "type": "array",
                    "optional": True,
                    "description": "Upper confidence bound",
                },
                "confidence_lower": {
                    "type": "array",
                    "optional": True,
                    "description": "Lower confidence bound",
                },
            },
            parameters={
                "trend_method": {
                    "type": "select",
                    "options": ["linear", "polynomial", "exponential"],
                    "default": "linear",
                },
                "confidence_level": {"type": "number", "default": 0.95},
                "show_outliers": {"type": "boolean", "default": True},
            },
            tags=["trend", "statistics", "long-term", "analysis"],
            created_by="system",
            is_system=True,
        )

    def _create_cycle_statistics_template(self) -> ChartTemplate:
        """Create cycle statistics template."""
        return ChartTemplate(
            name="Cycle Statistics",
            description="Statistical analysis of cycling data",
            category="Battery Analysis",
            chart_type=ChartType.BOX,
            config_template={
                "type": "box",
                "style": {"opacity": 0.7, "colors": ["#1f77b4", "#ff7f0e", "#2ca02c"]},
                "x_axis": {
                    "title": "Cycle Range",
                    "type": "category",
                    "show_grid": False,
                },
                "y_axis": {
                    "title": "Capacity (Ah)",
                    "type": "linear",
                    "show_grid": True,
                },
                "layout": {
                    "title": "Cycle Statistics",
                    "show_legend": False,
                    "height": 500,
                },
                "custom_config": {"boxpoints": "outliers", "show_mean": True},
            },
            data_requirements={
                "cycle_range": {
                    "type": "array",
                    "description": "Cycle range categories",
                },
                "capacity": {"type": "array", "description": "Capacity values"},
                "efficiency": {
                    "type": "array",
                    "optional": True,
                    "description": "Efficiency values",
                },
            },
            parameters={
                "statistic_type": {
                    "type": "select",
                    "options": ["capacity", "efficiency", "voltage"],
                    "default": "capacity",
                },
                "cycle_grouping": {
                    "type": "number",
                    "default": 50,
                    "description": "Cycles per group",
                },
                "show_outliers": {"type": "boolean", "default": True},
            },
            tags=["statistics", "box-plot", "distribution", "cycling"],
            created_by="system",
            is_system=True,
        )

    def _create_rate_capability_template(self) -> ChartTemplate:
        """Create rate capability template."""
        return ChartTemplate(
            name="Rate Capability",
            description="Battery performance at different C-rates",
            category="Battery Analysis",
            chart_type=ChartType.BAR,
            config_template={
                "type": "bar",
                "style": {"opacity": 0.8, "color": "#9467bd"},
                "x_axis": {"title": "C-Rate", "type": "category", "show_grid": False},
                "y_axis": {
                    "title": "Discharge Capacity (Ah)",
                    "type": "linear",
                    "show_grid": True,
                },
                "layout": {
                    "title": "Rate Capability Analysis",
                    "show_legend": False,
                    "height": 500,
                },
            },
            data_requirements={
                "c_rate": {"type": "array", "description": "C-rate values"},
                "capacity": {
                    "type": "array",
                    "description": "Discharge capacity at each C-rate",
                },
                "capacity_retention": {
                    "type": "array",
                    "optional": True,
                    "description": "Capacity retention percentage",
                },
            },
            parameters={
                "show_retention": {"type": "boolean", "default": True},
                "normalize_capacity": {"type": "boolean", "default": False},
                "reference_rate": {
                    "type": "select",
                    "options": ["C/10", "C/5", "1C"],
                    "default": "C/10",
                },
            },
            tags=["rate", "capability", "c-rate", "performance"],
            created_by="system",
            is_system=True,
        )

    def get_template(self, name: str) -> Optional[ChartTemplate]:
        """Get template by name.

        Args:
            name: Template name

        Returns:
            Chart template or None if not found
        """
        templates = self.template_manager.list_templates()
        for template in templates:
            if template.name.lower() == name.lower():
                return template
        return None

    def list_templates(self) -> list[ChartTemplate]:
        """List all battery analysis templates.

        Returns:
            List of templates
        """
        return [
            t for t in self.template_manager.list_templates() if t.category == "Battery Analysis"
        ]

    def search_templates(self, query: str) -> list[ChartTemplate]:
        """Search templates by query.

        Args:
            query: Search query

        Returns:
            List of matching templates
        """
        return self.template_manager.search_templates(query)


# Specific template classes for advanced functionality
class CyclingAnalysisTemplate:
    """Advanced cycling analysis template with data processing."""

    @staticmethod
    def create_config_from_data(
        df: pd.DataFrame,
        time_col: str = "time",
        voltage_col: str = "voltage",
        current_col: str = "current",
        capacity_col: Optional[str] = None,
        cycle_col: Optional[str] = None,
    ) -> ChartConfig:
        """Create cycling analysis configuration from DataFrame.

        Args:
            df: Input DataFrame
            time_col: Time column name
            voltage_col: Voltage column name
            current_col: Current column name
            capacity_col: Capacity column name
            cycle_col: Cycle number column name

        Returns:
            Chart configuration
        """
        try:
            # Validate required columns
            required_cols = [time_col, voltage_col, current_col]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise DataFormatError(f"Missing required columns: {missing_cols}")

            # Prepare data
            chart_data = ChartData(
                x=df[time_col].tolist(),
                y=df[voltage_col].tolist(),
                text=[
                    f"V: {v:.3f}V, I: {i:.3f}A" for v, i in zip(df[voltage_col], df[current_col])
                ],
            )

            # Create configuration
            config = ChartConfig(
                type=ChartType.LINE,
                title="Battery Cycling Analysis",
                data=chart_data,
                style=ChartStyle(line_width=1.5, marker_size=0, opacity=0.8, color="#1f77b4"),
                x_axis=AxisConfig(title="Time (hours)", type="linear", show_grid=True),
                y_axis=AxisConfig(title="Voltage (V)", type="linear", show_grid=True),
                layout=LayoutConfig(title="Battery Cycling Analysis", height=600, show_legend=True),
            )

            return config

        except Exception as e:
            logger.error(f"Failed to create cycling analysis config: {str(e)}")
            raise TemplateError(f"Configuration creation failed: {str(e)}")


class CapacityFadeTemplate:
    """Advanced capacity fade analysis template."""

    @staticmethod
    def create_config_from_data(
        df: pd.DataFrame,
        cycle_col: str = "cycle_number",
        capacity_col: str = "discharge_capacity",
        fit_model: str = "exponential",
    ) -> ChartConfig:
        """Create capacity fade configuration from DataFrame.

        Args:
            df: Input DataFrame
            cycle_col: Cycle number column name
            capacity_col: Capacity column name
            fit_model: Fitting model type

        Returns:
            Chart configuration
        """
        try:
            # Validate columns
            if cycle_col not in df.columns or capacity_col not in df.columns:
                raise DataFormatError(f"Required columns not found: {cycle_col}, {capacity_col}")

            # Calculate capacity retention
            initial_capacity = df[capacity_col].iloc[0]
            retention = (df[capacity_col] / initial_capacity * 100).tolist()

            chart_data = ChartData(
                x=df[cycle_col].tolist(),
                y=retention,
                text=[f"Cycle {c}: {r:.1f}%" for c, r in zip(df[cycle_col], retention)],
            )

            config = ChartConfig(
                type=ChartType.SCATTER,
                title="Capacity Fade Analysis",
                data=chart_data,
                style=ChartStyle(marker_size=6.0, opacity=0.7, color="#d62728"),
                x_axis=AxisConfig(title="Cycle Number", type="linear", show_grid=True),
                y_axis=AxisConfig(
                    title="Capacity Retention (%)",
                    type="linear",
                    show_grid=True,
                    range=(80, 100),
                ),
                layout=LayoutConfig(title="Capacity Fade Analysis", height=500),
                custom_config={"fit_model": fit_model, "show_trendline": True},
            )

            return config

        except Exception as e:
            logger.error(f"Failed to create capacity fade config: {str(e)}")
            raise TemplateError(f"Configuration creation failed: {str(e)}")


class ImpedanceAnalysisTemplate:
    """Advanced impedance analysis template."""

    @staticmethod
    def create_nyquist_config(
        df: pd.DataFrame,
        real_col: str = "real_impedance",
        imag_col: str = "imaginary_impedance",
        freq_col: Optional[str] = None,
    ) -> ChartConfig:
        """Create Nyquist plot configuration.

        Args:
            df: Input DataFrame
            real_col: Real impedance column name
            imag_col: Imaginary impedance column name
            freq_col: Frequency column name

        Returns:
            Chart configuration
        """
        try:
            # Validate columns
            required_cols = [real_col, imag_col]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise DataFormatError(f"Missing required columns: {missing_cols}")

            # Prepare hover text
            if freq_col and freq_col in df.columns:
                hover_text = [f"f: {f:.2e} Hz" for f in df[freq_col]]
            else:
                hover_text = None

            chart_data = ChartData(
                x=df[real_col].tolist(),
                y=(-df[imag_col]).tolist(),  # Negative imaginary for conventional Nyquist
                hover_text=hover_text,
            )

            config = ChartConfig(
                type=ChartType.SCATTER,
                title="Nyquist Plot - Impedance Analysis",
                data=chart_data,
                style=ChartStyle(marker_size=8.0, opacity=0.8, color="#9467bd"),
                x_axis=AxisConfig(title="Real Impedance (Ω)", type="linear", show_grid=True),
                y_axis=AxisConfig(title="-Imaginary Impedance (Ω)", type="linear", show_grid=True),
                layout=LayoutConfig(
                    title="Nyquist Plot - Impedance Analysis",
                    height=500,
                    width=500,  # Square aspect ratio
                ),
                custom_config={
                    "equal_aspect_ratio": True,
                    "show_frequency_labels": freq_col is not None,
                },
            )

            return config

        except Exception as e:
            logger.error(f"Failed to create impedance analysis config: {str(e)}")
            raise TemplateError(f"Configuration creation failed: {str(e)}")


class VoltageProfileTemplate:
    """Advanced voltage profile template."""

    @staticmethod
    def create_config_from_data(
        df: pd.DataFrame,
        capacity_col: str = "capacity",
        voltage_col: str = "voltage",
        charge_discharge_col: Optional[str] = None,
    ) -> ChartConfig:
        """Create voltage profile configuration.

        Args:
            df: Input DataFrame
            capacity_col: Capacity column name
            voltage_col: Voltage column name
            charge_discharge_col: Charge/discharge indicator column

        Returns:
            Chart configuration
        """
        try:
            # Validate columns
            required_cols = [capacity_col, voltage_col]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise DataFormatError(f"Missing required columns: {missing_cols}")

            chart_data = ChartData(x=df[capacity_col].tolist(), y=df[voltage_col].tolist())

            config = ChartConfig(
                type=ChartType.LINE,
                title="Voltage Profile",
                data=chart_data,
                style=ChartStyle(line_width=2.0, marker_size=0, opacity=0.8, color="#1f77b4"),
                x_axis=AxisConfig(title="Capacity (Ah)", type="linear", show_grid=True),
                y_axis=AxisConfig(title="Voltage (V)", type="linear", show_grid=True),
                layout=LayoutConfig(title="Voltage Profile", height=500),
                custom_config={"separate_charge_discharge": charge_discharge_col is not None},
            )

            return config

        except Exception as e:
            logger.error(f"Failed to create voltage profile config: {str(e)}")
            raise TemplateError(f"Configuration creation failed: {str(e)}")


class EfficiencyAnalysisTemplate:
    """Advanced efficiency analysis template."""

    @staticmethod
    def create_config_from_data(
        df: pd.DataFrame,
        cycle_col: str = "cycle_number",
        coulombic_eff_col: str = "coulombic_efficiency",
        energy_eff_col: Optional[str] = None,
    ) -> ChartConfig:
        """Create efficiency analysis configuration.

        Args:
            df: Input DataFrame
            cycle_col: Cycle number column name
            coulombic_eff_col: Coulombic efficiency column name
            energy_eff_col: Energy efficiency column name

        Returns:
            Chart configuration
        """
        try:
            # Validate columns
            required_cols = [cycle_col, coulombic_eff_col]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise DataFormatError(f"Missing required columns: {missing_cols}")

            chart_data = ChartData(
                x=df[cycle_col].tolist(),
                y=(df[coulombic_eff_col] * 100).tolist(),  # Convert to percentage
                text=[f"CE: {ce:.2f}%" for ce in df[coulombic_eff_col] * 100],
            )

            config = ChartConfig(
                type=ChartType.LINE,
                title="Coulombic Efficiency Analysis",
                data=chart_data,
                style=ChartStyle(line_width=2.0, marker_size=4.0, opacity=0.8, color="#2ca02c"),
                x_axis=AxisConfig(title="Cycle Number", type="linear", show_grid=True),
                y_axis=AxisConfig(
                    title="Efficiency (%)",
                    type="linear",
                    show_grid=True,
                    range=(95, 100),
                ),
                layout=LayoutConfig(title="Coulombic Efficiency Analysis", height=500),
                custom_config={"show_energy_efficiency": energy_eff_col is not None},
            )

            return config

        except Exception as e:
            logger.error(f"Failed to create efficiency analysis config: {str(e)}")
            raise TemplateError(f"Configuration creation failed: {str(e)}")


class ComparisonTemplate:
    """Advanced comparison template."""

    @staticmethod
    def create_multi_battery_config(
        data_dict: dict[str, pd.DataFrame],
        x_col: str,
        y_col: str,
        battery_names: Optional[list[str]] = None,
    ) -> ChartConfig:
        """Create multi-battery comparison configuration.

        Args:
            data_dict: Dictionary of battery_id -> DataFrame
            x_col: X-axis column name
            y_col: Y-axis column name
            battery_names: Custom battery names

        Returns:
            Chart configuration
        """
        try:
            if not data_dict:
                raise DataFormatError("No data provided for comparison")

            # Combine all data
            all_x = []
            all_y = []
            all_text = []

            colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

            for i, (battery_id, df) in enumerate(data_dict.items()):
                if x_col not in df.columns or y_col not in df.columns:
                    continue

                battery_name = (
                    battery_names[i] if battery_names and i < len(battery_names) else battery_id
                )

                all_x.extend(df[x_col].tolist())
                all_y.extend(df[y_col].tolist())
                all_text.extend([f"{battery_name}: {y:.3f}" for y in df[y_col]])

            chart_data = ChartData(x=all_x, y=all_y, text=all_text)

            config = ChartConfig(
                type=ChartType.LINE,
                title="Battery Comparison",
                data=chart_data,
                style=ChartStyle(
                    line_width=2.0,
                    marker_size=4.0,
                    opacity=0.8,
                    colors=colors[: len(data_dict)],
                ),
                x_axis=AxisConfig(
                    title=x_col.replace("_", " ").title(), type="linear", show_grid=True
                ),
                y_axis=AxisConfig(
                    title=y_col.replace("_", " ").title(), type="linear", show_grid=True
                ),
                layout=LayoutConfig(title="Battery Comparison", height=500, show_legend=True),
            )

            return config

        except Exception as e:
            logger.error(f"Failed to create comparison config: {str(e)}")
            raise TemplateError(f"Configuration creation failed: {str(e)}")


class TrendAnalysisTemplate:
    """Advanced trend analysis template."""

    @staticmethod
    def create_config_with_statistics(
        df: pd.DataFrame, x_col: str, y_col: str, confidence_level: float = 0.95
    ) -> ChartConfig:
        """Create trend analysis configuration with statistical indicators.

        Args:
            df: Input DataFrame
            x_col: X-axis column name
            y_col: Y-axis column name
            confidence_level: Confidence level for intervals

        Returns:
            Chart configuration
        """
        try:
            # Validate columns
            if x_col not in df.columns or y_col not in df.columns:
                raise DataFormatError(f"Required columns not found: {x_col}, {y_col}")

            chart_data = ChartData(
                x=df[x_col].tolist(),
                y=df[y_col].tolist(),
                text=[f"{x}: {y:.3f}" for x, y in zip(df[x_col], df[y_col])],
            )

            config = ChartConfig(
                type=ChartType.SCATTER,
                title="Long-term Trend Analysis",
                data=chart_data,
                style=ChartStyle(marker_size=6.0, opacity=0.7, color="#2ca02c"),
                x_axis=AxisConfig(
                    title=x_col.replace("_", " ").title(), type="linear", show_grid=True
                ),
                y_axis=AxisConfig(
                    title=y_col.replace("_", " ").title(), type="linear", show_grid=True
                ),
                layout=LayoutConfig(title="Long-term Trend Analysis", height=500),
                custom_config={
                    "show_trendline": True,
                    "confidence_level": confidence_level,
                    "show_confidence_interval": True,
                },
            )

            return config

        except Exception as e:
            logger.error(f"Failed to create trend analysis config: {str(e)}")
            raise TemplateError(f"Configuration creation failed: {str(e)}")
