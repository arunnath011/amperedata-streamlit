"""Electrochemical visualization classes for battery data analysis.

This module provides specialized visualization classes for various types of
electrochemical analysis including voltage profiles, capacity fade, differential
analysis, EIS, rate capability, aging, and batch comparisons.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

# Try to import Plotly (optional)
try:
    import plotly.colors as pc
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    go = None
    px = None
    make_subplots = None
    pc = None
    PLOTLY_AVAILABLE = False

from .exceptions import PlottingError
from .models import (
    AgingConfig,
    AgingData,
    AnalysisType,
    ChargeState,
    ComparisonConfig,
    ComparisonData,
    CycleData,
    CycleLifeConfig,
    DifferentialConfig,
    DifferentialData,
    EISConfig,
    EISData,
    ElectrochemicalConfig,
    ElectrochemicalData,
    PlotStyle,
    RateCapabilityConfig,
    RateCapabilityData,
    VoltageCapacityConfig,
)
from .processors import (
    AgingAnalyzer,
    ComparisonAnalyzer,
    CycleAnalyzer,
    DifferentialAnalyzer,
    EISAnalyzer,
    RateAnalyzer,
)

logger = logging.getLogger(__name__)


class ElectrochemicalPlot(ABC):
    """Abstract base class for electrochemical plots."""

    def __init__(self, config: ElectrochemicalConfig):
        """Initialize plot with configuration.

        Args:
            config: Plot configuration
        """
        self.config = config
        self._figure = None
        self._data = None

    @abstractmethod
    def create_plot(self, data: ElectrochemicalData) -> Any:
        """Create the plot.

        Args:
            data: Electrochemical data

        Returns:
            Plot figure
        """

    def validate_data(self, data: ElectrochemicalData) -> bool:
        """Validate input data for plotting.

        Args:
            data: Data to validate

        Returns:
            True if data is valid

        Raises:
            PlottingError: If data is invalid
        """
        if not data:
            raise PlottingError("No data provided")

        if not hasattr(data, "id") or not data.id:
            raise PlottingError("Data missing required ID")

        return True

    def apply_style(self, figure: Any) -> Any:
        """Apply styling to the figure.

        Args:
            figure: Plot figure

        Returns:
            Styled figure
        """
        if not PLOTLY_AVAILABLE or not figure:
            return figure

        # Apply plot style
        if self.config.plot_style == PlotStyle.SCIENTIFIC:
            figure.update_layout(
                template="plotly_white",
                font={"family": "Arial", "size": self.config.font_size},
                showlegend=self.config.show_legend,
                width=self.config.width,
                height=self.config.height,
            )
        elif self.config.plot_style == PlotStyle.PUBLICATION:
            figure.update_layout(
                template="simple_white",
                font={"family": "Times New Roman", "size": self.config.font_size},
                showlegend=self.config.show_legend,
                width=self.config.width,
                height=self.config.height,
                plot_bgcolor="white",
                paper_bgcolor="white",
            )
        elif self.config.plot_style == PlotStyle.PRESENTATION:
            figure.update_layout(
                template="plotly_dark",
                font={"family": "Arial", "size": self.config.font_size + 2},
                showlegend=self.config.show_legend,
                width=self.config.width,
                height=self.config.height,
            )

        # Apply grid settings
        figure.update_xaxes(showgrid=self.config.show_grid, gridwidth=1, gridcolor="lightgray")
        figure.update_yaxes(showgrid=self.config.show_grid, gridwidth=1, gridcolor="lightgray")

        return figure

    def export_plot(self, figure: Any, filename: str, format: str = "png") -> bool:
        """Export plot to file.

        Args:
            figure: Plot figure
            filename: Output filename
            format: Export format

        Returns:
            True if export successful
        """
        if not PLOTLY_AVAILABLE or not figure:
            logger.warning("Cannot export plot: Plotly not available")
            return False

        try:
            if format.lower() == "png":
                figure.write_image(
                    filename,
                    format="png",
                    width=self.config.width,
                    height=self.config.height,
                    scale=self.config.dpi / 100,
                )
            elif format.lower() == "pdf":
                figure.write_image(
                    filename,
                    format="pdf",
                    width=self.config.width,
                    height=self.config.height,
                )
            elif format.lower() == "html":
                figure.write_html(filename)
            else:
                logger.error(f"Unsupported export format: {format}")
                return False

            return True

        except Exception as e:
            logger.error(f"Plot export failed: {str(e)}")
            return False


class VoltageCapacityPlot(ElectrochemicalPlot):
    """Voltage vs. capacity plot for battery cycling data."""

    def __init__(self, config: VoltageCapacityConfig):
        super().__init__(config)
        self.vc_config = config

    def create_plot(self, data: CycleData) -> Any:
        """Create voltage vs capacity plot.

        Args:
            data: Cycle data

        Returns:
            Plotly figure
        """
        self.validate_data(data)

        if not PLOTLY_AVAILABLE:
            raise PlottingError("Plotly not available for plotting")

        if not data.voltage or not data.capacity:
            raise PlottingError("Missing voltage or capacity data")

        try:
            fig = go.Figure()

            voltage = np.array(data.voltage)
            capacity = np.array(data.capacity)
            charge_state = (
                data.charge_state if data.charge_state else [ChargeState.UNKNOWN] * len(voltage)
            )

            # Separate charge and discharge data
            charge_mask = np.array([state == ChargeState.CHARGE for state in charge_state])
            discharge_mask = np.array([state == ChargeState.DISCHARGE for state in charge_state])

            # Plot charge curve
            if self.vc_config.show_charge and np.any(charge_mask):
                fig.add_trace(
                    go.Scatter(
                        x=capacity[charge_mask],
                        y=voltage[charge_mask],
                        mode="lines+markers",
                        name="Charge",
                        line={"color": "red", "width": self.config.line_width},
                        marker={"size": self.config.marker_size / 2},
                        hovertemplate="<b>Charge</b><br>Capacity: %{x:.3f} "
                        + self.vc_config.capacity_units
                        + "<br>Voltage: %{y:.3f} V<extra></extra>",
                    )
                )

            # Plot discharge curve
            if self.vc_config.show_discharge and np.any(discharge_mask):
                fig.add_trace(
                    go.Scatter(
                        x=capacity[discharge_mask],
                        y=voltage[discharge_mask],
                        mode="lines+markers",
                        name="Discharge",
                        line={"color": "blue", "width": self.config.line_width},
                        marker={"size": self.config.marker_size / 2},
                        hovertemplate="<b>Discharge</b><br>Capacity: %{x:.3f} "
                        + self.vc_config.capacity_units
                        + "<br>Voltage: %{y:.3f} V<extra></extra>",
                    )
                )

            # Highlight voltage plateaus if requested
            if self.vc_config.highlight_plateaus:
                self._highlight_plateaus(fig, voltage, capacity, charge_state)

            # Update layout
            fig.update_layout(
                title=f"Voltage vs Capacity - Cycle {data.cycle_number}",
                xaxis_title=f"Capacity ({self.vc_config.capacity_units})",
                yaxis_title="Voltage (V)",
                hovermode="closest",
            )

            # Apply axis ranges if specified
            if self.vc_config.capacity_range:
                fig.update_xaxes(range=self.vc_config.capacity_range)
            if self.vc_config.voltage_range:
                fig.update_yaxes(range=self.vc_config.voltage_range)

            # Apply styling
            fig = self.apply_style(fig)

            self._figure = fig
            return fig

        except Exception as e:
            logger.error(f"Voltage-capacity plot creation failed: {str(e)}")
            raise PlottingError(f"Plot creation failed: {str(e)}") from e

    def _highlight_plateaus(
        self,
        fig: Any,
        voltage: np.ndarray,
        capacity: np.ndarray,
        charge_state: list[ChargeState],
    ) -> None:
        """Highlight voltage plateaus in the plot."""
        try:
            # Simple plateau detection based on voltage derivative
            dv_dq = np.abs(np.diff(voltage) / np.diff(capacity))
            plateau_threshold = np.percentile(dv_dq, 10)  # Bottom 10% of derivatives

            plateau_indices = np.where(dv_dq < plateau_threshold)[0]

            if len(plateau_indices) > 0:
                # Add plateau regions as filled areas
                for _i, idx in enumerate(plateau_indices):
                    if idx < len(capacity) - 1:
                        fig.add_shape(
                            type="rect",
                            x0=capacity[idx],
                            x1=capacity[idx + 1],
                            y0=min(voltage),
                            y1=max(voltage),
                            fillcolor="yellow",
                            opacity=0.2,
                            layer="below",
                            line_width=0,
                        )
        except Exception as e:
            logger.warning(f"Plateau highlighting failed: {str(e)}")


class CycleLifePlot(ElectrochemicalPlot):
    """Cycle life and capacity fade plot."""

    def __init__(self, config: CycleLifeConfig):
        super().__init__(config)
        self.cl_config = config

    def create_plot(self, data: list[CycleData]) -> Any:
        """Create cycle life plot.

        Args:
            data: List of cycle data

        Returns:
            Plotly figure
        """
        if not PLOTLY_AVAILABLE:
            raise PlottingError("Plotly not available for plotting")

        if not data:
            raise PlottingError("No cycle data provided")

        try:
            # Extract cycle numbers and capacities
            cycles = []
            capacities = []

            for cycle_data in data:
                if cycle_data.capacity:
                    cycles.append(cycle_data.cycle_number)
                    # Use discharge capacity (last capacity value in discharge state)
                    discharge_capacity = self._extract_discharge_capacity(cycle_data)
                    capacities.append(discharge_capacity)

            if not cycles:
                raise PlottingError("No valid capacity data found")

            cycles = np.array(cycles)
            capacities = np.array(capacities)

            # Calculate capacity retention
            initial_capacity = capacities[0] if len(capacities) > 0 else 1.0
            capacity_retention = (capacities / initial_capacity) * 100

            fig = go.Figure()

            # Plot capacity retention
            if self.cl_config.show_capacity:
                fig.add_trace(
                    go.Scatter(
                        x=cycles,
                        y=capacity_retention,
                        mode="lines+markers",
                        name="Capacity Retention",
                        line={"color": "blue", "width": self.config.line_width},
                        marker={"size": self.config.marker_size},
                        hovertemplate="<b>Cycle %{x}</b><br>Retention: %{y:.1f}%<extra></extra>",
                    )
                )

            # Add EOL threshold line
            if self.cl_config.eol_threshold:
                fig.add_hline(
                    y=self.cl_config.eol_threshold,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"EOL ({self.cl_config.eol_threshold}%)",
                )

            # Fit fade model if requested
            if self.cl_config.fit_fade_model and len(cycles) > 2:
                self._add_fade_model(fig, cycles, capacity_retention)

            # Update layout
            fig.update_layout(
                title="Cycle Life and Capacity Fade",
                xaxis_title="Cycle Number",
                yaxis_title="Capacity Retention (%)",
                hovermode="x unified",
            )

            # Apply styling
            fig = self.apply_style(fig)

            self._figure = fig
            return fig

        except Exception as e:
            logger.error(f"Cycle life plot creation failed: {str(e)}")
            raise PlottingError(f"Plot creation failed: {str(e)}") from e

    def _extract_discharge_capacity(self, cycle_data: CycleData) -> float:
        """Extract discharge capacity from cycle data."""
        if not cycle_data.capacity or not cycle_data.charge_state:
            return cycle_data.capacity[-1] if cycle_data.capacity else 0.0

        # Find discharge capacity (maximum capacity during discharge)
        discharge_capacities = []
        for i, state in enumerate(cycle_data.charge_state):
            if state == ChargeState.DISCHARGE and i < len(cycle_data.capacity):
                discharge_capacities.append(cycle_data.capacity[i])

        return max(discharge_capacities) if discharge_capacities else cycle_data.capacity[-1]

    def _add_fade_model(self, fig: Any, cycles: np.ndarray, capacity_retention: np.ndarray) -> None:
        """Add capacity fade model to the plot."""
        try:
            if self.cl_config.fade_model_type == "linear":
                # Linear fit
                coeffs = np.polyfit(cycles, capacity_retention, 1)
                model_y = np.polyval(coeffs, cycles)

                fig.add_trace(
                    go.Scatter(
                        x=cycles,
                        y=model_y,
                        mode="lines",
                        name="Linear Fit",
                        line={"color": "red", "dash": "dash", "width": 2},
                        hovertemplate="<b>Linear Model</b><br>Cycle %{x}<br>Predicted: %{y:.1f}%<extra></extra>",
                    )
                )

        except Exception as e:
            logger.warning(f"Fade model fitting failed: {str(e)}")


class CapacityFadePlot(CycleLifePlot):
    """Specialized capacity fade analysis plot."""

    def create_plot(self, data: list[CycleData]) -> Any:
        """Create capacity fade plot with detailed analysis."""
        fig = super().create_plot(data)

        if not fig:
            return fig

        try:
            # Add fade rate annotation
            if len(data) > 1:
                analyzer = CycleAnalyzer()
                # Use first cycle data for analysis (would need aggregated data in practice)
                result = analyzer.process(data[0])

                fig.add_annotation(
                    x=0.02,
                    y=0.98,
                    xref="paper",
                    yref="paper",
                    text=f"Fade Rate: {result.fade_rate_per_cycle:.3f}%/cycle",
                    showarrow=False,
                    bgcolor="white",
                    bordercolor="black",
                    borderwidth=1,
                )

            return fig

        except Exception as e:
            logger.warning(f"Fade analysis annotation failed: {str(e)}")
            return fig


class DifferentialPlot(ElectrochemicalPlot):
    """Differential capacity (dQ/dV) and voltage (dV/dQ) plot."""

    def __init__(self, config: DifferentialConfig):
        super().__init__(config)
        self.diff_config = config

    def create_plot(self, data: DifferentialData) -> Any:
        """Create differential analysis plot.

        Args:
            data: Differential data

        Returns:
            Plotly figure
        """
        self.validate_data(data)

        if not PLOTLY_AVAILABLE:
            raise PlottingError("Plotly not available for plotting")

        try:
            # Process data using analyzer
            analyzer = DifferentialAnalyzer()
            result = analyzer.process(data)

            # Create subplot figure
            fig = make_subplots(
                rows=2,
                cols=1,
                subplot_titles=("dQ/dV Analysis", "dV/dQ Analysis"),
                vertical_spacing=0.1,
            )

            voltage_points = result.results.get("voltage_points", [])
            dq_dv_data = result.results.get("dq_dv_data", [])
            dv_dq_data = result.results.get("dv_dq_data", [])

            # Plot dQ/dV
            if self.diff_config.analysis_type in ["dq_dv", "both"] and dq_dv_data:
                fig.add_trace(
                    go.Scatter(
                        x=voltage_points,
                        y=dq_dv_data,
                        mode="lines",
                        name="dQ/dV",
                        line={"color": "blue", "width": self.config.line_width},
                        hovertemplate="<b>dQ/dV</b><br>Voltage: %{x:.3f} V<br>dQ/dV: %{y:.3f} Ah/V<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )

                # Add detected peaks
                for peak in result.peaks_detected:
                    if peak.get("type") == "dq_dv_peak":
                        fig.add_trace(
                            go.Scatter(
                                x=[peak["voltage"]],
                                y=[peak["intensity"]],
                                mode="markers",
                                name="Peak",
                                marker={"color": "red", "size": 10, "symbol": "diamond"},
                                showlegend=False,
                                hovertemplate=f'<b>Peak</b><br>Voltage: {peak["voltage"]:.3f} V<br>Intensity: {peak["intensity"]:.3f}<extra></extra>',
                            ),
                            row=1,
                            col=1,
                        )

            # Plot dV/dQ
            if self.diff_config.analysis_type in ["dv_dq", "both"] and dv_dq_data:
                fig.add_trace(
                    go.Scatter(
                        x=voltage_points,
                        y=dv_dq_data,
                        mode="lines",
                        name="dV/dQ",
                        line={"color": "green", "width": self.config.line_width},
                        hovertemplate="<b>dV/dQ</b><br>Voltage: %{x:.3f} V<br>dV/dQ: %{y:.3f} V/Ah<extra></extra>",
                    ),
                    row=2,
                    col=1,
                )

            # Update layout
            fig.update_layout(
                title=f"Differential Analysis - Cycle {data.cycle_number} ({data.charge_state.value})",
                height=self.config.height,
                hovermode="x unified",
            )

            fig.update_xaxes(title_text="Voltage (V)", row=2, col=1)
            fig.update_yaxes(title_text="dQ/dV (Ah/V)", row=1, col=1)
            fig.update_yaxes(title_text="dV/dQ (V/Ah)", row=2, col=1)

            # Apply styling
            fig = self.apply_style(fig)

            self._figure = fig
            return fig

        except Exception as e:
            logger.error(f"Differential plot creation failed: {str(e)}")
            raise PlottingError(f"Plot creation failed: {str(e)}") from e


class NyquistPlot(ElectrochemicalPlot):
    """Nyquist plot for EIS data."""

    def __init__(self, config: EISConfig):
        super().__init__(config)
        self.eis_config = config

    def create_plot(self, data: EISData) -> Any:
        """Create Nyquist plot.

        Args:
            data: EIS data

        Returns:
            Plotly figure
        """
        self.validate_data(data)

        if not PLOTLY_AVAILABLE:
            raise PlottingError("Plotly not available for plotting")

        if not data.impedance_real or not data.impedance_imag:
            raise PlottingError("Missing impedance data")

        try:
            fig = go.Figure()

            z_real = np.array(data.impedance_real)
            z_imag = np.array(data.impedance_imag)
            frequency = np.array(data.frequency) if data.frequency else np.arange(len(z_real))

            # Main Nyquist plot
            fig.add_trace(
                go.Scatter(
                    x=z_real,
                    y=-z_imag,  # Negative imaginary for standard convention
                    mode="lines+markers",
                    name="Nyquist",
                    line={"color": "blue", "width": self.config.line_width},
                    marker={"size": self.config.marker_size},
                    text=[f"{f:.2e} Hz" for f in frequency],
                    hovertemplate="<b>Nyquist Plot</b><br>Z' (Real): %{x:.3f} Ω<br>-Z'' (Imag): %{y:.3f} Ω<br>Frequency: %{text}<extra></extra>",
                )
            )

            # Add frequency labels if requested
            if self.eis_config.show_frequency_labels and len(frequency) > 0:
                # Label high and low frequency points
                high_freq_idx = np.argmax(frequency)
                low_freq_idx = np.argmin(frequency)

                fig.add_annotation(
                    x=z_real[high_freq_idx],
                    y=-z_imag[high_freq_idx],
                    text=f"{frequency[high_freq_idx]:.1e} Hz",
                    showarrow=True,
                    arrowhead=2,
                )

                fig.add_annotation(
                    x=z_real[low_freq_idx],
                    y=-z_imag[low_freq_idx],
                    text=f"{frequency[low_freq_idx]:.1e} Hz",
                    showarrow=True,
                    arrowhead=2,
                )

            # Update layout
            title_text = "Nyquist Plot"
            if data.soc is not None:
                title_text += f" (SOC: {data.soc}%)"
            if data.temperature is not None:
                title_text += f" (T: {data.temperature}°C)"

            fig.update_layout(
                title=title_text,
                xaxis_title="Z' (Real Impedance, Ω)",
                yaxis_title="-Z'' (Imaginary Impedance, Ω)",
                hovermode="closest",
            )

            # Equal aspect ratio for proper circle representation
            fig.update_yaxes(scaleanchor="x", scaleratio=1)

            # Apply axis ranges if specified
            if self.eis_config.impedance_range:
                fig.update_xaxes(range=self.eis_config.impedance_range)
                fig.update_yaxes(
                    range=[
                        -self.eis_config.impedance_range[1],
                        -self.eis_config.impedance_range[0],
                    ]
                )

            # Apply styling
            fig = self.apply_style(fig)

            self._figure = fig
            return fig

        except Exception as e:
            logger.error(f"Nyquist plot creation failed: {str(e)}")
            raise PlottingError(f"Plot creation failed: {str(e)}") from e


class BodePlot(ElectrochemicalPlot):
    """Bode plot for EIS data."""

    def __init__(self, config: EISConfig):
        super().__init__(config)
        self.eis_config = config

    def create_plot(self, data: EISData) -> Any:
        """Create Bode plot.

        Args:
            data: EIS data

        Returns:
            Plotly figure
        """
        self.validate_data(data)

        if not PLOTLY_AVAILABLE:
            raise PlottingError("Plotly not available for plotting")

        try:
            # Process data using analyzer
            analyzer = EISAnalyzer()
            result = analyzer.process(data)

            frequency = result.results.get("frequency", [])
            z_magnitude = result.results.get("z_magnitude", [])
            phase_angle = result.results.get("phase_angle", [])

            if not frequency or not z_magnitude or not phase_angle:
                raise PlottingError("Missing processed EIS data")

            # Create subplot figure
            fig = make_subplots(
                rows=2,
                cols=1,
                subplot_titles=("Magnitude", "Phase"),
                vertical_spacing=0.1,
            )

            # Plot magnitude
            fig.add_trace(
                go.Scatter(
                    x=frequency,
                    y=z_magnitude,
                    mode="lines+markers",
                    name="|Z|",
                    line={"color": "blue", "width": self.config.line_width},
                    marker={"size": self.config.marker_size / 2},
                    hovertemplate="<b>Magnitude</b><br>Frequency: %{x:.2e} Hz<br>|Z|: %{y:.3f} Ω<extra></extra>",
                ),
                row=1,
                col=1,
            )

            # Plot phase
            fig.add_trace(
                go.Scatter(
                    x=frequency,
                    y=phase_angle,
                    mode="lines+markers",
                    name="Phase",
                    line={"color": "red", "width": self.config.line_width},
                    marker={"size": self.config.marker_size / 2},
                    hovertemplate="<b>Phase</b><br>Frequency: %{x:.2e} Hz<br>Phase: %{y:.1f}°<extra></extra>",
                ),
                row=2,
                col=1,
            )

            # Update layout
            title_text = "Bode Plot"
            if data.soc is not None:
                title_text += f" (SOC: {data.soc}%)"
            if data.temperature is not None:
                title_text += f" (T: {data.temperature}°C)"

            fig.update_layout(title=title_text, height=self.config.height, hovermode="x unified")

            # Update axes
            fig.update_xaxes(type="log", title_text="Frequency (Hz)", row=2, col=1)
            fig.update_xaxes(type="log", row=1, col=1)
            fig.update_yaxes(type="log", title_text="|Z| (Ω)", row=1, col=1)
            fig.update_yaxes(title_text="Phase (°)", row=2, col=1)

            # Apply frequency range if specified
            if self.eis_config.frequency_range:
                fig.update_xaxes(
                    range=[
                        np.log10(self.eis_config.frequency_range[0]),
                        np.log10(self.eis_config.frequency_range[1]),
                    ]
                )

            # Apply styling
            fig = self.apply_style(fig)

            self._figure = fig
            return fig

        except Exception as e:
            logger.error(f"Bode plot creation failed: {str(e)}")
            raise PlottingError(f"Plot creation failed: {str(e)}") from e


class VoltageProfilePlot(ElectrochemicalPlot):
    """Voltage profile plot showing voltage vs time."""

    def create_plot(self, data: CycleData) -> Any:
        """Create voltage profile plot.

        Args:
            data: Cycle data

        Returns:
            Plotly figure
        """
        self.validate_data(data)

        if not PLOTLY_AVAILABLE:
            raise PlottingError("Plotly not available for plotting")

        if not data.time or not data.voltage:
            raise PlottingError("Missing time or voltage data")

        try:
            fig = go.Figure()

            time = np.array(data.time)
            voltage = np.array(data.voltage)
            current = np.array(data.current) if data.current else np.zeros_like(time)

            # Convert time to hours
            time_hours = time / 3600

            # Plot voltage profile
            fig.add_trace(
                go.Scatter(
                    x=time_hours,
                    y=voltage,
                    mode="lines",
                    name="Voltage",
                    line={"color": "blue", "width": self.config.line_width},
                    yaxis="y",
                    hovertemplate="<b>Voltage Profile</b><br>Time: %{x:.2f} h<br>Voltage: %{y:.3f} V<extra></extra>",
                )
            )

            # Add current on secondary y-axis
            if data.current:
                fig.add_trace(
                    go.Scatter(
                        x=time_hours,
                        y=current,
                        mode="lines",
                        name="Current",
                        line={"color": "red", "width": self.config.line_width, "dash": "dash"},
                        yaxis="y2",
                        hovertemplate="<b>Current</b><br>Time: %{x:.2f} h<br>Current: %{y:.3f} A<extra></extra>",
                    )
                )

            # Update layout
            fig.update_layout(
                title=f"Voltage Profile - Cycle {data.cycle_number}",
                xaxis_title="Time (h)",
                yaxis={"title": "Voltage (V)", "side": "left"},
                yaxis2={"title": "Current (A)", "side": "right", "overlaying": "y"},
                hovermode="x unified",
            )

            # Apply styling
            fig = self.apply_style(fig)

            self._figure = fig
            return fig

        except Exception as e:
            logger.error(f"Voltage profile plot creation failed: {str(e)}")
            raise PlottingError(f"Plot creation failed: {str(e)}") from e


class RateCapabilityPlot(ElectrochemicalPlot):
    """Rate capability plot showing capacity vs C-rate."""

    def __init__(self, config: RateCapabilityConfig):
        super().__init__(config)
        self.rate_config = config

    def create_plot(self, data: RateCapabilityData) -> Any:
        """Create rate capability plot.

        Args:
            data: Rate capability data

        Returns:
            Plotly figure
        """
        self.validate_data(data)

        if not PLOTLY_AVAILABLE:
            raise PlottingError("Plotly not available for plotting")

        try:
            # Process data using analyzer
            analyzer = RateAnalyzer()
            analyzer.process(data)

            c_rates = np.array(data.c_rates)
            discharge_capacity = np.array(data.discharge_capacity)

            fig = go.Figure()

            # Normalize capacity if requested
            if self.rate_config.normalize_capacity and len(discharge_capacity) > 0:
                normalized_capacity = (discharge_capacity / discharge_capacity[0]) * 100
                y_data = normalized_capacity
                y_title = "Normalized Capacity (%)"
                hover_template = (
                    "<b>Rate Capability</b><br>C-rate: %{x}<br>Capacity: %{y:.1f}%<extra></extra>"
                )
            else:
                y_data = discharge_capacity
                y_title = "Discharge Capacity (Ah)"
                hover_template = (
                    "<b>Rate Capability</b><br>C-rate: %{x}<br>Capacity: %{y:.3f} Ah<extra></extra>"
                )

            # Plot capacity vs C-rate
            if self.rate_config.show_capacity:
                fig.add_trace(
                    go.Scatter(
                        x=c_rates,
                        y=y_data,
                        mode="lines+markers",
                        name="Discharge Capacity",
                        line={"color": "blue", "width": self.config.line_width},
                        marker={"size": self.config.marker_size},
                        hovertemplate=hover_template,
                    )
                )

            # Add efficiency if available
            if self.rate_config.show_efficiency and data.efficiency:
                fig.add_trace(
                    go.Scatter(
                        x=c_rates,
                        y=data.efficiency,
                        mode="lines+markers",
                        name="Coulombic Efficiency",
                        line={"color": "green", "width": self.config.line_width},
                        marker={"size": self.config.marker_size},
                        yaxis="y2",
                        hovertemplate="<b>Efficiency</b><br>C-rate: %{x}<br>Efficiency: %{y:.1f}%<extra></extra>",
                    )
                )

            # Update layout
            fig.update_layout(
                title="Rate Capability Analysis",
                xaxis_title="C-rate",
                yaxis={"title": y_title, "side": "left"},
                hovermode="x unified",
            )

            # Add secondary y-axis for efficiency
            if self.rate_config.show_efficiency and data.efficiency:
                fig.update_layout(
                    yaxis2={"title": "Efficiency (%)", "side": "right", "overlaying": "y"}
                )

            # Use log scale for C-rate if requested
            if self.rate_config.log_scale_rate:
                fig.update_xaxes(type="log")

            # Apply styling
            fig = self.apply_style(fig)

            self._figure = fig
            return fig

        except Exception as e:
            logger.error(f"Rate capability plot creation failed: {str(e)}")
            raise PlottingError(f"Plot creation failed: {str(e)}") from e


class CalendarAgingPlot(ElectrochemicalPlot):
    """Calendar aging plot showing capacity retention vs time."""

    def __init__(self, config: AgingConfig):
        super().__init__(config)
        self.aging_config = config

    def create_plot(self, data: AgingData) -> Any:
        """Create calendar aging plot.

        Args:
            data: Aging data

        Returns:
            Plotly figure
        """
        self.validate_data(data)

        if not PLOTLY_AVAILABLE:
            raise PlottingError("Plotly not available for plotting")

        try:
            # Process data using analyzer
            analyzer = AgingAnalyzer()
            result = analyzer.process(data)

            time_data = np.array(data.time_days)
            capacity_retention = np.array(data.capacity_retention)

            # Convert time units if needed
            if self.aging_config.time_units == "years":
                time_data = time_data / 365.25
                time_label = "Time (years)"
            elif self.aging_config.time_units == "months":
                time_data = time_data / 30.44
                time_label = "Time (months)"
            else:
                time_label = "Time (days)"

            fig = go.Figure()

            # Plot capacity retention
            if self.aging_config.show_capacity:
                fig.add_trace(
                    go.Scatter(
                        x=time_data,
                        y=capacity_retention,
                        mode="lines+markers",
                        name="Capacity Retention",
                        line={"color": "blue", "width": self.config.line_width},
                        marker={"size": self.config.marker_size},
                        hovertemplate=f"<b>Aging</b><br>{time_label}: %{{x:.1f}}<br>Retention: %{{y:.1f}}%<extra></extra>",
                    )
                )

            # Add resistance increase if available
            if self.aging_config.show_resistance and data.resistance_increase:
                fig.add_trace(
                    go.Scatter(
                        x=time_data,
                        y=data.resistance_increase,
                        mode="lines+markers",
                        name="Resistance Increase",
                        line={"color": "red", "width": self.config.line_width},
                        marker={"size": self.config.marker_size},
                        yaxis="y2",
                        hovertemplate=f"<b>Resistance</b><br>{time_label}: %{{x:.1f}}<br>Increase: %{{y:.1f}}%<extra></extra>",
                    )
                )

            # Add aging model fit if requested
            if self.aging_config.fit_aging_model and len(time_data) > 2:
                self._add_aging_model(fig, time_data, capacity_retention, result)

            # Add EOL prediction
            if result.calendar_life_prediction:
                eol_time = result.calendar_life_prediction
                if self.aging_config.time_units == "years":
                    eol_time = eol_time
                elif self.aging_config.time_units == "months":
                    eol_time = eol_time * 12
                else:
                    eol_time = eol_time * 365.25

                fig.add_vline(
                    x=eol_time,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Predicted EOL ({eol_time:.1f} {self.aging_config.time_units})",
                )

            # Update layout
            title_text = f"Calendar Aging Analysis ({data.aging_type})"
            if data.storage_temperature:
                title_text += f" - {data.storage_temperature}°C"
            if data.storage_soc:
                title_text += f" - {data.storage_soc}% SOC"

            fig.update_layout(
                title=title_text,
                xaxis_title=time_label,
                yaxis={"title": "Capacity Retention (%)", "side": "left"},
                hovermode="x unified",
            )

            # Add secondary y-axis for resistance
            if self.aging_config.show_resistance and data.resistance_increase:
                fig.update_layout(
                    yaxis2={"title": "Resistance Increase (%)", "side": "right", "overlaying": "y"}
                )

            # Apply styling
            fig = self.apply_style(fig)

            self._figure = fig
            return fig

        except Exception as e:
            logger.error(f"Calendar aging plot creation failed: {str(e)}")
            raise PlottingError(f"Plot creation failed: {str(e)}") from e

    def _add_aging_model(
        self,
        fig: Any,
        time_data: np.ndarray,
        capacity_retention: np.ndarray,
        result: Any,
    ) -> None:
        """Add aging model fit to the plot."""
        try:
            if self.aging_config.aging_model_type == "sqrt_time":
                # Square root of time model
                sqrt_time = np.sqrt(time_data)
                coeffs = np.polyfit(sqrt_time, capacity_retention, 1)

                # Extend model for prediction
                if self.aging_config.extrapolate_years:
                    max_time = self.aging_config.extrapolate_years
                    if self.aging_config.time_units == "days":
                        max_time *= 365.25
                    elif self.aging_config.time_units == "months":
                        max_time *= 12

                    extended_time = np.linspace(time_data[0], max_time, 100)
                    model_y = np.polyval(coeffs, np.sqrt(extended_time))

                    fig.add_trace(
                        go.Scatter(
                            x=extended_time,
                            y=model_y,
                            mode="lines",
                            name="√t Model",
                            line={"color": "red", "dash": "dash", "width": 2},
                            hovertemplate="<b>√t Model</b><br>Time: %{x:.1f}<br>Predicted: %{y:.1f}%<extra></extra>",
                        )
                    )

        except Exception as e:
            logger.warning(f"Aging model fitting failed: {str(e)}")


class BatchComparisonPlot(ElectrochemicalPlot):
    """Batch comparison plot for statistical analysis."""

    def __init__(self, config: ComparisonConfig):
        super().__init__(config)
        self.comp_config = config

    def create_plot(self, data: ComparisonData) -> Any:
        """Create batch comparison plot.

        Args:
            data: Comparison data

        Returns:
            Plotly figure
        """
        self.validate_data(data)

        if not PLOTLY_AVAILABLE:
            raise PlottingError("Plotly not available for plotting")

        if not data.datasets:
            raise PlottingError("No datasets provided for comparison")

        try:
            # Process data using analyzer
            analyzer = ComparisonAnalyzer()
            result = analyzer.process(data)

            fig = go.Figure()

            # Extract values for comparison
            dataset_values = result.results.get("dataset_values", {})

            if data.comparison_type == AnalysisType.CAPACITY_FADE:
                self._create_capacity_comparison(fig, data, dataset_values)
            elif data.comparison_type == AnalysisType.CYCLE_LIFE:
                self._create_cycle_life_comparison(fig, data, dataset_values)
            else:
                self._create_generic_comparison(fig, data, dataset_values)

            # Add statistical summary if requested
            if self.comp_config.show_statistics:
                self._add_statistical_summary(fig, result)

            # Update layout
            fig.update_layout(
                title=f'Batch Comparison - {data.comparison_type.value.replace("_", " ").title()}',
                hovermode="closest",
            )

            # Apply styling
            fig = self.apply_style(fig)

            self._figure = fig
            return fig

        except Exception as e:
            logger.error(f"Batch comparison plot creation failed: {str(e)}")
            raise PlottingError(f"Plot creation failed: {str(e)}") from e

    def _create_capacity_comparison(
        self, fig: Any, data: ComparisonData, dataset_values: dict[str, float]
    ) -> None:
        """Create capacity comparison plot."""
        list(dataset_values.keys())
        capacities = list(dataset_values.values())

        # Box plot for statistical distribution
        fig.add_trace(
            go.Box(
                y=capacities,
                name="Capacity Distribution",
                boxpoints="all",
                jitter=0.3,
                pointpos=-1.8,
                marker={"color": "blue"},
                hovertemplate="<b>Capacity</b><br>Value: %{y:.3f} Ah<extra></extra>",
            )
        )

        fig.update_layout(yaxis_title="Capacity (Ah)", xaxis_title="Sample Distribution")

    def _create_cycle_life_comparison(
        self, fig: Any, data: ComparisonData, dataset_values: dict[str, float]
    ) -> None:
        """Create cycle life comparison plot."""
        sample_names = list(dataset_values.keys())
        cycle_lives = list(dataset_values.values())

        # Bar plot for cycle life comparison
        fig.add_trace(
            go.Bar(
                x=sample_names,
                y=cycle_lives,
                name="Cycle Life",
                marker={"color": "green"},
                hovertemplate="<b>%{x}</b><br>Cycle Life: %{y:.0f} cycles<extra></extra>",
            )
        )

        fig.update_layout(yaxis_title="Cycle Life (cycles)", xaxis_title="Sample ID")

    def _create_generic_comparison(
        self, fig: Any, data: ComparisonData, dataset_values: dict[str, float]
    ) -> None:
        """Create generic comparison plot."""
        sample_names = list(dataset_values.keys())
        values = list(dataset_values.values())

        # Scatter plot for generic comparison
        fig.add_trace(
            go.Scatter(
                x=sample_names,
                y=values,
                mode="markers",
                name="Samples",
                marker={"size": self.config.marker_size, "color": "blue"},
                hovertemplate="<b>%{x}</b><br>Value: %{y:.3f}<extra></extra>",
            )
        )

        fig.update_layout(yaxis_title="Value", xaxis_title="Sample ID")

    def _add_statistical_summary(self, fig: Any, result: Any) -> None:
        """Add statistical summary to the plot."""
        try:
            stats = result.statistical_summary.get("all", {})

            # Add mean line
            if "mean" in stats:
                fig.add_hline(
                    y=stats["mean"],
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Mean: {stats['mean']:.3f}",
                )

            # Add confidence interval
            if "mean" in stats and "std" in stats:
                ci_upper = stats["mean"] + 1.96 * stats["std"]
                ci_lower = stats["mean"] - 1.96 * stats["std"]

                fig.add_hrect(
                    y0=ci_lower,
                    y1=ci_upper,
                    fillcolor="gray",
                    opacity=0.2,
                    layer="below",
                    line_width=0,
                    annotation_text="95% CI",
                )

        except Exception as e:
            logger.warning(f"Statistical summary addition failed: {str(e)}")


# Factory function for creating plots
def create_electrochemical_plot(
    analysis_type: AnalysisType,
    config: ElectrochemicalConfig,
    data: ElectrochemicalData,
) -> ElectrochemicalPlot:
    """Create electrochemical plot based on analysis type.

    Args:
        analysis_type: Type of analysis/plot
        config: Plot configuration
        data: Electrochemical data

    Returns:
        Electrochemical plot instance

    Raises:
        PlottingError: If plot type is not supported
    """
    plot_classes = {
        AnalysisType.VOLTAGE_CAPACITY: VoltageCapacityPlot,
        AnalysisType.CYCLE_LIFE: CycleLifePlot,
        AnalysisType.CAPACITY_FADE: CapacityFadePlot,
        AnalysisType.DIFFERENTIAL: DifferentialPlot,
        AnalysisType.EIS_NYQUIST: NyquistPlot,
        AnalysisType.EIS_BODE: BodePlot,
        AnalysisType.VOLTAGE_PROFILE: VoltageProfilePlot,
        AnalysisType.RATE_CAPABILITY: RateCapabilityPlot,
        AnalysisType.CALENDAR_AGING: CalendarAgingPlot,
        AnalysisType.BATCH_COMPARISON: BatchComparisonPlot,
    }

    plot_class = plot_classes.get(analysis_type)
    if not plot_class:
        raise PlottingError(f"Unsupported analysis type: {analysis_type}")

    try:
        plot_instance = plot_class(config)
        return plot_instance
    except Exception as e:
        logger.error(f"Failed to create plot: {str(e)}")
        raise PlottingError(f"Plot creation failed: {str(e)}") from e
