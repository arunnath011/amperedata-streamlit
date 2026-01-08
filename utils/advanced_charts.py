"""
Advanced Battery Analysis Visualizations
=========================================

Professional chart types for electrochemical analysis:
1. Ragone Plot - Energy vs Power density
2. Coulombic Efficiency - Charge/discharge efficiency over cycles
3. Temperature Analysis - Thermal behavior monitoring
4. State of Health (SOH) - Capacity retention and degradation
5. Comparative Analysis - Multi-battery comparison

All charts are publication-ready with professional styling.
"""

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class AdvancedBatteryCharts:
    """Generate advanced battery analysis visualizations."""

    # Professional color schemes
    COLORS = {
        "primary": "#1f77b4",
        "secondary": "#ff7f0e",
        "success": "#2ca02c",
        "danger": "#d62728",
        "warning": "#ff9800",
        "info": "#17a2b8",
        "purple": "#9467bd",
        "pink": "#e377c2",
        "brown": "#8c564b",
        "gray": "#7f7f7f",
    }

    COLORSCALE = [
        "#440154",
        "#482878",
        "#3e4989",
        "#31688e",
        "#26828e",
        "#1f9e89",
        "#35b779",
        "#6ece58",
        "#b5de2b",
        "#fde724",
    ]

    @staticmethod
    def create_ragone_plot(
        batteries_data: dict[str, pd.DataFrame],
        mass_g: Optional[dict[str, float]] = None,
    ) -> go.Figure:
        """
        Create Ragone plot (Energy density vs Power density).

        Shows the trade-off between energy and power for battery comparison.

        Args:
            batteries_data: Dict of battery_id -> DataFrame with columns:
                           ['energy_Wh', 'power_W', 'time_h']
            mass_g: Optional dict of battery_id -> mass in grams
                   If None, uses absolute values instead of specific values

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        for battery_id, data in batteries_data.items():
            if data.empty:
                continue

            # Calculate energy and power metrics
            if mass_g and battery_id in mass_g:
                # Specific energy/power (per unit mass)
                energy_density = data["energy_Wh"] / mass_g[battery_id]
                power_density = data["power_W"] / mass_g[battery_id]
                x_label = "Specific Power (W/kg)"
                y_label = "Specific Energy (Wh/kg)"
                multiplier = 1000  # Convert g to kg
            else:
                # Absolute values
                energy_density = data["energy_Wh"]
                power_density = data["power_W"]
                x_label = "Power (W)"
                y_label = "Energy (Wh)"
                multiplier = 1

            # Create trace
            fig.add_trace(
                go.Scatter(
                    x=power_density * multiplier,
                    y=energy_density * multiplier,
                    mode="markers+lines",
                    name=battery_id,
                    marker={"size": 10, "line": {"width": 1, "color": "white"}},
                    line={"width": 2},
                    hovertemplate=(
                        f"<b>{battery_id}</b><br>"
                        f"{x_label}: %{{x:.2f}}<br>"
                        f"{y_label}: %{{y:.2f}}<br>"
                        "<extra></extra>"
                    ),
                )
            )

        # Reference lines for different battery types
        fig.add_shape(
            type="rect",
            x0=50,
            x1=300,
            y0=50,
            y1=150,
            line={"color": "lightgray", "width": 1, "dash": "dash"},
            fillcolor="lightgray",
            opacity=0.1,
            layer="below",
        )

        fig.update_layout(
            title={
                "text": "Ragone Plot - Energy vs Power Performance",
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 20, "family": "Arial, sans-serif"},
            },
            xaxis_title=x_label,
            yaxis_title=y_label,
            xaxis_type="log",
            yaxis_type="log",
            template="plotly_white",
            height=600,
            hovermode="closest",
            showlegend=True,
            legend={
                "yanchor": "top",
                "y": 0.99,
                "xanchor": "right",
                "x": 0.99,
                "bgcolor": "rgba(255,255,255,0.8)",
            },
        )

        return fig

    @staticmethod
    def create_coulombic_efficiency_plot(
        batteries_data: dict[str, pd.DataFrame],
        show_moving_average: bool = True,
        ma_window: int = 10,
    ) -> go.Figure:
        """
        Create Coulombic Efficiency plot.

        Shows charge/discharge efficiency over cycle life.

        Args:
            batteries_data: Dict of battery_id -> DataFrame with columns:
                           ['cycle_number', 'charge_capacity', 'discharge_capacity']
            show_moving_average: Show moving average line
            ma_window: Moving average window size

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        for battery_id, data in batteries_data.items():
            if data.empty or "charge_capacity" not in data.columns:
                continue

            # Calculate Coulombic Efficiency (%)
            ce = (data["discharge_capacity"] / data["charge_capacity"]) * 100
            ce = ce.replace([np.inf, -np.inf], np.nan).dropna()

            if len(ce) == 0:
                continue

            # Plot raw efficiency
            fig.add_trace(
                go.Scatter(
                    x=data["cycle_number"][: len(ce)],
                    y=ce,
                    mode="markers",
                    name=f"{battery_id}",
                    marker={"size": 6, "opacity": 0.6},
                    hovertemplate=(
                        f"<b>{battery_id}</b><br>"
                        "Cycle: %{x}<br>"
                        "Efficiency: %{y:.2f}%<br>"
                        "<extra></extra>"
                    ),
                )
            )

            # Add moving average
            if show_moving_average and len(ce) >= ma_window:
                ce_ma = ce.rolling(window=ma_window, center=True).mean()
                fig.add_trace(
                    go.Scatter(
                        x=data["cycle_number"][: len(ce_ma)],
                        y=ce_ma,
                        mode="lines",
                        name=f"{battery_id} (MA-{ma_window})",
                        line={"width": 3},
                        hovertemplate=(
                            f"<b>{battery_id} Moving Avg</b><br>"
                            "Cycle: %{x}<br>"
                            "Efficiency: %{y:.2f}%<br>"
                            "<extra></extra>"
                        ),
                    )
                )

        # Add reference line at 100%
        fig.add_hline(
            y=100,
            line_dash="dash",
            line_color="gray",
            annotation_text="100% Efficiency",
            annotation_position="right",
        )

        # Add warning zone (< 95%)
        fig.add_hrect(y0=0, y1=95, fillcolor="red", opacity=0.1, layer="below", line_width=0)

        fig.update_layout(
            title={
                "text": "Coulombic Efficiency - Charge/Discharge Performance",
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 20, "family": "Arial, sans-serif"},
            },
            xaxis_title="Cycle Number",
            yaxis_title="Coulombic Efficiency (%)",
            template="plotly_white",
            height=600,
            hovermode="closest",
            showlegend=True,
            yaxis={"range": [90, 105]},
        )

        return fig

    @staticmethod
    def create_temperature_analysis(
        batteries_data: dict[str, pd.DataFrame], safety_limit_c: float = 60.0
    ) -> go.Figure:
        """
        Create Temperature Analysis plot.

        Monitors thermal behavior during operation.

        Args:
            batteries_data: Dict of battery_id -> DataFrame with columns:
                           ['time_h', 'temperature_C', 'current_A']
            safety_limit_c: Safety temperature limit in Celsius

        Returns:
            Plotly figure with dual y-axes
        """
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=("Temperature Profile", "Current Profile"),
            row_heights=[0.6, 0.4],
        )

        colors = list(AdvancedBatteryCharts.COLORS.values())

        for idx, (battery_id, data) in enumerate(batteries_data.items()):
            if data.empty or "temperature_C" not in data.columns:
                continue

            color = colors[idx % len(colors)]

            # Temperature trace
            fig.add_trace(
                go.Scatter(
                    x=data["time_h"],
                    y=data["temperature_C"],
                    mode="lines",
                    name=f"{battery_id} Temp",
                    line={"color": color, "width": 2},
                    hovertemplate=(
                        f"<b>{battery_id}</b><br>"
                        "Time: %{x:.2f} h<br>"
                        "Temp: %{y:.1f}°C<br>"
                        "<extra></extra>"
                    ),
                ),
                row=1,
                col=1,
            )

            # Current trace (if available)
            if "current_A" in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data["time_h"],
                        y=data["current_A"],
                        mode="lines",
                        name=f"{battery_id} Current",
                        line={"color": color, "width": 2, "dash": "dot"},
                        hovertemplate=(
                            f"<b>{battery_id}</b><br>"
                            "Time: %{x:.2f} h<br>"
                            "Current: %{y:.2f} A<br>"
                            "<extra></extra>"
                        ),
                    ),
                    row=2,
                    col=1,
                )

        # Add safety limit line
        fig.add_hline(
            y=safety_limit_c,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Safety Limit ({safety_limit_c}°C)",
            annotation_position="right",
            row=1,
            col=1,
        )

        # Add warning zone
        fig.add_hrect(
            y0=safety_limit_c,
            y1=safety_limit_c * 1.2,
            fillcolor="red",
            opacity=0.1,
            layer="below",
            line_width=0,
            row=1,
            col=1,
        )

        fig.update_xaxes(title_text="Time (hours)", row=2, col=1)
        fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
        fig.update_yaxes(title_text="Current (A)", row=2, col=1)

        fig.update_layout(
            title={
                "text": "Temperature Analysis - Thermal Behavior Monitoring",
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 20, "family": "Arial, sans-serif"},
            },
            template="plotly_white",
            height=700,
            hovermode="x unified",
            showlegend=True,
        )

        return fig

    @staticmethod
    def create_soh_plot(
        batteries_data: dict[str, pd.DataFrame],
        initial_capacity: Optional[dict[str, float]] = None,
        eol_threshold: float = 80.0,
    ) -> go.Figure:
        """
        Create State of Health (SOH) plot.

        Shows capacity retention and degradation over time.

        Args:
            batteries_data: Dict of battery_id -> DataFrame with columns:
                           ['cycle_number', 'discharge_capacity']
            initial_capacity: Optional dict of battery_id -> initial capacity (Ah)
                             If None, uses first cycle as reference
            eol_threshold: End-of-life threshold (% of initial capacity)

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        for battery_id, data in batteries_data.items():
            if data.empty or "discharge_capacity" not in data.columns:
                continue

            # Get initial capacity
            if initial_capacity and battery_id in initial_capacity:
                init_cap = initial_capacity[battery_id]
            else:
                init_cap = data["discharge_capacity"].iloc[0]

            # Calculate SOH (%)
            soh = (data["discharge_capacity"] / init_cap) * 100

            # Plot SOH
            fig.add_trace(
                go.Scatter(
                    x=data["cycle_number"],
                    y=soh,
                    mode="lines+markers",
                    name=battery_id,
                    line={"width": 2},
                    marker={"size": 5},
                    hovertemplate=(
                        f"<b>{battery_id}</b><br>"
                        "Cycle: %{x}<br>"
                        "SOH: %{y:.1f}%<br>"
                        f"Capacity: {data['discharge_capacity'].iloc[0]:.2f} Ah<br>"
                        "<extra></extra>"
                    ),
                )
            )

            # Add trend line
            if len(data) > 10:
                z = np.polyfit(data["cycle_number"], soh, 2)
                p = np.poly1d(z)
                fig.add_trace(
                    go.Scatter(
                        x=data["cycle_number"],
                        y=p(data["cycle_number"]),
                        mode="lines",
                        name=f"{battery_id} Trend",
                        line={"width": 2, "dash": "dash"},
                        opacity=0.5,
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

        # Add EOL threshold
        fig.add_hline(
            y=eol_threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"EOL Threshold ({eol_threshold}%)",
            annotation_position="right",
        )

        # Add zones
        fig.add_hrect(
            y0=eol_threshold,
            y1=100,
            fillcolor="green",
            opacity=0.05,
            layer="below",
            line_width=0,
        )

        fig.add_hrect(
            y0=0,
            y1=eol_threshold,
            fillcolor="red",
            opacity=0.05,
            layer="below",
            line_width=0,
        )

        fig.update_layout(
            title={
                "text": "State of Health (SOH) - Capacity Retention Analysis",
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 20, "family": "Arial, sans-serif"},
            },
            xaxis_title="Cycle Number",
            yaxis_title="State of Health (%)",
            template="plotly_white",
            height=600,
            hovermode="closest",
            showlegend=True,
            yaxis={"range": [0, 110]},
        )

        return fig

    @staticmethod
    def create_comparative_analysis(
        batteries_data: dict[str, pd.DataFrame],
        metrics: list[str] = None,
    ) -> go.Figure:
        """
        Create Comparative Analysis radar/spider chart.

        Multi-battery performance comparison across key metrics.

        Args:
            batteries_data: Dict of battery_id -> DataFrame with metrics
            metrics: List of metrics to compare

        Returns:
            Plotly figure
        """
        # Prepare data for radar chart
        if metrics is None:
            metrics = ["capacity", "voltage", "efficiency", "resistance"]
        categories = []
        battery_ids = list(batteries_data.keys())

        # Calculate normalized metrics (0-100 scale)
        metric_data = {bid: [] for bid in battery_ids}

        for metric in metrics:
            categories.append(metric.capitalize())
            values = []

            for _battery_id, data in batteries_data.items():
                if metric == "capacity":
                    val = (
                        data["discharge_capacity"].mean()
                        if "discharge_capacity" in data.columns
                        else 0
                    )
                elif metric == "voltage":
                    val = data["voltage"].mean() if "voltage" in data.columns else 0
                elif metric == "efficiency":
                    if "charge_capacity" in data.columns and "discharge_capacity" in data.columns:
                        val = (data["discharge_capacity"] / data["charge_capacity"]).mean() * 100
                    else:
                        val = 0
                elif metric == "resistance":
                    val = 100 - (
                        data["Re"].mean() if "Re" in data.columns else 0
                    )  # Inverted (lower is better)
                else:
                    val = 0

                values.append(val)

            # Normalize to 0-100
            if max(values) > 0:
                normalized = [(v / max(values)) * 100 for v in values]
            else:
                normalized = [0] * len(values)

            for idx, battery_id in enumerate(battery_ids):
                metric_data[battery_id].append(normalized[idx])

        # Create radar chart
        fig = go.Figure()

        colors = list(AdvancedBatteryCharts.COLORS.values())

        for idx, battery_id in enumerate(battery_ids):
            fig.add_trace(
                go.Scatterpolar(
                    r=metric_data[battery_id] + [metric_data[battery_id][0]],  # Close the loop
                    theta=categories + [categories[0]],
                    fill="toself",
                    name=battery_id,
                    line={"color": colors[idx % len(colors)], "width": 2},
                    opacity=0.6,
                )
            )

        fig.update_layout(
            title={
                "text": "Comparative Analysis - Multi-Battery Performance",
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 20, "family": "Arial, sans-serif"},
            },
            polar={"radialaxis": {"visible": True, "range": [0, 100], "ticksuffix": "%"}},
            template="plotly_white",
            height=600,
            showlegend=True,
            legend={"orientation": "v", "yanchor": "middle", "y": 0.5, "xanchor": "left", "x": 1.1},
        )

        return fig

    @staticmethod
    def export_chart_high_res(fig: go.Figure, filename: str, dpi: int = 300) -> str:
        """
        Export chart as high-resolution image for publication.

        Args:
            fig: Plotly figure
            filename: Output filename (without extension)
            dpi: Resolution in DPI

        Returns:
            Path to saved file
        """
        # Update layout for publication quality
        fig.update_layout(
            font={"size": 14, "family": "Arial, sans-serif"},
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        # Export as PNG
        output_path = f"{filename}.png"
        fig.write_image(output_path, width=1920, height=1080, scale=dpi / 100)

        return output_path
