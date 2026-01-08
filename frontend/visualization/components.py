"""Reusable chart components for battery data visualization.

This module provides a comprehensive library of chart components built on Plotly
for creating interactive visualizations of battery testing data.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Union

import numpy as np

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    go = None
    px = None
    make_subplots = None
    PLOTLY_AVAILABLE = False

from .exceptions import ChartRenderingError, ComponentError, ConfigurationError, DataFormatError
from .models import ChartConfig, ChartType, ExportConfig

logger = logging.getLogger(__name__)


class BaseChart(ABC):
    """Base class for all chart components."""

    def __init__(self, config: ChartConfig):
        """Initialize chart with configuration.

        Args:
            config: Chart configuration
        """
        if not PLOTLY_AVAILABLE:
            raise ComponentError(
                "Plotly is not available. Install with: pip install plotly",
                component_name="BaseChart",
            )

        self.config = config
        self.figure = None
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate chart configuration."""
        try:
            # Validate data
            if not self.config.data.x or not self.config.data.y:
                raise DataFormatError("Chart data must have both x and y values")

            if len(self.config.data.x) != len(self.config.data.y):
                raise DataFormatError("X and Y data must have the same length")

            # Validate chart type compatibility
            if not self._is_chart_type_compatible():
                raise ConfigurationError(
                    f"Chart type {self.config.type} is not compatible with {self.__class__.__name__}"
                )

        except Exception as e:
            logger.error(f"Chart configuration validation failed: {str(e)}")
            raise ConfigurationError(f"Invalid chart configuration: {str(e)}") from e

    @abstractmethod
    def _is_chart_type_compatible(self) -> bool:
        """Check if chart type is compatible with this component."""

    @abstractmethod
    def _create_trace(self) -> Union["go.Scatter", "go.Bar", "go.Heatmap"]:
        """Create the main chart trace."""

    def render(self) -> Any:
        """Render the chart and return Plotly figure.

        Returns:
            Plotly figure object
        """
        try:
            # Create figure
            self.figure = go.Figure()

            # Add main trace
            trace = self._create_trace()
            self.figure.add_trace(trace)

            # Apply layout
            self._apply_layout()

            # Apply styling
            self._apply_styling()

            # Configure interactions
            self._configure_interactions()

            # Apply animations
            self._apply_animations()

            logger.info(f"Chart {self.config.id} rendered successfully")
            return self.figure

        except Exception as e:
            logger.error(f"Chart rendering failed: {str(e)}")
            raise ChartRenderingError(
                f"Failed to render chart: {str(e)}",
                chart_id=self.config.id,
                chart_type=self.config.type.value,
                rendering_stage="render",
            ) from e

    def _apply_layout(self) -> None:
        """Apply layout configuration to figure."""
        layout_config = self.config.layout

        layout_update = {
            "title": {
                "text": layout_config.title or self.config.title,
                "font": {
                    "family": layout_config.font_family,
                    "size": layout_config.font_size + 4,  # Title slightly larger
                    "color": layout_config.font_color,
                },
            },
            "width": layout_config.width,
            "height": layout_config.height,
            "margin": layout_config.margin,
            "paper_bgcolor": layout_config.paper_color,
            "plot_bgcolor": layout_config.background_color,
            "font": {
                "family": layout_config.font_family,
                "size": layout_config.font_size,
                "color": layout_config.font_color,
            },
            "showlegend": layout_config.show_legend,
        }

        # Configure legend position
        if layout_config.show_legend:
            legend_positions = {
                "right": {"x": 1.02, "y": 1},
                "left": {"x": -0.02, "y": 1, "xanchor": "right"},
                "top": {"x": 0.5, "y": 1.02, "xanchor": "center", "orientation": "h"},
                "bottom": {
                    "x": 0.5,
                    "y": -0.02,
                    "xanchor": "center",
                    "yanchor": "top",
                    "orientation": "h",
                },
            }
            layout_update["legend"] = legend_positions.get(
                layout_config.legend_position, legend_positions["right"]
            )

        self.figure.update_layout(**layout_update)

        # Configure axes
        self._configure_axes()

    def _configure_axes(self) -> None:
        """Configure chart axes."""
        # X-axis configuration
        x_axis_config = {
            "title": self.config.x_axis.title,
            "type": self.config.x_axis.type,
            "range": self.config.x_axis.range,
            "tickformat": self.config.x_axis.tick_format,
            "tickangle": self.config.x_axis.tick_angle,
            "showgrid": self.config.x_axis.show_grid,
            "gridcolor": self.config.x_axis.grid_color,
            "gridwidth": self.config.x_axis.grid_width,
            "zeroline": self.config.x_axis.zero_line,
            "mirror": self.config.x_axis.mirror,
        }

        # Y-axis configuration
        y_axis_config = {
            "title": self.config.y_axis.title,
            "type": self.config.y_axis.type,
            "range": self.config.y_axis.range,
            "tickformat": self.config.y_axis.tick_format,
            "tickangle": self.config.y_axis.tick_angle,
            "showgrid": self.config.y_axis.show_grid,
            "gridcolor": self.config.y_axis.grid_color,
            "gridwidth": self.config.y_axis.grid_width,
            "zeroline": self.config.y_axis.zero_line,
            "mirror": self.config.y_axis.mirror,
        }

        self.figure.update_xaxes(**x_axis_config)
        self.figure.update_yaxes(**y_axis_config)

        # Z-axis for 3D charts
        if self.config.z_axis and hasattr(self.figure, "update_scenes"):
            z_axis_config = {
                "title": self.config.z_axis.title,
                "type": self.config.z_axis.type,
                "range": self.config.z_axis.range,
            }
            self.figure.update_scenes(zaxis=z_axis_config)

    def _apply_styling(self) -> None:
        """Apply styling configuration to traces.
        
        Override in subclasses for specific styling behavior.
        """
        pass  # Default implementation - subclasses may override

    def _configure_interactions(self) -> None:
        """Configure chart interactions."""
        interaction_config = self.config.interaction

        # Configure interaction modes
        dragmode = "zoom"  # default
        if "pan" in [mode.value for mode in interaction_config.mode]:
            dragmode = "pan"
        elif "select" in [mode.value for mode in interaction_config.mode]:
            dragmode = "select"
        elif "lasso" in [mode.value for mode in interaction_config.mode]:
            dragmode = "lasso"

        config_update = {
            "dragmode": dragmode,
            "scrollZoom": interaction_config.scroll_zoom,
            "doubleClick": interaction_config.double_click,
            "showTips": interaction_config.show_tips,
            "displayModeBar": True,
            "modeBarButtonsToRemove": [],
        }

        # Remove interaction buttons based on configuration
        if not any(mode.value in ["zoom", "pan"] for mode in interaction_config.mode):
            config_update["modeBarButtonsToRemove"].extend(["zoom2d", "pan2d"])

        if not any(mode.value in ["select", "lasso"] for mode in interaction_config.mode):
            config_update["modeBarButtonsToRemove"].extend(["select2d", "lasso2d"])

        self.figure.update_layout(dragmode=dragmode)

    def _apply_animations(self) -> None:
        """Apply animation configuration."""
        animation_config = self.config.animation

        if animation_config.type.value != "none":
            layout_update = {
                "transition": {
                    "duration": animation_config.transition_duration,
                    "easing": animation_config.easing,
                },
                "updatemenus": [
                    {
                        "type": "buttons",
                        "showactive": False,
                        "buttons": [
                            {
                                "label": "Play",
                                "method": "animate",
                                "args": [
                                    None,
                                    {
                                        "frame": {"duration": animation_config.frame_duration},
                                        "transition": {"duration": animation_config.duration},
                                    },
                                ],
                            }
                        ],
                    }
                ],
            }
            self.figure.update_layout(**layout_update)

    def export(self, export_config: ExportConfig) -> str:
        """Export chart to specified format.

        Args:
            export_config: Export configuration

        Returns:
            Path to exported file
        """
        if not self.figure:
            raise ChartRenderingError("Chart must be rendered before export")

        try:
            filename = (
                export_config.filename or f"chart_{self.config.id}.{export_config.format.value}"
            )

            if export_config.format.value == "png":
                self.figure.write_image(
                    filename,
                    width=export_config.width,
                    height=export_config.height,
                    scale=export_config.scale,
                )
            elif export_config.format.value == "html":
                self.figure.write_html(
                    filename,
                    include_plotlyjs=export_config.include_plotlyjs,
                    config=export_config.config,
                )
            elif export_config.format.value == "pdf":
                self.figure.write_image(
                    filename,
                    format="pdf",
                    width=export_config.width,
                    height=export_config.height,
                    scale=export_config.scale,
                )
            elif export_config.format.value == "svg":
                self.figure.write_image(
                    filename,
                    format="svg",
                    width=export_config.width,
                    height=export_config.height,
                    scale=export_config.scale,
                )
            elif export_config.format.value == "json":
                self.figure.write_json(filename)
            else:
                raise ConfigurationError(f"Unsupported export format: {export_config.format}")

            logger.info(f"Chart exported to {filename}")
            return filename

        except Exception as e:
            logger.error(f"Chart export failed: {str(e)}")
            raise ChartRenderingError(f"Export failed: {str(e)}", chart_id=self.config.id) from e


class LineChart(BaseChart):
    """Line chart component for time series and continuous data."""

    def _is_chart_type_compatible(self) -> bool:
        return self.config.type == ChartType.LINE

    def _create_trace(self) -> go.Scatter:
        """Create line chart trace."""
        style = self.config.style
        data = self.config.data

        trace = go.Scatter(
            x=data.x,
            y=data.y,
            mode="lines+markers" if style.marker_size > 0 else "lines",
            name=self.config.title or "Line Chart",
            line={"color": style.color, "width": style.line_width, "dash": style.dash},
            marker={
                "size": style.marker_size,
                "symbol": style.marker_symbol,
                "color": style.color,
                "opacity": style.opacity,
            },
            text=data.text,
            hovertext=data.hover_text,
            opacity=style.opacity,
            error_x={"array": data.error_x} if data.error_x else None,
            error_y={"array": data.error_y} if data.error_y else None,
        )

        return trace


class ScatterPlot(BaseChart):
    """Scatter plot component for correlation analysis."""

    def _is_chart_type_compatible(self) -> bool:
        return self.config.type == ChartType.SCATTER

    def _create_trace(self) -> go.Scatter:
        """Create scatter plot trace."""
        style = self.config.style
        data = self.config.data

        # Handle color mapping
        marker_color = data.color if data.color else style.color
        marker_size = data.size if data.size else [style.marker_size] * len(data.x)

        trace = go.Scatter(
            x=data.x,
            y=data.y,
            mode="markers",
            name=self.config.title or "Scatter Plot",
            marker={
                "size": marker_size,
                "color": marker_color,
                "symbol": style.marker_symbol,
                "opacity": style.opacity,
                "colorscale": style.color_scale.value if style.color_scale else None,
                "showscale": True if data.color else False,
                "line": {"width": 1, "color": "white"},
            },
            text=data.text,
            hovertext=data.hover_text,
            error_x={"array": data.error_x} if data.error_x else None,
            error_y={"array": data.error_y} if data.error_y else None,
        )

        return trace


class BarChart(BaseChart):
    """Bar chart component for categorical data."""

    def _is_chart_type_compatible(self) -> bool:
        return self.config.type == ChartType.BAR

    def _create_trace(self) -> go.Bar:
        """Create bar chart trace."""
        style = self.config.style
        data = self.config.data

        trace = go.Bar(
            x=data.x,
            y=data.y,
            name=self.config.title or "Bar Chart",
            marker={
                "color": style.color or style.colors,
                "opacity": style.opacity,
                "line": {"width": 1, "color": "white"},
            },
            text=data.text,
            hovertext=data.hover_text,
            error_x={"array": data.error_x} if data.error_x else None,
            error_y={"array": data.error_y} if data.error_y else None,
        )

        return trace


class HeatmapChart(BaseChart):
    """Heatmap chart component for matrix data visualization."""

    def _is_chart_type_compatible(self) -> bool:
        return self.config.type == ChartType.HEATMAP

    def _create_trace(self) -> go.Heatmap:
        """Create heatmap trace."""
        style = self.config.style
        data = self.config.data

        # Reshape data for heatmap if needed
        if data.z is None:
            # Create 2D array from x, y, and color data
            if data.color is None:
                raise DataFormatError("Heatmap requires z data or color data")

            # Convert to matrix format
            x_unique = sorted(set(data.x))
            y_unique = sorted(set(data.y))
            z_matrix = np.zeros((len(y_unique), len(x_unique)))

            for _i, (x_val, y_val, z_val) in enumerate(zip(data.x, data.y, data.color)):
                x_idx = x_unique.index(x_val)
                y_idx = y_unique.index(y_val)
                z_matrix[y_idx, x_idx] = z_val

            z_data = z_matrix
            x_data = x_unique
            y_data = y_unique
        else:
            z_data = data.z
            x_data = data.x
            y_data = data.y

        trace = go.Heatmap(
            x=x_data,
            y=y_data,
            z=z_data,
            colorscale=style.color_scale.value if style.color_scale else "viridis",
            showscale=True,
            hovertext=data.hover_text,
            text=data.text,
            texttemplate="%{text}",
            textfont={"size": 10},
        )

        return trace


class HistogramChart(BaseChart):
    """Histogram chart component for distribution analysis."""

    def _is_chart_type_compatible(self) -> bool:
        return self.config.type == ChartType.HISTOGRAM

    def _create_trace(self) -> go.Histogram:
        """Create histogram trace."""
        style = self.config.style
        data = self.config.data

        trace = go.Histogram(
            x=data.x,
            name=self.config.title or "Histogram",
            marker={
                "color": style.color,
                "opacity": style.opacity,
                "line": {"width": 1, "color": "white"},
            },
            nbinsx=self.config.custom_config.get("bins", 30),
            histnorm=self.config.custom_config.get("histnorm", ""),
        )

        return trace


class BoxPlot(BaseChart):
    """Box plot component for statistical distribution visualization."""

    def _is_chart_type_compatible(self) -> bool:
        return self.config.type == ChartType.BOX

    def _create_trace(self) -> go.Box:
        """Create box plot trace."""
        style = self.config.style
        data = self.config.data

        trace = go.Box(
            y=data.y,
            x=(
                data.x if len(set(data.x)) < len(data.x) else None
            ),  # Use x for grouping if categorical
            name=self.config.title or "Box Plot",
            marker={"color": style.color, "opacity": style.opacity},
            boxpoints=self.config.custom_config.get("boxpoints", "outliers"),
            jitter=self.config.custom_config.get("jitter", 0.3),
            pointpos=self.config.custom_config.get("pointpos", -1.8),
        )

        return trace


class ViolinPlot(BaseChart):
    """Violin plot component for distribution shape visualization."""

    def _is_chart_type_compatible(self) -> bool:
        return self.config.type == ChartType.VIOLIN

    def _create_trace(self) -> go.Violin:
        """Create violin plot trace."""
        style = self.config.style
        data = self.config.data

        trace = go.Violin(
            y=data.y,
            x=data.x if len(set(data.x)) < len(data.x) else None,
            name=self.config.title or "Violin Plot",
            marker={"color": style.color, "opacity": style.opacity},
            box_visible=self.config.custom_config.get("box_visible", True),
            meanline_visible=self.config.custom_config.get("meanline_visible", True),
            points=self.config.custom_config.get("points", "outliers"),
        )

        return trace


class SurfacePlot(BaseChart):
    """3D surface plot component for three-dimensional data."""

    def _is_chart_type_compatible(self) -> bool:
        return self.config.type == ChartType.SURFACE

    def _create_trace(self) -> go.Surface:
        """Create 3D surface trace."""
        style = self.config.style
        data = self.config.data

        if data.z is None:
            raise DataFormatError("Surface plot requires z data")

        # Reshape z data into matrix if needed
        if isinstance(data.z[0], (list, np.ndarray)):
            z_data = data.z
        else:
            # Assume z is flat array, reshape based on x and y
            x_unique = sorted(set(data.x))
            y_unique = sorted(set(data.y))
            z_data = np.array(data.z).reshape(len(y_unique), len(x_unique))

        trace = go.Surface(
            x=data.x,
            y=data.y,
            z=z_data,
            colorscale=style.color_scale.value if style.color_scale else "viridis",
            showscale=True,
            opacity=style.opacity,
        )

        return trace


class ContourPlot(BaseChart):
    """Contour plot component for 2D representation of 3D data."""

    def _is_chart_type_compatible(self) -> bool:
        return self.config.type == ChartType.CONTOUR

    def _create_trace(self) -> go.Contour:
        """Create contour plot trace."""
        style = self.config.style
        data = self.config.data

        if data.z is None:
            raise DataFormatError("Contour plot requires z data")

        trace = go.Contour(
            x=data.x,
            y=data.y,
            z=data.z,
            colorscale=style.color_scale.value if style.color_scale else "viridis",
            showscale=True,
            contours={
                "showlabels": self.config.custom_config.get("show_labels", True),
                "labelfont": {"size": 10, "color": "white"},
            },
        )

        return trace


class CandlestickChart(BaseChart):
    """Candlestick chart component for OHLC data."""

    def _is_chart_type_compatible(self) -> bool:
        return self.config.type == ChartType.CANDLESTICK

    def _create_trace(self) -> go.Candlestick:
        """Create candlestick trace."""
        data = self.config.data

        # Expect OHLC data in custom_config
        ohlc_data = self.config.custom_config.get("ohlc_data", {})
        if not all(key in ohlc_data for key in ["open", "high", "low", "close"]):
            raise DataFormatError("Candlestick chart requires OHLC data (open, high, low, close)")

        trace = go.Candlestick(
            x=data.x,
            open=ohlc_data["open"],
            high=ohlc_data["high"],
            low=ohlc_data["low"],
            close=ohlc_data["close"],
            name=self.config.title or "Candlestick Chart",
        )

        return trace


class GaugeChart(BaseChart):
    """Gauge chart component for single value visualization."""

    def _is_chart_type_compatible(self) -> bool:
        return self.config.type == ChartType.GAUGE

    def _create_trace(self) -> go.Indicator:
        """Create gauge indicator trace."""
        style = self.config.style
        data = self.config.data

        # Use first y value as gauge value
        value = data.y[0] if data.y else 0

        gauge_config = self.config.custom_config.get("gauge", {})

        trace = go.Indicator(
            mode="gauge+number+delta",
            value=value,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": self.config.title or "Gauge"},
            delta={"reference": gauge_config.get("reference", 0)},
            gauge={
                "axis": {"range": gauge_config.get("range", [0, 100])},
                "bar": {"color": style.color or "darkblue"},
                "steps": gauge_config.get(
                    "steps",
                    [
                        {"range": [0, 50], "color": "lightgray"},
                        {"range": [50, 100], "color": "gray"},
                    ],
                ),
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": gauge_config.get("threshold", 90),
                },
            },
        )

        return trace


class TreemapChart(BaseChart):
    """Treemap chart component for hierarchical data."""

    def _is_chart_type_compatible(self) -> bool:
        return self.config.type == ChartType.TREEMAP

    def _create_trace(self) -> go.Treemap:
        """Create treemap trace."""
        style = self.config.style
        data = self.config.data

        # Expect hierarchical data in custom_config
        treemap_data = self.config.custom_config.get("treemap_data", {})

        trace = go.Treemap(
            labels=treemap_data.get("labels", data.text or data.x),
            values=data.y,
            parents=treemap_data.get("parents", [""] * len(data.y)),
            textinfo="label+value+percent parent",
            marker={
                "colorscale": style.color_scale.value if style.color_scale else "viridis",
                "colorbar": {"thickness": 15, "len": 0.7},
            },
        )

        return trace


class SankeyDiagram(BaseChart):
    """Sankey diagram component for flow visualization."""

    def _is_chart_type_compatible(self) -> bool:
        return self.config.type == ChartType.SANKEY

    def _create_trace(self) -> go.Sankey:
        """Create Sankey diagram trace."""
        style = self.config.style

        # Expect flow data in custom_config
        sankey_data = self.config.custom_config.get("sankey_data", {})
        if not all(key in sankey_data for key in ["source", "target", "value"]):
            raise DataFormatError("Sankey diagram requires source, target, and value data")

        trace = go.Sankey(
            node={
                "pad": 15,
                "thickness": 20,
                "line": {"color": "black", "width": 0.5},
                "label": sankey_data.get("node_labels", []),
                "color": sankey_data.get(
                    "node_colors",
                    style.colors or ["blue"] * len(sankey_data.get("node_labels", [])),
                ),
            },
            link={
                "source": sankey_data["source"],
                "target": sankey_data["target"],
                "value": sankey_data["value"],
                "color": sankey_data.get(
                    "link_colors", ["rgba(0,0,255,0.3)"] * len(sankey_data["source"])
                ),
            },
        )

        return trace


# Chart factory function
def create_chart(config: ChartConfig) -> BaseChart:
    """Create appropriate chart component based on configuration.

    Args:
        config: Chart configuration

    Returns:
        Chart component instance
    """
    chart_classes = {
        ChartType.LINE: LineChart,
        ChartType.SCATTER: ScatterPlot,
        ChartType.BAR: BarChart,
        ChartType.HEATMAP: HeatmapChart,
        ChartType.HISTOGRAM: HistogramChart,
        ChartType.BOX: BoxPlot,
        ChartType.VIOLIN: ViolinPlot,
        ChartType.SURFACE: SurfacePlot,
        ChartType.CONTOUR: ContourPlot,
        ChartType.CANDLESTICK: CandlestickChart,
        ChartType.GAUGE: GaugeChart,
        ChartType.TREEMAP: TreemapChart,
        ChartType.SANKEY: SankeyDiagram,
    }

    chart_class = chart_classes.get(config.type)
    if not chart_class:
        raise ConfigurationError(f"Unsupported chart type: {config.type}")

    return chart_class(config)
