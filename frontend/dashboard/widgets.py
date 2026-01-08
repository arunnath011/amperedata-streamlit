"""Dashboard widget library with reusable components.

This module provides a comprehensive library of dashboard widgets including
KPI cards, charts, tables, metrics, gauges, and custom widgets for battery data analysis.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd

# Try to import Plotly (optional)
try:
    import plotly.express as px
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    go = None
    px = None
    PLOTLY_AVAILABLE = False

from frontend.visualization.components import create_chart

from .exceptions import DataSourceError, RenderingError, WidgetError
from .models import (
    AlertConfig,
    ChartWidgetConfig,
    DataSource,
    FilterConfig,
    GaugeConfig,
    ImageConfig,
    KPIConfig,
    MetricConfig,
    ProgressConfig,
    TableConfig,
    TextConfig,
    WidgetConfig,
    WidgetType,
)

logger = logging.getLogger(__name__)


class BaseWidget(ABC):
    """Abstract base class for all dashboard widgets."""

    def __init__(self, widget_id: str, config: WidgetConfig):
        """Initialize widget with configuration.

        Args:
            widget_id: Unique widget identifier
            config: Widget configuration
        """
        self.widget_id = widget_id
        self.config = config
        self._data = None
        self._last_update = None

    @abstractmethod
    def render(self) -> Any:
        """Render the widget.

        Returns:
            Rendered widget component
        """

    @abstractmethod
    def validate_config(self) -> bool:
        """Validate widget configuration.

        Returns:
            True if configuration is valid

        Raises:
            WidgetError: If configuration is invalid
        """

    def load_data(self) -> Optional[pd.DataFrame]:
        """Load data from configured data source.

        Returns:
            Loaded data as DataFrame or None

        Raises:
            DataSourceError: If data loading fails
        """
        if not self.config.data_source:
            return None

        try:
            data_source = self.config.data_source

            if data_source.type == "database":
                return self._load_database_data(data_source)
            elif data_source.type == "api":
                return self._load_api_data(data_source)
            elif data_source.type == "file":
                return self._load_file_data(data_source)
            elif data_source.type == "mock":
                return self._generate_mock_data()
            else:
                raise DataSourceError(f"Unsupported data source type: {data_source.type}")

        except Exception as e:
            logger.error(f"Failed to load data for widget {self.widget_id}: {str(e)}")
            raise DataSourceError(f"Data loading failed: {str(e)}")

    def _load_database_data(self, data_source: DataSource) -> pd.DataFrame:
        """Load data from database."""
        # TODO: Implement database connection
        # For now, return mock data
        return self._generate_mock_data()

    def _load_api_data(self, data_source: DataSource) -> pd.DataFrame:
        """Load data from API."""
        # TODO: Implement API data loading
        # For now, return mock data
        return self._generate_mock_data()

    def _load_file_data(self, data_source: DataSource) -> pd.DataFrame:
        """Load data from file."""
        # TODO: Implement file data loading
        # For now, return mock data
        return self._generate_mock_data()

    def _generate_mock_data(self) -> pd.DataFrame:
        """Generate mock data for testing."""
        np.random.seed(42)  # For reproducible results
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        data = pd.DataFrame(
            {
                "date": dates,
                "voltage": 3.7 + 0.1 * np.random.randn(100),
                "current": 1.0 + 0.05 * np.random.randn(100),
                "capacity": 2.5 + 0.2 * np.random.randn(100),
                "temperature": 25 + 5 * np.random.randn(100),
                "cycle": np.arange(1, 101),
            }
        )
        return data

    def refresh_data(self) -> None:
        """Refresh widget data."""
        try:
            self._data = self.load_data()
            self._last_update = datetime.now()
            logger.info(f"Data refreshed for widget {self.widget_id}")
        except Exception as e:
            logger.error(f"Failed to refresh data for widget {self.widget_id}: {str(e)}")
            raise

    def get_data(self) -> Optional[pd.DataFrame]:
        """Get current widget data."""
        return self._data

    def get_last_update(self) -> Optional[datetime]:
        """Get last data update timestamp."""
        return self._last_update


class KPIWidget(BaseWidget):
    """KPI (Key Performance Indicator) widget."""

    def __init__(self, widget_id: str, config: KPIConfig):
        super().__init__(widget_id, config)
        self.kpi_config = config

    def validate_config(self) -> bool:
        """Validate KPI configuration."""
        if not isinstance(self.kpi_config, KPIConfig):
            raise WidgetError("Invalid KPI configuration")

        if self.kpi_config.value is None:
            raise WidgetError("KPI value is required")

        return True

    def render(self) -> dict[str, Any]:
        """Render KPI widget.

        Returns:
            KPI widget data for rendering
        """
        self.validate_config()

        # Calculate trend indicator
        trend_indicator = ""
        trend_color = "gray"

        if self.kpi_config.trend is not None and self.kpi_config.show_trend:
            if self.kpi_config.trend > 0:
                trend_indicator = "↗"
                trend_color = "green"
            elif self.kpi_config.trend < 0:
                trend_indicator = "↘"
                trend_color = "red"
            else:
                trend_indicator = "→"
                trend_color = "gray"

        # Format value
        formatted_value = self.kpi_config.value
        if self.kpi_config.format and isinstance(self.kpi_config.value, (int, float)):
            try:
                formatted_value = self.kpi_config.format.format(self.kpi_config.value)
            except:
                formatted_value = str(self.kpi_config.value)

        # Calculate target comparison
        target_comparison = None
        if self.kpi_config.target is not None and self.kpi_config.show_target:
            if isinstance(self.kpi_config.value, (int, float)):
                diff = self.kpi_config.value - self.kpi_config.target
                target_comparison = {
                    "difference": diff,
                    "percentage": (
                        (diff / self.kpi_config.target) * 100 if self.kpi_config.target != 0 else 0
                    ),
                }

        return {
            "type": "kpi",
            "title": self.kpi_config.title,
            "value": formatted_value,
            "unit": self.kpi_config.unit,
            "trend": (
                {
                    "value": self.kpi_config.trend,
                    "indicator": trend_indicator,
                    "color": trend_color,
                }
                if self.kpi_config.trend is not None
                else None
            ),
            "target": target_comparison,
            "color_scheme": self.kpi_config.color_scheme,
            "widget_id": self.widget_id,
        }


class ChartWidget(BaseWidget):
    """Chart widget using visualization framework."""

    def __init__(self, widget_id: str, config: ChartWidgetConfig):
        super().__init__(widget_id, config)
        self.chart_config = config

    def validate_config(self) -> bool:
        """Validate chart configuration."""
        if not isinstance(self.chart_config, ChartWidgetConfig):
            raise WidgetError("Invalid chart configuration")

        if not self.chart_config.chart_config:
            raise WidgetError("Chart configuration is required")

        return True

    def render(self) -> Any:
        """Render chart widget.

        Returns:
            Plotly figure or chart data
        """
        self.validate_config()

        try:
            # Create chart using visualization framework
            chart = create_chart(self.chart_config.chart_config)
            figure = chart.render()

            return {
                "type": "chart",
                "title": self.chart_config.title,
                "figure": figure,
                "export_enabled": self.chart_config.export_enabled,
                "fullscreen_enabled": self.chart_config.fullscreen_enabled,
                "widget_id": self.widget_id,
            }

        except Exception as e:
            logger.error(f"Failed to render chart widget {self.widget_id}: {str(e)}")
            raise RenderingError(f"Chart rendering failed: {str(e)}", widget_id=self.widget_id)


class TableWidget(BaseWidget):
    """Table widget for tabular data display."""

    def __init__(self, widget_id: str, config: TableConfig):
        super().__init__(widget_id, config)
        self.table_config = config

    def validate_config(self) -> bool:
        """Validate table configuration."""
        if not isinstance(self.table_config, TableConfig):
            raise WidgetError("Invalid table configuration")

        if not self.table_config.columns:
            raise WidgetError("Table columns are required")

        return True

    def render(self) -> dict[str, Any]:
        """Render table widget.

        Returns:
            Table widget data for rendering
        """
        self.validate_config()

        # Load data
        data = self.get_data()
        if data is None:
            data = self.load_data()

        # Process table data
        table_data = []
        if data is not None:
            # Apply column filtering if specified
            available_columns = [
                col["field"] for col in self.table_config.columns if col["field"] in data.columns
            ]
            if available_columns:
                filtered_data = data[available_columns]
                table_data = filtered_data.to_dict("records")

        return {
            "type": "table",
            "title": self.table_config.title,
            "columns": self.table_config.columns,
            "data": table_data,
            "sortable": self.table_config.sortable,
            "filterable": self.table_config.filterable,
            "searchable": self.table_config.searchable,
            "pagination": self.table_config.pagination,
            "page_size": self.table_config.page_size,
            "export_enabled": self.table_config.export_enabled,
            "row_selection": self.table_config.row_selection,
            "widget_id": self.widget_id,
        }


class TextWidget(BaseWidget):
    """Text widget for displaying formatted text content."""

    def __init__(self, widget_id: str, config: TextConfig):
        super().__init__(widget_id, config)
        self.text_config = config

    def validate_config(self) -> bool:
        """Validate text configuration."""
        if not isinstance(self.text_config, TextConfig):
            raise WidgetError("Invalid text configuration")

        if not self.text_config.content:
            raise WidgetError("Text content is required")

        return True

    def render(self) -> dict[str, Any]:
        """Render text widget.

        Returns:
            Text widget data for rendering
        """
        self.validate_config()

        return {
            "type": "text",
            "title": self.text_config.title,
            "content": self.text_config.content,
            "markdown_enabled": self.text_config.markdown_enabled,
            "html_enabled": self.text_config.html_enabled,
            "auto_size": self.text_config.auto_size,
            "widget_id": self.widget_id,
        }


class ImageWidget(BaseWidget):
    """Image widget for displaying images."""

    def __init__(self, widget_id: str, config: ImageConfig):
        super().__init__(widget_id, config)
        self.image_config = config

    def validate_config(self) -> bool:
        """Validate image configuration."""
        if not isinstance(self.image_config, ImageConfig):
            raise WidgetError("Invalid image configuration")

        if not self.image_config.image_url and not self.image_config.image_data:
            raise WidgetError("Image URL or image data is required")

        return True

    def render(self) -> dict[str, Any]:
        """Render image widget.

        Returns:
            Image widget data for rendering
        """
        self.validate_config()

        return {
            "type": "image",
            "title": self.image_config.title,
            "image_url": self.image_config.image_url,
            "image_data": self.image_config.image_data,
            "alt_text": self.image_config.alt_text,
            "fit_mode": self.image_config.fit_mode,
            "clickable": self.image_config.clickable,
            "click_url": self.image_config.click_url,
            "widget_id": self.widget_id,
        }


class MetricWidget(BaseWidget):
    """Metric widget for displaying single metrics with trends."""

    def __init__(self, widget_id: str, config: MetricConfig):
        super().__init__(widget_id, config)
        self.metric_config = config

    def validate_config(self) -> bool:
        """Validate metric configuration."""
        if not isinstance(self.metric_config, MetricConfig):
            raise WidgetError("Invalid metric configuration")

        if self.metric_config.value is None:
            raise WidgetError("Metric value is required")

        if not self.metric_config.label:
            raise WidgetError("Metric label is required")

        return True

    def render(self) -> dict[str, Any]:
        """Render metric widget.

        Returns:
            Metric widget data for rendering
        """
        self.validate_config()

        # Format value with precision
        formatted_value = self.metric_config.value
        if isinstance(self.metric_config.value, float):
            formatted_value = round(self.metric_config.value, self.metric_config.precision)

        # Generate trend sparkline if data available
        trend_sparkline = None
        if self.metric_config.trend_data and PLOTLY_AVAILABLE:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    y=self.metric_config.trend_data,
                    mode="lines",
                    line=dict(color=self.metric_config.color or "#1f77b4", width=2),
                    showlegend=False,
                )
            )
            fig.update_layout(
                showlegend=False,
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                margin=dict(l=0, r=0, t=0, b=0),
                height=50,
            )
            trend_sparkline = fig

        return {
            "type": "metric",
            "title": self.metric_config.title,
            "label": self.metric_config.label,
            "value": formatted_value,
            "unit": self.metric_config.unit,
            "color": self.metric_config.color,
            "icon": self.metric_config.icon,
            "trend_sparkline": trend_sparkline,
            "widget_id": self.widget_id,
        }


class GaugeWidget(BaseWidget):
    """Gauge widget for displaying values with thresholds."""

    def __init__(self, widget_id: str, config: GaugeConfig):
        super().__init__(widget_id, config)
        self.gauge_config = config

    def validate_config(self) -> bool:
        """Validate gauge configuration."""
        if not isinstance(self.gauge_config, GaugeConfig):
            raise WidgetError("Invalid gauge configuration")

        if self.gauge_config.value is None:
            raise WidgetError("Gauge value is required")

        if self.gauge_config.min_value >= self.gauge_config.max_value:
            raise WidgetError("Gauge max_value must be greater than min_value")

        return True

    def render(self) -> Any:
        """Render gauge widget.

        Returns:
            Plotly gauge figure or gauge data
        """
        self.validate_config()

        if not PLOTLY_AVAILABLE:
            # Return simple gauge data if Plotly not available
            percentage = (
                (self.gauge_config.value - self.gauge_config.min_value)
                / (self.gauge_config.max_value - self.gauge_config.min_value)
            ) * 100

            return {
                "type": "gauge",
                "title": self.gauge_config.title,
                "value": self.gauge_config.value,
                "percentage": percentage,
                "unit": self.gauge_config.unit,
                "show_value": self.gauge_config.show_value,
                "widget_id": self.widget_id,
            }

        # Create Plotly gauge
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number+delta" if self.gauge_config.show_value else "gauge",
                value=self.gauge_config.value,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": self.gauge_config.title or ""},
                gauge={
                    "axis": {
                        "range": [
                            self.gauge_config.min_value,
                            self.gauge_config.max_value,
                        ]
                    },
                    "bar": {"color": "darkblue"},
                    "steps": self._get_gauge_steps(),
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": self.gauge_config.max_value * 0.9,
                    },
                },
            )
        )

        fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))

        return {
            "type": "gauge",
            "title": self.gauge_config.title,
            "figure": fig,
            "widget_id": self.widget_id,
        }

    def _get_gauge_steps(self) -> list[dict[str, Any]]:
        """Get gauge color steps from thresholds."""
        if not self.gauge_config.thresholds:
            return [
                {
                    "range": [
                        self.gauge_config.min_value,
                        self.gauge_config.max_value * 0.7,
                    ],
                    "color": "lightgray",
                },
                {
                    "range": [
                        self.gauge_config.max_value * 0.7,
                        self.gauge_config.max_value * 0.9,
                    ],
                    "color": "yellow",
                },
                {
                    "range": [
                        self.gauge_config.max_value * 0.9,
                        self.gauge_config.max_value,
                    ],
                    "color": "red",
                },
            ]

        steps = []
        for threshold in self.gauge_config.thresholds:
            steps.append(
                {
                    "range": threshold.get(
                        "range",
                        [self.gauge_config.min_value, self.gauge_config.max_value],
                    ),
                    "color": threshold.get("color", "lightgray"),
                }
            )

        return steps


class ProgressWidget(BaseWidget):
    """Progress widget for displaying progress bars."""

    def __init__(self, widget_id: str, config: ProgressConfig):
        super().__init__(widget_id, config)
        self.progress_config = config

    def validate_config(self) -> bool:
        """Validate progress configuration."""
        if not isinstance(self.progress_config, ProgressConfig):
            raise WidgetError("Invalid progress configuration")

        if self.progress_config.value is None:
            raise WidgetError("Progress value is required")

        if self.progress_config.value < 0:
            raise WidgetError("Progress value cannot be negative")

        return True

    def render(self) -> dict[str, Any]:
        """Render progress widget.

        Returns:
            Progress widget data for rendering
        """
        self.validate_config()

        # Calculate percentage
        percentage = (self.progress_config.value / self.progress_config.max_value) * 100
        percentage = min(100, max(0, percentage))  # Clamp to 0-100

        return {
            "type": "progress",
            "title": self.progress_config.title,
            "value": self.progress_config.value,
            "max_value": self.progress_config.max_value,
            "percentage": percentage,
            "label": self.progress_config.label,
            "show_percentage": self.progress_config.show_percentage,
            "color": self.progress_config.color,
            "animated": self.progress_config.animated,
            "widget_id": self.widget_id,
        }


class AlertWidget(BaseWidget):
    """Alert widget for displaying notifications and messages."""

    def __init__(self, widget_id: str, config: AlertConfig):
        super().__init__(widget_id, config)
        self.alert_config = config

    def validate_config(self) -> bool:
        """Validate alert configuration."""
        if not isinstance(self.alert_config, AlertConfig):
            raise WidgetError("Invalid alert configuration")

        if not self.alert_config.message:
            raise WidgetError("Alert message is required")

        valid_types = ["info", "warning", "error", "success"]
        if self.alert_config.alert_type not in valid_types:
            raise WidgetError(f"Alert type must be one of: {valid_types}")

        return True

    def render(self) -> dict[str, Any]:
        """Render alert widget.

        Returns:
            Alert widget data for rendering
        """
        self.validate_config()

        # Get alert color based on type
        color_map = {
            "info": "#17a2b8",
            "warning": "#ffc107",
            "error": "#dc3545",
            "success": "#28a745",
        }

        return {
            "type": "alert",
            "title": self.alert_config.title,
            "message": self.alert_config.message,
            "alert_type": self.alert_config.alert_type,
            "color": color_map.get(self.alert_config.alert_type, "#17a2b8"),
            "dismissible": self.alert_config.dismissible,
            "auto_dismiss": self.alert_config.auto_dismiss,
            "icon": self.alert_config.icon,
            "widget_id": self.widget_id,
        }


class FilterWidget(BaseWidget):
    """Filter widget for controlling other widgets."""

    def __init__(self, widget_id: str, config: FilterConfig):
        super().__init__(widget_id, config)
        self.filter_config = config

    def validate_config(self) -> bool:
        """Validate filter configuration."""
        if not isinstance(self.filter_config, FilterConfig):
            raise WidgetError("Invalid filter configuration")

        if not self.filter_config.filter_type:
            raise WidgetError("Filter type is required")

        valid_types = ["dropdown", "slider", "date", "text", "checkbox", "radio"]
        if self.filter_config.filter_type not in valid_types:
            raise WidgetError(f"Filter type must be one of: {valid_types}")

        return True

    def render(self) -> dict[str, Any]:
        """Render filter widget.

        Returns:
            Filter widget data for rendering
        """
        self.validate_config()

        return {
            "type": "filter",
            "title": self.filter_config.title,
            "filter_type": self.filter_config.filter_type,
            "options": self.filter_config.options,
            "default_value": self.filter_config.default_value,
            "multi_select": self.filter_config.multi_select,
            "target_widgets": self.filter_config.target_widgets,
            "widget_id": self.widget_id,
        }


# Widget factory function
def create_widget(widget_type: WidgetType, widget_id: str, config: WidgetConfig) -> BaseWidget:
    """Create a widget instance based on type and configuration.

    Args:
        widget_type: Type of widget to create
        widget_id: Unique widget identifier
        config: Widget configuration

    Returns:
        Widget instance

    Raises:
        WidgetError: If widget type is not supported
    """
    widget_classes = {
        WidgetType.KPI: KPIWidget,
        WidgetType.CHART: ChartWidget,
        WidgetType.TABLE: TableWidget,
        WidgetType.TEXT: TextWidget,
        WidgetType.IMAGE: ImageWidget,
        WidgetType.METRIC: MetricWidget,
        WidgetType.GAUGE: GaugeWidget,
        WidgetType.PROGRESS: ProgressWidget,
        WidgetType.ALERT: AlertWidget,
        WidgetType.FILTER: FilterWidget,
    }

    widget_class = widget_classes.get(widget_type)
    if not widget_class:
        raise WidgetError(f"Unsupported widget type: {widget_type}")

    try:
        return widget_class(widget_id, config)
    except Exception as e:
        logger.error(f"Failed to create widget {widget_id}: {str(e)}")
        raise WidgetError(f"Widget creation failed: {str(e)}")


# Widget utility functions
def get_supported_widget_types() -> list[WidgetType]:
    """Get list of supported widget types."""
    return list(WidgetType)


def validate_widget_config(widget_type: WidgetType, config: dict[str, Any]) -> bool:
    """Validate widget configuration without creating widget instance.

    Args:
        widget_type: Widget type
        config: Configuration dictionary

    Returns:
        True if configuration is valid

    Raises:
        WidgetError: If configuration is invalid
    """
    # Create temporary widget to validate configuration
    temp_widget = create_widget(widget_type, "temp", config)
    return temp_widget.validate_config()
