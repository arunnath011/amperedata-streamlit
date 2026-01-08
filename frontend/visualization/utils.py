"""Utility classes for visualization framework.

This module provides utility classes for data processing, chart export,
color management, layout management, and validation.
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    import plotly.io as pio

    PLOTLY_AVAILABLE = True
except ImportError:
    go = None
    pio = None
    PLOTLY_AVAILABLE = False

from .exceptions import DataFormatError, ExportError, PerformanceError
from .models import ChartConfig, ChartValidation, ExportConfig, ExportFormat

logger = logging.getLogger(__name__)


class DataProcessor:
    """Utility class for data processing and transformation."""

    @staticmethod
    def validate_data(data: pd.DataFrame) -> ChartValidation:
        """Validate DataFrame for chart compatibility.

        Args:
            data: DataFrame to validate

        Returns:
            Validation result
        """
        errors = []
        warnings = []
        suggestions = []

        try:
            # Check if DataFrame is empty
            if data.empty:
                errors.append("DataFrame is empty")

            # Check for missing values
            missing_count = data.isnull().sum().sum()
            if missing_count > 0:
                warnings.append(f"Found {missing_count} missing values")
                suggestions.append("Consider filling or removing missing values")

            # Check data types
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                warnings.append("No numeric columns found")
                suggestions.append("Ensure at least one column contains numeric data")

            # Check for infinite values
            if np.isinf(data.select_dtypes(include=[np.number])).any().any():
                errors.append("Found infinite values in numeric columns")

            # Check memory usage
            memory_mb = data.memory_usage(deep=True).sum() / 1024 / 1024
            if memory_mb > 100:  # 100 MB threshold
                warnings.append(f"Large dataset: {memory_mb:.1f} MB")
                suggestions.append("Consider sampling or filtering data for better performance")

            # Performance score
            performance_score = max(0.0, 1.0 - (len(data) / 100000) - (memory_mb / 1000))
            performance_score = min(1.0, performance_score)

            # Accessibility score (based on data completeness and structure)
            total_values = len(data) * len(data.columns)
            if total_values > 0:
                completeness = 1.0 - (missing_count / total_values)
            else:
                completeness = 0.0

            if len(data.columns) > 0:
                accessibility_score = (
                    completeness * 0.8 + (len(numeric_cols) / len(data.columns)) * 0.2
                )
            else:
                accessibility_score = 0.0

            return ChartValidation(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                suggestions=suggestions,
                performance_score=performance_score,
                accessibility_score=accessibility_score,
            )

        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}")
            return ChartValidation(
                is_valid=False,
                errors=[f"Validation error: {str(e)}"],
                warnings=[],
                suggestions=[],
                performance_score=0.0,
                accessibility_score=0.0,
            )

    @staticmethod
    def clean_data(
        data: pd.DataFrame,
        remove_duplicates: bool = True,
        fill_missing: Optional[str] = None,
        remove_outliers: bool = False,
        outlier_method: str = "iqr",
    ) -> pd.DataFrame:
        """Clean and preprocess data.

        Args:
            data: DataFrame to clean
            remove_duplicates: Remove duplicate rows
            fill_missing: Method to fill missing values ('mean', 'median', 'mode', 'forward', 'backward')
            remove_outliers: Remove statistical outliers
            outlier_method: Method for outlier detection ('iqr', 'zscore')

        Returns:
            Cleaned DataFrame
        """
        try:
            cleaned_data = data.copy()

            # Remove duplicates
            if remove_duplicates:
                initial_len = len(cleaned_data)
                cleaned_data = cleaned_data.drop_duplicates()
                removed = initial_len - len(cleaned_data)
                if removed > 0:
                    logger.info(f"Removed {removed} duplicate rows")

            # Handle missing values
            if fill_missing:
                numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns

                if fill_missing == "mean":
                    cleaned_data[numeric_cols] = cleaned_data[numeric_cols].fillna(
                        cleaned_data[numeric_cols].mean()
                    )
                elif fill_missing == "median":
                    cleaned_data[numeric_cols] = cleaned_data[numeric_cols].fillna(
                        cleaned_data[numeric_cols].median()
                    )
                elif fill_missing == "mode":
                    for col in cleaned_data.columns:
                        mode_val = cleaned_data[col].mode()
                        if not mode_val.empty:
                            cleaned_data[col] = cleaned_data[col].fillna(mode_val.iloc[0])
                elif fill_missing == "forward":
                    cleaned_data = cleaned_data.fillna(method="ffill")
                elif fill_missing == "backward":
                    cleaned_data = cleaned_data.fillna(method="bfill")

            # Remove outliers
            if remove_outliers:
                numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns

                for col in numeric_cols:
                    if outlier_method == "iqr":
                        Q1 = cleaned_data[col].quantile(0.25)
                        Q3 = cleaned_data[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR

                        cleaned_data = cleaned_data[
                            (cleaned_data[col] >= lower_bound) & (cleaned_data[col] <= upper_bound)
                        ]

                    elif outlier_method == "zscore":
                        z_scores = np.abs(
                            (cleaned_data[col] - cleaned_data[col].mean()) / cleaned_data[col].std()
                        )
                        cleaned_data = cleaned_data[z_scores < 3]

            logger.info(f"Data cleaning completed: {len(data)} -> {len(cleaned_data)} rows")
            return cleaned_data

        except Exception as e:
            logger.error(f"Data cleaning failed: {str(e)}")
            raise DataFormatError(f"Data cleaning failed: {str(e)}") from e

    @staticmethod
    def resample_data(
        data: pd.DataFrame, time_column: str, frequency: str, aggregation: str = "mean"
    ) -> pd.DataFrame:
        """Resample time series data.

        Args:
            data: DataFrame with time series data
            time_column: Name of time column
            frequency: Resampling frequency ('1min', '1H', '1D', etc.)
            aggregation: Aggregation method ('mean', 'sum', 'min', 'max', 'first', 'last')

        Returns:
            Resampled DataFrame
        """
        try:
            if time_column not in data.columns:
                raise DataFormatError(f"Time column '{time_column}' not found")

            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(data[time_column]):
                data[time_column] = pd.to_datetime(data[time_column])

            # Set time column as index
            resampled_data = data.set_index(time_column)

            # Resample based on aggregation method
            if aggregation == "mean":
                resampled_data = resampled_data.resample(frequency).mean()
            elif aggregation == "sum":
                resampled_data = resampled_data.resample(frequency).sum()
            elif aggregation == "min":
                resampled_data = resampled_data.resample(frequency).min()
            elif aggregation == "max":
                resampled_data = resampled_data.resample(frequency).max()
            elif aggregation == "first":
                resampled_data = resampled_data.resample(frequency).first()
            elif aggregation == "last":
                resampled_data = resampled_data.resample(frequency).last()
            else:
                raise ValueError(f"Unsupported aggregation method: {aggregation}")

            # Reset index
            resampled_data = resampled_data.reset_index()

            logger.info(f"Data resampled: {len(data)} -> {len(resampled_data)} rows")
            return resampled_data

        except Exception as e:
            logger.error(f"Data resampling failed: {str(e)}")
            raise DataFormatError(f"Data resampling failed: {str(e)}") from e

    @staticmethod
    def calculate_statistics(data: pd.DataFrame) -> dict[str, Any]:
        """Calculate comprehensive statistics for DataFrame.

        Args:
            data: DataFrame to analyze

        Returns:
            Dictionary of statistics
        """
        try:
            numeric_data = data.select_dtypes(include=[np.number])

            stats = {
                "shape": data.shape,
                "memory_usage_mb": data.memory_usage(deep=True).sum() / 1024 / 1024,
                "missing_values": data.isnull().sum().to_dict(),
                "data_types": data.dtypes.astype(str).to_dict(),
            }

            if not numeric_data.empty:
                stats.update(
                    {
                        "numeric_summary": numeric_data.describe().to_dict(),
                        "correlation_matrix": numeric_data.corr().to_dict(),
                        "skewness": numeric_data.skew().to_dict(),
                        "kurtosis": numeric_data.kurtosis().to_dict(),
                    }
                )

            return stats

        except Exception as e:
            logger.error(f"Statistics calculation failed: {str(e)}")
            return {"error": str(e)}


class ChartExporter:
    """Utility class for chart export operations."""

    def __init__(self):
        """Initialize chart exporter."""
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available. Export functionality limited.")

    def export_figure(self, figure: Any, export_config: ExportConfig) -> str:
        """Export Plotly figure to file.

        Args:
            figure: Plotly figure to export
            export_config: Export configuration

        Returns:
            Path to exported file
        """
        if not PLOTLY_AVAILABLE:
            raise ExportError("Plotly not available for export")

        try:
            filename = export_config.filename or f"chart.{export_config.format.value}"

            if export_config.format == ExportFormat.PNG:
                figure.write_image(
                    filename,
                    format="png",
                    width=export_config.width,
                    height=export_config.height,
                    scale=export_config.scale,
                )

            elif export_config.format == ExportFormat.JPEG:
                figure.write_image(
                    filename,
                    format="jpeg",
                    width=export_config.width,
                    height=export_config.height,
                    scale=export_config.scale,
                )

            elif export_config.format == ExportFormat.SVG:
                figure.write_image(
                    filename,
                    format="svg",
                    width=export_config.width,
                    height=export_config.height,
                    scale=export_config.scale,
                )

            elif export_config.format == ExportFormat.PDF:
                figure.write_image(
                    filename,
                    format="pdf",
                    width=export_config.width,
                    height=export_config.height,
                    scale=export_config.scale,
                )

            elif export_config.format == ExportFormat.HTML:
                figure.write_html(
                    filename,
                    include_plotlyjs=export_config.include_plotlyjs,
                    config=export_config.config,
                )

            elif export_config.format == ExportFormat.JSON:
                figure.write_json(filename)

            else:
                raise ExportError(f"Unsupported export format: {export_config.format}")

            logger.info(f"Figure exported to: {filename}")
            return filename

        except Exception as e:
            logger.error(f"Figure export failed: {str(e)}")
            raise ExportError(f"Export failed: {str(e)}") from e

    def create_export_batch(
        self, figures: list[Any], export_configs: list[ExportConfig]
    ) -> list[str]:
        """Export multiple figures in batch.

        Args:
            figures: List of Plotly figures
            export_configs: List of export configurations

        Returns:
            List of exported file paths
        """
        if len(figures) != len(export_configs):
            raise ExportError("Number of figures must match number of export configs")

        exported_files = []

        for i, (figure, config) in enumerate(zip(figures, export_configs)):
            try:
                filename = self.export_figure(figure, config)
                exported_files.append(filename)
            except Exception as e:
                logger.error(f"Failed to export figure {i}: {str(e)}")
                exported_files.append(None)

        return exported_files


class ColorPalette:
    """Utility class for color management and palette generation."""

    # Predefined color palettes
    PALETTES = {
        "default": [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ],
        "pastel": [
            "#AEC7E8",
            "#FFBB78",
            "#98DF8A",
            "#FF9896",
            "#C5B0D5",
            "#C49C94",
            "#F7B6D3",
            "#C7C7C7",
            "#DBDB8D",
            "#9EDAE5",
        ],
        "bright": [
            "#023EFF",
            "#FF7C00",
            "#1AC938",
            "#E8000B",
            "#8B2BE2",
            "#9F4800",
            "#F14CC1",
            "#A3A3A3",
            "#FFC400",
            "#00D7FF",
        ],
        "earth": [
            "#8B4513",
            "#D2691E",
            "#CD853F",
            "#DEB887",
            "#F4A460",
            "#D2B48C",
            "#BC8F8F",
            "#F5DEB3",
            "#FFE4B5",
            "#FFDAB9",
        ],
        "ocean": [
            "#000080",
            "#0000CD",
            "#4169E1",
            "#1E90FF",
            "#00BFFF",
            "#87CEEB",
            "#87CEFA",
            "#B0E0E6",
            "#ADD8E6",
            "#E0F6FF",
        ],
        "battery": [
            "#FF6B35",
            "#F7931E",
            "#FFD23F",
            "#06FFA5",
            "#118AB2",
            "#073B4C",
            "#EF476F",
            "#F78C6B",
            "#FFD166",
            "#06D6A0",
        ],
    }

    @classmethod
    def get_palette(cls, name: str) -> list[str]:
        """Get color palette by name.

        Args:
            name: Palette name

        Returns:
            List of color hex codes
        """
        return cls.PALETTES.get(name, cls.PALETTES["default"])

    @classmethod
    def generate_gradient(cls, start_color: str, end_color: str, steps: int = 10) -> list[str]:
        """Generate gradient color palette.

        Args:
            start_color: Starting color (hex)
            end_color: Ending color (hex)
            steps: Number of gradient steps

        Returns:
            List of gradient colors
        """
        try:
            # Convert hex to RGB
            start_rgb = cls._hex_to_rgb(start_color)
            end_rgb = cls._hex_to_rgb(end_color)

            # Generate gradient
            colors = []
            for i in range(steps):
                ratio = i / (steps - 1) if steps > 1 else 0

                r = int(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * ratio)
                g = int(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * ratio)
                b = int(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * ratio)

                colors.append(cls._rgb_to_hex(r, g, b))

            return colors

        except Exception as e:
            logger.error(f"Gradient generation failed: {str(e)}")
            return [start_color, end_color]

    @classmethod
    def generate_complementary(cls, base_color: str, count: int = 5) -> list[str]:
        """Generate complementary color palette.

        Args:
            base_color: Base color (hex)
            count: Number of colors to generate

        Returns:
            List of complementary colors
        """
        try:
            # Convert to HSV for easier manipulation
            rgb = cls._hex_to_rgb(base_color)
            hsv = cls._rgb_to_hsv(*rgb)

            colors = [base_color]

            # Generate colors by rotating hue
            for i in range(1, count):
                hue_shift = (360 / count) * i
                new_hue = (hsv[0] + hue_shift) % 360
                new_rgb = cls._hsv_to_rgb(new_hue, hsv[1], hsv[2])
                colors.append(cls._rgb_to_hex(*new_rgb))

            return colors

        except Exception as e:
            logger.error(f"Complementary palette generation failed: {str(e)}")
            return [base_color] * count

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
        """Convert hex color to RGB."""
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

    @staticmethod
    def _rgb_to_hex(r: int, g: int, b: int) -> str:
        """Convert RGB to hex color."""
        return f"#{r:02x}{g:02x}{b:02x}"

    @staticmethod
    def _rgb_to_hsv(r: int, g: int, b: int) -> tuple[float, float, float]:
        """Convert RGB to HSV."""
        r, g, b = r / 255.0, g / 255.0, b / 255.0
        mx = max(r, g, b)
        mn = min(r, g, b)
        df = mx - mn

        if mx == mn:
            h = 0
        elif mx == r:
            h = (60 * ((g - b) / df) + 360) % 360
        elif mx == g:
            h = (60 * ((b - r) / df) + 120) % 360
        elif mx == b:
            h = (60 * ((r - g) / df) + 240) % 360

        s = 0 if mx == 0 else df / mx
        v = mx

        return h, s, v

    @staticmethod
    def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[int, int, int]:
        """Convert HSV to RGB."""
        c = v * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = v - c

        if 0 <= h < 60:
            r, g, b = c, x, 0
        elif 60 <= h < 120:
            r, g, b = x, c, 0
        elif 120 <= h < 180:
            r, g, b = 0, c, x
        elif 180 <= h < 240:
            r, g, b = 0, x, c
        elif 240 <= h < 300:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x

        return int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)


class LayoutManager:
    """Utility class for chart layout management."""

    @staticmethod
    def calculate_optimal_size(
        data_points: int,
        chart_type: str,
        container_width: Optional[int] = None,
        container_height: Optional[int] = None,
    ) -> tuple[int, int]:
        """Calculate optimal chart size based on data and constraints.

        Args:
            data_points: Number of data points
            chart_type: Type of chart
            container_width: Container width constraint
            container_height: Container height constraint

        Returns:
            Optimal width and height
        """
        # Base sizes by chart type
        base_sizes = {
            "line": (800, 500),
            "scatter": (700, 500),
            "bar": (600, 400),
            "heatmap": (600, 600),
            "histogram": (600, 400),
            "box": (500, 400),
            "violin": (500, 400),
            "surface": (700, 700),
            "contour": (600, 600),
        }

        base_width, base_height = base_sizes.get(chart_type, (800, 500))

        # Adjust for data density
        if data_points > 10000:
            base_width = int(base_width * 1.2)
            base_height = int(base_height * 1.1)
        elif data_points < 100:
            base_width = int(base_width * 0.8)
            base_height = int(base_height * 0.9)

        # Apply container constraints
        if container_width:
            base_width = min(base_width, container_width)
        if container_height:
            base_height = min(base_height, container_height)

        return base_width, base_height

    @staticmethod
    def calculate_margins(
        title: Optional[str] = None,
        x_axis_title: Optional[str] = None,
        y_axis_title: Optional[str] = None,
        legend: bool = False,
        legend_position: str = "right",
    ) -> dict[str, int]:
        """Calculate optimal margins for chart layout.

        Args:
            title: Chart title
            x_axis_title: X-axis title
            y_axis_title: Y-axis title
            legend: Whether legend is shown
            legend_position: Legend position

        Returns:
            Dictionary of margin values
        """
        margins = {"l": 50, "r": 50, "t": 50, "b": 50}

        # Adjust for title
        if title:
            margins["t"] += 30

        # Adjust for axis titles
        if y_axis_title:
            margins["l"] += 20
        if x_axis_title:
            margins["b"] += 20

        # Adjust for legend
        if legend:
            if legend_position == "right":
                margins["r"] += 100
            elif legend_position == "left":
                margins["l"] += 100
            elif legend_position == "top":
                margins["t"] += 50
            elif legend_position == "bottom":
                margins["b"] += 50

        return margins


class ValidationUtils:
    """Utility class for chart validation and quality checks."""

    @staticmethod
    def validate_chart_config(config: ChartConfig) -> ChartValidation:
        """Validate chart configuration.

        Args:
            config: Chart configuration to validate

        Returns:
            Validation result
        """
        errors = []
        warnings = []
        suggestions = []

        try:
            # Validate data
            if not config.data.x or not config.data.y:
                errors.append("Chart data must have both x and y values")

            if len(config.data.x) != len(config.data.y):
                errors.append("X and Y data must have the same length")

            # Check data size
            data_size = len(config.data.x)
            if data_size > 50000:
                warnings.append(f"Large dataset ({data_size} points) may impact performance")
                suggestions.append("Consider data sampling or aggregation")

            # Validate color data
            if config.data.color and len(config.data.color) != len(config.data.x):
                errors.append("Color data must have same length as x/y data")

            # Validate size data
            if config.data.size and len(config.data.size) != len(config.data.x):
                errors.append("Size data must have same length as x/y data")

            # Check for reasonable axis ranges
            if config.x_axis.range:
                x_min, x_max = config.x_axis.range
                if x_min >= x_max:
                    errors.append("X-axis range minimum must be less than maximum")

            if config.y_axis.range:
                y_min, y_max = config.y_axis.range
                if y_min >= y_max:
                    errors.append("Y-axis range minimum must be less than maximum")

            # Validate style parameters
            if config.style.opacity < 0 or config.style.opacity > 1:
                errors.append("Opacity must be between 0 and 1")

            if config.style.line_width < 0:
                errors.append("Line width must be positive")

            if config.style.marker_size < 0:
                errors.append("Marker size must be positive")

            # Performance scoring
            performance_factors = [
                1.0 - min(data_size / 100000, 1.0),  # Data size factor
                1.0 if not config.data.color else 0.9,  # Color mapping factor
                (
                    1.0
                    if not config.animation.type or config.animation.type.value == "none"
                    else 0.8
                ),  # Animation factor
            ]
            performance_score = np.mean(performance_factors)

            # Accessibility scoring
            accessibility_factors = [
                1.0 if config.x_axis.title else 0.8,  # X-axis title
                1.0 if config.y_axis.title else 0.8,  # Y-axis title
                1.0 if config.layout.title else 0.9,  # Chart title
                1.0 if config.layout.show_legend and config.data.color else 1.0,  # Legend
            ]
            accessibility_score = np.mean(accessibility_factors)

            return ChartValidation(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                suggestions=suggestions,
                performance_score=performance_score,
                accessibility_score=accessibility_score,
            )

        except Exception as e:
            logger.error(f"Chart validation failed: {str(e)}")
            return ChartValidation(
                is_valid=False,
                errors=[f"Validation error: {str(e)}"],
                warnings=[],
                suggestions=[],
                performance_score=0.0,
                accessibility_score=0.0,
            )

    @staticmethod
    def check_performance_limits(config: ChartConfig) -> None:
        """Check if chart configuration exceeds performance limits.

        Args:
            config: Chart configuration to check

        Raises:
            PerformanceError: If performance limits are exceeded
        """
        data_size = len(config.data.x)

        # Check data size limits
        if data_size > 100000:
            raise PerformanceError(
                f"Data size ({data_size}) exceeds recommended limit",
                metric_name="data_size",
                current_value=data_size,
                limit_value=100000,
                suggestion="Consider data sampling or aggregation",
            )

        # Check memory usage estimate
        estimated_memory = data_size * 8 * 4  # Rough estimate for 4 arrays of floats
        if estimated_memory > 100 * 1024 * 1024:  # 100 MB
            raise PerformanceError(
                f"Estimated memory usage ({estimated_memory / 1024 / 1024:.1f} MB) is high",
                metric_name="memory_usage",
                current_value=estimated_memory,
                limit_value=100 * 1024 * 1024,
                suggestion="Reduce data size or use data streaming",
            )

    @staticmethod
    def generate_chart_hash(config: ChartConfig) -> str:
        """Generate hash for chart configuration.

        Args:
            config: Chart configuration

        Returns:
            SHA-256 hash of configuration
        """
        try:
            # Convert config to JSON string (excluding dynamic fields)
            config_dict = config.dict()
            config_dict.pop("id", None)  # Remove ID for consistent hashing

            config_str = json.dumps(config_dict, sort_keys=True, default=str)
            return hashlib.sha256(config_str.encode()).hexdigest()

        except Exception as e:
            logger.error(f"Hash generation failed: {str(e)}")
            return hashlib.sha256(str(datetime.now()).encode()).hexdigest()
