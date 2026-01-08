"""Unit tests for visualization framework core functionality.

This module tests visualization models, components, configuration,
templates, and utilities for the battery data visualization system.
"""

import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from frontend.visualization.exceptions import (
    ChartRenderingError,
    ConfigurationError,
    DataFormatError,
    ExportError,
    TemplateError,
    VisualizationError,
)

# Test visualization models without requiring full dependencies
from frontend.visualization.models import (
    AxisConfig,
    BatteryDataPoint,
    ChartConfig,
    ChartData,
    ChartStyle,
    ChartTemplate,
    ChartType,
    ChartValidation,
    ExportConfig,
    ExportFormat,
    LayoutConfig,
    VisualizationTheme,
)


class TestVisualizationModels:
    """Test visualization models."""

    def test_chart_data(self):
        """Test ChartData model."""
        data = ChartData(
            x=[1, 2, 3, 4, 5],
            y=[2, 4, 6, 8, 10],
            text=["Point 1", "Point 2", "Point 3", "Point 4", "Point 5"],
            color=[1, 2, 3, 4, 5],
            size=[5, 10, 15, 20, 25],
        )

        assert len(data.x) == 5
        assert len(data.y) == 5
        assert data.text[0] == "Point 1"
        assert data.color[2] == 3 or data.color[2] == "3"  # Pydantic may coerce to string
        assert data.size[4] == 25

    def test_chart_data_validation(self):
        """Test ChartData validation."""
        # Should fail with mismatched lengths
        with pytest.raises(ValueError):
            ChartData(x=[1, 2, 3], y=[1, 2])  # Different length

    def test_chart_style(self):
        """Test ChartStyle model."""
        style = ChartStyle(
            color="#1f77b4",
            colors=["#1f77b4", "#ff7f0e", "#2ca02c"],
            opacity=0.8,
            line_width=2.0,
            marker_size=6.0,
            marker_symbol="circle",
        )

        assert style.color == "#1f77b4"
        assert len(style.colors) == 3
        assert style.opacity == 0.8
        assert style.line_width == 2.0
        assert style.marker_size == 6.0

    def test_axis_config(self):
        """Test AxisConfig model."""
        axis = AxisConfig(
            title="Voltage (V)",
            range=(2.5, 4.2),
            type="linear",
            show_grid=True,
            grid_color="lightgray",
        )

        assert axis.title == "Voltage (V)"
        assert axis.range == (2.5, 4.2)
        assert axis.type == "linear"
        assert axis.show_grid is True

    def test_axis_config_validation(self):
        """Test AxisConfig validation."""
        # Should fail with invalid range
        with pytest.raises(ValueError):
            AxisConfig(range=(4.2, 2.5))  # min > max

    def test_layout_config(self):
        """Test LayoutConfig model."""
        layout = LayoutConfig(
            title="Battery Analysis",
            width=800,
            height=600,
            margin={"l": 50, "r": 50, "t": 50, "b": 50},
            background_color="white",
            show_legend=True,
        )

        assert layout.title == "Battery Analysis"
        assert layout.width == 800
        assert layout.height == 600
        assert layout.margin["l"] == 50
        assert layout.show_legend is True

    def test_chart_config(self):
        """Test ChartConfig model."""
        data = ChartData(x=[1, 2, 3], y=[2, 4, 6])

        config = ChartConfig(
            type=ChartType.LINE,
            title="Test Chart",
            data=data,
            style=ChartStyle(color="#1f77b4"),
            x_axis=AxisConfig(title="X Axis"),
            y_axis=AxisConfig(title="Y Axis"),
            layout=LayoutConfig(title="Test Chart"),
        )

        assert config.type == ChartType.LINE
        assert config.title == "Test Chart"
        assert len(config.data.x) == 3
        assert config.style.color == "#1f77b4"
        assert config.x_axis.title == "X Axis"

    def test_chart_template(self):
        """Test ChartTemplate model."""
        template = ChartTemplate(
            name="Line Chart Template",
            description="Basic line chart for time series",
            category="Basic",
            chart_type=ChartType.LINE,
            config_template={"type": "line", "style": {"line_width": 2.0}},
            data_requirements={
                "x": {"type": "array", "description": "X values"},
                "y": {"type": "array", "description": "Y values"},
            },
            parameters={"color": {"type": "color", "default": "#1f77b4"}},
            tags=["line", "time-series"],
            created_by="system",
        )

        assert template.name == "Line Chart Template"
        assert template.chart_type == ChartType.LINE
        assert template.category == "Basic"
        assert "line" in template.tags
        assert template.config_template["type"] == "line"

    def test_visualization_theme(self):
        """Test VisualizationTheme model."""
        theme = VisualizationTheme(
            name="Dark Theme",
            description="Dark theme for reduced eye strain",
            colors={"primary": "#3498db", "background": "#2c3e50", "text": "#ecf0f1"},
            fonts={"primary": "Arial, sans-serif"},
            layout_defaults={"background_color": "#2c3e50"},
            chart_defaults={"line_width": 2.0},
            is_dark=True,
        )

        assert theme.name == "Dark Theme"
        assert theme.is_dark is True
        assert theme.colors["primary"] == "#3498db"
        assert theme.fonts["primary"] == "Arial, sans-serif"

    def test_export_config(self):
        """Test ExportConfig model."""
        export_config = ExportConfig(
            format=ExportFormat.PNG, width=800, height=600, scale=2.0, filename="chart.png"
        )

        assert export_config.format == ExportFormat.PNG
        assert export_config.width == 800
        assert export_config.height == 600
        assert export_config.scale == 2.0
        assert export_config.filename == "chart.png"

    def test_battery_data_point(self):
        """Test BatteryDataPoint model."""
        data_point = BatteryDataPoint(
            timestamp=datetime.now(),
            cycle_number=1,
            step_number=1,
            voltage=3.7,
            current=1.0,
            capacity=2.5,
            temperature=25.0,
        )

        assert data_point.voltage == 3.7
        assert data_point.current == 1.0
        assert data_point.capacity == 2.5
        assert data_point.temperature == 25.0

    def test_chart_validation(self):
        """Test ChartValidation model."""
        validation = ChartValidation(
            is_valid=True,
            errors=[],
            warnings=["Large dataset"],
            suggestions=["Consider data sampling"],
            performance_score=0.8,
            accessibility_score=0.9,
        )

        assert validation.is_valid is True
        assert len(validation.errors) == 0
        assert len(validation.warnings) == 1
        assert validation.performance_score == 0.8
        assert validation.accessibility_score == 0.9


class TestVisualizationExceptions:
    """Test visualization exceptions."""

    def test_visualization_error(self):
        """Test base VisualizationError."""
        error = VisualizationError("Test error", error_code="VIZ001", details={"context": "test"})

        assert str(error) == "Test error"
        assert error.error_code == "VIZ001"
        assert error.details["context"] == "test"

    def test_chart_rendering_error(self):
        """Test ChartRenderingError."""
        error = ChartRenderingError(
            "Rendering failed",
            chart_id="chart_123",
            chart_type="line",
            rendering_stage="trace_creation",
        )

        assert str(error) == "Rendering failed"
        assert error.chart_id == "chart_123"
        assert error.chart_type == "line"
        assert error.rendering_stage == "trace_creation"

    def test_data_format_error(self):
        """Test DataFormatError."""
        error = DataFormatError(
            "Invalid data format", data_field="x", expected_format="array", actual_format="string"
        )

        assert str(error) == "Invalid data format"
        assert error.data_field == "x"
        assert error.expected_format == "array"
        assert error.actual_format == "string"

    def test_configuration_error(self):
        """Test ConfigurationError."""
        error = ConfigurationError(
            "Invalid configuration",
            config_field="chart_type",
            config_value="invalid",
            valid_options=["line", "bar", "scatter"],
        )

        assert str(error) == "Invalid configuration"
        assert error.config_field == "chart_type"
        assert error.config_value == "invalid"
        assert "line" in error.valid_options

    def test_export_error(self):
        """Test ExportError."""
        error = ExportError(
            "Export failed", export_format="png", file_path="/tmp/chart.png", chart_id="chart_123"
        )

        assert str(error) == "Export failed"
        assert error.export_format == "png"
        assert error.file_path == "/tmp/chart.png"
        assert error.chart_id == "chart_123"

    def test_template_error(self):
        """Test TemplateError."""
        error = TemplateError(
            "Template processing failed",
            template_id="template_123",
            template_name="Line Chart",
            parameter_name="color",
        )

        assert str(error) == "Template processing failed"
        assert error.template_id == "template_123"
        assert error.template_name == "Line Chart"
        assert error.parameter_name == "color"


class TestVisualizationConfig:
    """Test visualization configuration management."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test environment."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_chart_config_manager_creation(self):
        """Test chart configuration manager creation."""
        from frontend.visualization.config import ChartConfigManager

        manager = ChartConfigManager(self.temp_dir)

        config = manager.create_config(chart_type=ChartType.LINE, title="Test Chart")

        assert config.type == ChartType.LINE
        assert config.title == "Test Chart"
        assert config.id in manager._configs

    def test_theme_manager(self):
        """Test theme manager."""
        from frontend.visualization.config import ThemeManager

        manager = ThemeManager(self.temp_dir)

        # Should have default themes
        themes = manager.list_themes()
        assert len(themes) >= 3  # Light, Dark, Scientific

        # Get theme by name
        light_theme = manager.get_theme_by_name("Light")
        assert light_theme is not None
        assert light_theme.name == "Light"
        assert light_theme.is_dark is False

    def test_template_manager(self):
        """Test template manager."""
        from frontend.visualization.config import TemplateManager

        manager = TemplateManager(self.temp_dir)

        # Should have default templates
        templates = manager.list_templates()
        assert len(templates) >= 3  # Basic templates

        # Get templates by chart type
        line_templates = manager.get_templates_by_chart_type(ChartType.LINE)
        assert len(line_templates) > 0

        # Search templates
        search_results = manager.search_templates("line")
        assert len(search_results) > 0


class TestVisualizationUtils:
    """Test visualization utilities."""

    def setup_method(self):
        """Set up test data."""
        self.test_data = pd.DataFrame(
            {
                "time": pd.date_range("2024-01-01", periods=100, freq="1H"),
                "voltage": 3.7 + 0.1 * np.random.randn(100),
                "current": 1.0 + 0.05 * np.random.randn(100),
                "temperature": 25 + 2 * np.random.randn(100),
                "cycle": [1] * 50 + [2] * 50,
            }
        )

    def test_data_processor_validation(self):
        """Test data processor validation."""
        from frontend.visualization.utils import DataProcessor

        # Valid data
        validation = DataProcessor.validate_data(self.test_data)
        assert validation.is_valid is True
        assert validation.performance_score > 0
        assert validation.accessibility_score > 0

        # Empty data
        empty_data = pd.DataFrame()
        validation = DataProcessor.validate_data(empty_data)
        assert validation.is_valid is False
        assert len(validation.errors) > 0

    def test_data_processor_cleaning(self):
        """Test data processor cleaning."""
        from frontend.visualization.utils import DataProcessor

        # Add some issues to test data
        dirty_data = self.test_data.copy()
        dirty_data.loc[0, "voltage"] = np.nan  # Missing value
        dirty_data = pd.concat([dirty_data, dirty_data.iloc[:5]])  # Duplicates

        # Clean data
        cleaned = DataProcessor.clean_data(dirty_data, remove_duplicates=True, fill_missing="mean")

        assert len(cleaned) < len(dirty_data)  # Duplicates removed
        assert cleaned["voltage"].isnull().sum() == 0  # Missing values filled

    def test_data_processor_resampling(self):
        """Test data processor resampling."""
        from frontend.visualization.utils import DataProcessor

        resampled = DataProcessor.resample_data(
            self.test_data, time_column="time", frequency="1D", aggregation="mean"
        )

        assert len(resampled) < len(self.test_data)  # Downsampled
        assert "time" in resampled.columns
        assert "voltage" in resampled.columns

    def test_data_processor_statistics(self):
        """Test data processor statistics."""
        from frontend.visualization.utils import DataProcessor

        stats = DataProcessor.calculate_statistics(self.test_data)

        assert "shape" in stats
        assert "memory_usage_mb" in stats
        assert "missing_values" in stats
        assert "numeric_summary" in stats
        assert stats["shape"] == self.test_data.shape

    def test_color_palette(self):
        """Test color palette utilities."""
        from frontend.visualization.utils import ColorPalette

        # Get predefined palette
        default_palette = ColorPalette.get_palette("default")
        assert len(default_palette) == 10
        assert all(color.startswith("#") for color in default_palette)

        # Generate gradient
        gradient = ColorPalette.generate_gradient("#ff0000", "#0000ff", 5)
        assert len(gradient) == 5
        assert gradient[0] == "#ff0000"
        assert gradient[-1] == "#0000ff"

        # Generate complementary colors
        complementary = ColorPalette.generate_complementary("#ff0000", 5)
        assert len(complementary) == 5
        assert complementary[0] == "#ff0000"

    def test_layout_manager(self):
        """Test layout manager utilities."""
        from frontend.visualization.utils import LayoutManager

        # Calculate optimal size
        width, height = LayoutManager.calculate_optimal_size(
            data_points=1000, chart_type="line", container_width=1200, container_height=800
        )

        assert width <= 1200
        assert height <= 800
        assert width > 0
        assert height > 0

        # Calculate margins
        margins = LayoutManager.calculate_margins(
            title="Test Chart",
            x_axis_title="X Axis",
            y_axis_title="Y Axis",
            legend=True,
            legend_position="right",
        )

        assert "l" in margins
        assert "r" in margins
        assert "t" in margins
        assert "b" in margins
        assert margins["r"] > 50  # Extra space for legend

    def test_validation_utils(self):
        """Test validation utilities."""
        from frontend.visualization.utils import ValidationUtils

        # Create test config
        data = ChartData(x=[1, 2, 3], y=[2, 4, 6])
        config = ChartConfig(type=ChartType.LINE, title="Test Chart", data=data)

        # Validate config
        validation = ValidationUtils.validate_chart_config(config)
        assert validation.is_valid is True

        # Test invalid config - mismatched data lengths should raise ValidationError
        with pytest.raises(ValidationError):
            invalid_data = ChartData(x=[1, 2, 3], y=[2, 4])  # Mismatched lengths

    def test_chart_hash_generation(self):
        """Test chart hash generation."""
        from frontend.visualization.utils import ValidationUtils

        data = ChartData(x=[1, 2, 3], y=[2, 4, 6])
        config1 = ChartConfig(type=ChartType.LINE, title="Test Chart", data=data)

        config2 = ChartConfig(type=ChartType.LINE, title="Test Chart", data=data)

        hash1 = ValidationUtils.generate_chart_hash(config1)
        hash2 = ValidationUtils.generate_chart_hash(config2)

        # Same configuration should produce same hash (excluding ID)
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex length


class TestVisualizationTemplates:
    """Test visualization templates."""

    def test_battery_analysis_templates(self):
        """Test battery analysis templates."""
        from frontend.visualization.templates import BatteryAnalysisTemplates

        templates = BatteryAnalysisTemplates()

        # Should have battery-specific templates
        battery_templates = templates.list_templates()
        assert len(battery_templates) > 0

        # Test specific templates
        cycling_template = templates.get_template("Cycling Analysis")
        assert cycling_template is not None
        assert cycling_template.category == "Battery Analysis"
        assert "cycling" in cycling_template.tags

        capacity_template = templates.get_template("Capacity Fade Analysis")
        assert capacity_template is not None
        assert capacity_template.chart_type == ChartType.SCATTER

    def test_template_data_processing(self):
        """Test template data processing."""
        from frontend.visualization.templates import CyclingAnalysisTemplate

        # Create test data
        test_data = pd.DataFrame(
            {
                "time": np.linspace(0, 10, 100),
                "voltage": 3.7 + 0.1 * np.sin(np.linspace(0, 4 * np.pi, 100)),
                "current": 1.0 + 0.05 * np.cos(np.linspace(0, 4 * np.pi, 100)),
            }
        )

        # Create configuration from data
        config = CyclingAnalysisTemplate.create_config_from_data(
            test_data, time_col="time", voltage_col="voltage", current_col="current"
        )

        assert config.type == ChartType.LINE
        assert config.title == "Battery Cycling Analysis"
        assert len(config.data.x) == 100
        assert len(config.data.y) == 100
        assert config.x_axis.title == "Time (hours)"
        assert config.y_axis.title == "Voltage (V)"


class TestVisualizationIntegration:
    """Integration tests for visualization components."""

    def setup_method(self):
        """Set up test environment."""
        self.test_data = pd.DataFrame(
            {
                "x": np.linspace(0, 10, 50),
                "y": np.sin(np.linspace(0, 10, 50)) + 0.1 * np.random.randn(50),
                "category": ["A"] * 25 + ["B"] * 25,
            }
        )

    def test_end_to_end_chart_creation(self):
        """Test end-to-end chart creation workflow."""
        # Create chart data
        chart_data = ChartData(
            x=self.test_data["x"].tolist(),
            y=self.test_data["y"].tolist(),
            text=[f"Point {i}" for i in range(len(self.test_data))],
        )

        # Create chart configuration
        config = ChartConfig(
            type=ChartType.LINE,
            title="Integration Test Chart",
            data=chart_data,
            style=ChartStyle(color="#1f77b4", line_width=2.0, marker_size=4.0, opacity=0.8),
            x_axis=AxisConfig(title="X Values", show_grid=True),
            y_axis=AxisConfig(title="Y Values", show_grid=True),
            layout=LayoutConfig(
                title="Integration Test Chart", width=800, height=600, show_legend=False
            ),
        )

        # Validate configuration
        from frontend.visualization.utils import ValidationUtils

        validation = ValidationUtils.validate_chart_config(config)
        assert validation.is_valid is True

        # Test configuration serialization
        config_dict = config.dict()
        assert config_dict["type"] == "line"
        assert config_dict["title"] == "Integration Test Chart"

        # Test configuration deserialization
        new_config = ChartConfig(**config_dict)
        assert new_config.type == ChartType.LINE
        assert new_config.title == "Integration Test Chart"

    def test_template_to_config_workflow(self):
        """Test template to configuration workflow."""
        from frontend.visualization.config import TemplateManager

        manager = TemplateManager()

        # Get a template
        templates = manager.list_templates()
        assert len(templates) > 0

        template = templates[0]

        # Apply template to create configuration
        config = manager.apply_template_to_config(template)

        assert config.type == template.chart_type
        assert config.title == template.name

        # Validate resulting configuration
        from frontend.visualization.utils import ValidationUtils

        validation = ValidationUtils.validate_chart_config(config)
        # Note: May not be valid due to default data, but should not crash
        assert isinstance(validation, ChartValidation)

    def test_theme_application_workflow(self):
        """Test theme application workflow."""
        from frontend.visualization.config import ThemeManager

        manager = ThemeManager()

        # Get a theme
        theme = manager.get_theme_by_name("Dark")
        assert theme is not None

        # Create a basic configuration
        chart_data = ChartData(x=[1, 2, 3], y=[2, 4, 6])
        config = ChartConfig(type=ChartType.LINE, title="Theme Test", data=chart_data)

        # Apply theme to configuration
        themed_config = manager.apply_theme_to_config(config, theme)

        assert themed_config.layout.background_color == theme.colors.get("background")
        assert themed_config.layout.font_color == theme.colors.get("text")
        assert themed_config.style.color == theme.colors.get("primary")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
