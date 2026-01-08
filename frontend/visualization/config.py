"""Configuration management for visualization framework.

This module provides configuration managers for charts, themes, templates,
and styling to enable consistent and customizable visualizations.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional, Union

from .exceptions import ConfigurationError, TemplateError, ThemeError
from .models import (
    AnimationConfig,
    AxisConfig,
    ChartConfig,
    ChartStyle,
    ChartTemplate,
    ChartType,
    InteractionConfig,
    LayoutConfig,
    VisualizationTheme,
)

logger = logging.getLogger(__name__)


class ChartConfigManager:
    """Manager for chart configurations."""

    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize configuration manager.

        Args:
            config_dir: Directory for storing configurations
        """
        self.config_dir = config_dir or Path.cwd() / "chart_configs"
        self.config_dir.mkdir(exist_ok=True)
        self._configs: dict[str, ChartConfig] = {}

    def create_config(
        self, chart_type: ChartType, title: Optional[str] = None, **kwargs
    ) -> ChartConfig:
        """Create a new chart configuration.

        Args:
            chart_type: Type of chart
            title: Chart title
            **kwargs: Additional configuration parameters

        Returns:
            Chart configuration
        """
        try:
            # Create default data structure
            from .models import ChartData

            default_data = ChartData(x=[1, 2, 3], y=[1, 4, 2])

            config = ChartConfig(
                type=chart_type,
                title=title,
                data=kwargs.get("data", default_data),
                style=kwargs.get("style", ChartStyle()),
                x_axis=kwargs.get("x_axis", AxisConfig()),
                y_axis=kwargs.get("y_axis", AxisConfig()),
                z_axis=kwargs.get("z_axis"),
                layout=kwargs.get("layout", LayoutConfig()),
                interaction=kwargs.get("interaction", InteractionConfig()),
                animation=kwargs.get("animation", AnimationConfig()),
                custom_config=kwargs.get("custom_config", {}),
            )

            self._configs[config.id] = config
            logger.info(f"Created chart configuration: {config.id}")
            return config

        except Exception as e:
            logger.error(f"Failed to create chart configuration: {str(e)}")
            raise ConfigurationError(f"Configuration creation failed: {str(e)}")

    def get_config(self, config_id: str) -> Optional[ChartConfig]:
        """Get chart configuration by ID.

        Args:
            config_id: Configuration ID

        Returns:
            Chart configuration or None if not found
        """
        return self._configs.get(config_id)

    def update_config(self, config_id: str, updates: dict[str, Any]) -> ChartConfig:
        """Update chart configuration.

        Args:
            config_id: Configuration ID
            updates: Configuration updates

        Returns:
            Updated chart configuration
        """
        config = self.get_config(config_id)
        if not config:
            raise ConfigurationError(f"Configuration not found: {config_id}")

        try:
            # Update configuration fields
            for field, value in updates.items():
                if hasattr(config, field):
                    setattr(config, field, value)
                else:
                    logger.warning(f"Unknown configuration field: {field}")

            logger.info(f"Updated chart configuration: {config_id}")
            return config

        except Exception as e:
            logger.error(f"Failed to update configuration: {str(e)}")
            raise ConfigurationError(f"Configuration update failed: {str(e)}")

    def delete_config(self, config_id: str) -> bool:
        """Delete chart configuration.

        Args:
            config_id: Configuration ID

        Returns:
            True if deleted successfully
        """
        if config_id in self._configs:
            del self._configs[config_id]
            logger.info(f"Deleted chart configuration: {config_id}")
            return True
        return False

    def list_configs(self) -> list[ChartConfig]:
        """List all chart configurations.

        Returns:
            List of chart configurations
        """
        return list(self._configs.values())

    def save_config(self, config: ChartConfig, filename: Optional[str] = None) -> Path:
        """Save chart configuration to file.

        Args:
            config: Chart configuration
            filename: Output filename

        Returns:
            Path to saved file
        """
        try:
            filename = filename or f"chart_config_{config.id}.json"
            file_path = self.config_dir / filename

            # Convert to dict and save
            config_dict = config.dict()
            with open(file_path, "w") as f:
                json.dump(config_dict, f, indent=2, default=str)

            logger.info(f"Saved chart configuration to: {file_path}")
            return file_path

        except Exception as e:
            logger.error(f"Failed to save configuration: {str(e)}")
            raise ConfigurationError(f"Configuration save failed: {str(e)}")

    def load_config(self, file_path: Union[str, Path]) -> ChartConfig:
        """Load chart configuration from file.

        Args:
            file_path: Path to configuration file

        Returns:
            Chart configuration
        """
        try:
            with open(file_path) as f:
                config_dict = json.load(f)

            config = ChartConfig(**config_dict)
            self._configs[config.id] = config

            logger.info(f"Loaded chart configuration from: {file_path}")
            return config

        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            raise ConfigurationError(f"Configuration load failed: {str(e)}")


class ThemeManager:
    """Manager for visualization themes."""

    def __init__(self, theme_dir: Optional[Path] = None):
        """Initialize theme manager.

        Args:
            theme_dir: Directory for storing themes
        """
        self.theme_dir = theme_dir or Path.cwd() / "themes"
        self.theme_dir.mkdir(exist_ok=True)
        self._themes: dict[str, VisualizationTheme] = {}
        self._load_default_themes()

    def _load_default_themes(self) -> None:
        """Load default themes."""
        # Light theme
        light_theme = VisualizationTheme(
            name="Light",
            description="Clean light theme for professional presentations",
            colors={
                "primary": "#1f77b4",
                "secondary": "#ff7f0e",
                "success": "#2ca02c",
                "warning": "#d62728",
                "info": "#17becf",
                "background": "#ffffff",
                "surface": "#f8f9fa",
                "text": "#212529",
                "grid": "#e9ecef",
            },
            fonts={
                "primary": "Arial, sans-serif",
                "secondary": "Helvetica, sans-serif",
                "monospace": "Consolas, monospace",
            },
            layout_defaults={
                "background_color": "#ffffff",
                "paper_color": "#ffffff",
                "font_color": "#212529",
                "grid_color": "#e9ecef",
            },
            chart_defaults={"line_width": 2.0, "marker_size": 6.0, "opacity": 0.8},
            is_dark=False,
        )

        # Dark theme
        dark_theme = VisualizationTheme(
            name="Dark",
            description="Modern dark theme for reduced eye strain",
            colors={
                "primary": "#3498db",
                "secondary": "#e74c3c",
                "success": "#2ecc71",
                "warning": "#f39c12",
                "info": "#1abc9c",
                "background": "#2c3e50",
                "surface": "#34495e",
                "text": "#ecf0f1",
                "grid": "#7f8c8d",
            },
            fonts={
                "primary": "Arial, sans-serif",
                "secondary": "Helvetica, sans-serif",
                "monospace": "Consolas, monospace",
            },
            layout_defaults={
                "background_color": "#2c3e50",
                "paper_color": "#34495e",
                "font_color": "#ecf0f1",
                "grid_color": "#7f8c8d",
            },
            chart_defaults={"line_width": 2.0, "marker_size": 6.0, "opacity": 0.9},
            is_dark=True,
        )

        # Scientific theme
        scientific_theme = VisualizationTheme(
            name="Scientific",
            description="Professional theme for scientific publications",
            colors={
                "primary": "#2E86AB",
                "secondary": "#A23B72",
                "success": "#F18F01",
                "warning": "#C73E1D",
                "info": "#592E83",
                "background": "#ffffff",
                "surface": "#fafafa",
                "text": "#333333",
                "grid": "#cccccc",
            },
            fonts={
                "primary": "Times New Roman, serif",
                "secondary": "Arial, sans-serif",
                "monospace": "Courier New, monospace",
            },
            layout_defaults={
                "background_color": "#ffffff",
                "paper_color": "#ffffff",
                "font_color": "#333333",
                "grid_color": "#cccccc",
            },
            chart_defaults={"line_width": 1.5, "marker_size": 4.0, "opacity": 0.85},
            is_dark=False,
        )

        self._themes[light_theme.id] = light_theme
        self._themes[dark_theme.id] = dark_theme
        self._themes[scientific_theme.id] = scientific_theme

    def get_theme(self, theme_id: str) -> Optional[VisualizationTheme]:
        """Get theme by ID.

        Args:
            theme_id: Theme ID

        Returns:
            Theme or None if not found
        """
        return self._themes.get(theme_id)

    def get_theme_by_name(self, name: str) -> Optional[VisualizationTheme]:
        """Get theme by name.

        Args:
            name: Theme name

        Returns:
            Theme or None if not found
        """
        for theme in self._themes.values():
            if theme.name.lower() == name.lower():
                return theme
        return None

    def create_theme(self, theme: VisualizationTheme) -> str:
        """Create a new theme.

        Args:
            theme: Theme configuration

        Returns:
            Theme ID
        """
        try:
            self._themes[theme.id] = theme
            logger.info(f"Created theme: {theme.name}")
            return theme.id

        except Exception as e:
            logger.error(f"Failed to create theme: {str(e)}")
            raise ThemeError(f"Theme creation failed: {str(e)}")

    def apply_theme_to_config(self, config: ChartConfig, theme: VisualizationTheme) -> ChartConfig:
        """Apply theme to chart configuration.

        Args:
            config: Chart configuration
            theme: Theme to apply

        Returns:
            Updated chart configuration
        """
        try:
            # Apply layout defaults
            for key, value in theme.layout_defaults.items():
                if hasattr(config.layout, key):
                    setattr(config.layout, key, value)

            # Apply chart defaults
            for key, value in theme.chart_defaults.items():
                if hasattr(config.style, key):
                    setattr(config.style, key, value)

            # Apply color scheme
            if not config.style.color:
                config.style.color = theme.colors.get("primary")

            if not config.style.colors:
                config.style.colors = [
                    theme.colors.get("primary"),
                    theme.colors.get("secondary"),
                    theme.colors.get("success"),
                    theme.colors.get("warning"),
                    theme.colors.get("info"),
                ]

            # Apply font settings
            config.layout.font_family = theme.fonts.get("primary", "Arial")

            logger.info(f"Applied theme {theme.name} to chart {config.id}")
            return config

        except Exception as e:
            logger.error(f"Failed to apply theme: {str(e)}")
            raise ThemeError(f"Theme application failed: {str(e)}")

    def list_themes(self) -> list[VisualizationTheme]:
        """List all available themes.

        Returns:
            List of themes
        """
        return list(self._themes.values())


class TemplateManager:
    """Manager for chart templates."""

    def __init__(self, template_dir: Optional[Path] = None):
        """Initialize template manager.

        Args:
            template_dir: Directory for storing templates
        """
        self.template_dir = template_dir or Path.cwd() / "templates"
        self.template_dir.mkdir(exist_ok=True)
        self._templates: dict[str, ChartTemplate] = {}
        self._load_default_templates()

    def _load_default_templates(self) -> None:
        """Load default chart templates."""
        # Basic line chart template
        line_template = ChartTemplate(
            name="Basic Line Chart",
            description="Simple line chart for time series data",
            category="Basic",
            chart_type=ChartType.LINE,
            config_template={
                "type": "line",
                "style": {"line_width": 2.0, "marker_size": 4.0, "opacity": 0.8},
                "x_axis": {"title": "Time", "type": "linear", "show_grid": True},
                "y_axis": {"title": "Value", "type": "linear", "show_grid": True},
                "layout": {"show_legend": True, "legend_position": "right"},
            },
            data_requirements={
                "x": {"type": "array", "description": "X-axis values"},
                "y": {"type": "array", "description": "Y-axis values"},
            },
            parameters={
                "line_color": {"type": "color", "default": "#1f77b4"},
                "line_width": {
                    "type": "number",
                    "default": 2.0,
                    "min": 0.5,
                    "max": 10.0,
                },
                "show_markers": {"type": "boolean", "default": True},
            },
            tags=["basic", "line", "time-series"],
            created_by="system",
        )

        # Scatter plot template
        scatter_template = ChartTemplate(
            name="Correlation Scatter Plot",
            description="Scatter plot for correlation analysis",
            category="Analysis",
            chart_type=ChartType.SCATTER,
            config_template={
                "type": "scatter",
                "style": {"marker_size": 8.0, "opacity": 0.7},
                "x_axis": {"title": "X Variable", "type": "linear", "show_grid": True},
                "y_axis": {"title": "Y Variable", "type": "linear", "show_grid": True},
            },
            data_requirements={
                "x": {"type": "array", "description": "X-axis values"},
                "y": {"type": "array", "description": "Y-axis values"},
                "color": {
                    "type": "array",
                    "optional": True,
                    "description": "Color mapping values",
                },
            },
            parameters={
                "marker_color": {"type": "color", "default": "#ff7f0e"},
                "marker_size": {
                    "type": "number",
                    "default": 8.0,
                    "min": 2.0,
                    "max": 20.0,
                },
                "show_trendline": {"type": "boolean", "default": False},
            },
            tags=["scatter", "correlation", "analysis"],
            created_by="system",
        )

        # Bar chart template
        bar_template = ChartTemplate(
            name="Category Bar Chart",
            description="Bar chart for categorical data comparison",
            category="Basic",
            chart_type=ChartType.BAR,
            config_template={
                "type": "bar",
                "style": {"opacity": 0.8},
                "x_axis": {"title": "Category", "type": "category", "show_grid": False},
                "y_axis": {"title": "Value", "type": "linear", "show_grid": True},
            },
            data_requirements={
                "x": {"type": "array", "description": "Category names"},
                "y": {"type": "array", "description": "Values"},
            },
            parameters={
                "bar_color": {"type": "color", "default": "#2ca02c"},
                "orientation": {
                    "type": "select",
                    "options": ["vertical", "horizontal"],
                    "default": "vertical",
                },
            },
            tags=["bar", "categorical", "comparison"],
            created_by="system",
        )

        self._templates[line_template.id] = line_template
        self._templates[scatter_template.id] = scatter_template
        self._templates[bar_template.id] = bar_template

    def get_template(self, template_id: str) -> Optional[ChartTemplate]:
        """Get template by ID.

        Args:
            template_id: Template ID

        Returns:
            Template or None if not found
        """
        return self._templates.get(template_id)

    def get_templates_by_category(self, category: str) -> list[ChartTemplate]:
        """Get templates by category.

        Args:
            category: Template category

        Returns:
            List of templates in category
        """
        return [t for t in self._templates.values() if t.category.lower() == category.lower()]

    def get_templates_by_chart_type(self, chart_type: ChartType) -> list[ChartTemplate]:
        """Get templates by chart type.

        Args:
            chart_type: Chart type

        Returns:
            List of templates for chart type
        """
        return [t for t in self._templates.values() if t.chart_type == chart_type]

    def create_template(self, template: ChartTemplate) -> str:
        """Create a new template.

        Args:
            template: Template configuration

        Returns:
            Template ID
        """
        try:
            self._templates[template.id] = template
            logger.info(f"Created template: {template.name}")
            return template.id

        except Exception as e:
            logger.error(f"Failed to create template: {str(e)}")
            raise TemplateError(f"Template creation failed: {str(e)}")

    def apply_template_to_config(
        self, template: ChartTemplate, parameters: Optional[dict[str, Any]] = None
    ) -> ChartConfig:
        """Apply template to create chart configuration.

        Args:
            template: Chart template
            parameters: Template parameters

        Returns:
            Chart configuration
        """
        try:
            # Start with template configuration
            config_dict = template.config_template.copy()

            # Apply parameters
            if parameters:
                for param_name, param_value in parameters.items():
                    if param_name in template.parameters:
                        # Apply parameter to appropriate config section
                        self._apply_parameter(
                            config_dict,
                            param_name,
                            param_value,
                            template.parameters[param_name],
                        )

            # Create chart configuration
            from .models import ChartData

            # Remove conflicting keys from config_dict
            config_dict.pop("type", None)
            config_dict.pop("title", None)
            config_dict.pop("data", None)

            config = ChartConfig(
                type=template.chart_type,
                title=template.name,
                data=ChartData(x=[1, 2, 3], y=[1, 4, 2]),  # Default data
                **config_dict,
            )

            logger.info(f"Applied template {template.name} to create configuration")
            return config

        except Exception as e:
            logger.error(f"Failed to apply template: {str(e)}")
            raise TemplateError(f"Template application failed: {str(e)}")

    def _apply_parameter(
        self,
        config_dict: dict[str, Any],
        param_name: str,
        param_value: Any,
        param_config: dict[str, Any],
    ) -> None:
        """Apply parameter value to configuration dictionary.

        Args:
            config_dict: Configuration dictionary
            param_name: Parameter name
            param_value: Parameter value
            param_config: Parameter configuration
        """
        # Map parameter names to config paths
        param_mappings = {
            "line_color": ["style", "color"],
            "line_width": ["style", "line_width"],
            "marker_color": ["style", "color"],
            "marker_size": ["style", "marker_size"],
            "bar_color": ["style", "color"],
            "show_markers": ["style", "marker_size"],  # Set to 0 if False
        }

        if param_name in param_mappings:
            path = param_mappings[param_name]

            # Navigate to the correct section
            current = config_dict
            for key in path[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]

            # Set the value
            if param_name == "show_markers":
                current[path[-1]] = 6.0 if param_value else 0.0
            else:
                current[path[-1]] = param_value

    def list_templates(self) -> list[ChartTemplate]:
        """List all available templates.

        Returns:
            List of templates
        """
        return list(self._templates.values())

    def search_templates(self, query: str) -> list[ChartTemplate]:
        """Search templates by name, description, or tags.

        Args:
            query: Search query

        Returns:
            List of matching templates
        """
        query_lower = query.lower()
        results = []

        for template in self._templates.values():
            # Search in name
            if query_lower in template.name.lower():
                results.append(template)
                continue

            # Search in description
            if template.description and query_lower in template.description.lower():
                results.append(template)
                continue

            # Search in tags
            if any(query_lower in tag.lower() for tag in template.tags):
                results.append(template)
                continue

        return results


class StyleManager:
    """Manager for chart styling and appearance."""

    def __init__(self):
        """Initialize style manager."""
        self._color_palettes = self._load_color_palettes()

    def _load_color_palettes(self) -> dict[str, list[str]]:
        """Load predefined color palettes.

        Returns:
            Dictionary of color palettes
        """
        return {
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
            "sunset": [
                "#FF4500",
                "#FF6347",
                "#FF7F50",
                "#FFA500",
                "#FFB347",
                "#FFCCCB",
                "#FFE4E1",
                "#FFF8DC",
                "#FFFACD",
                "#FFFFE0",
            ],
            "forest": [
                "#006400",
                "#228B22",
                "#32CD32",
                "#90EE90",
                "#98FB98",
                "#F0FFF0",
                "#ADFF2F",
                "#7CFC00",
                "#7FFF00",
                "#9AFF9A",
            ],
            "scientific": [
                "#2E86AB",
                "#A23B72",
                "#F18F01",
                "#C73E1D",
                "#592E83",
                "#1B998B",
                "#ED217C",
                "#F18F01",
                "#C73E1D",
                "#592E83",
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

    def get_color_palette(self, name: str) -> list[str]:
        """Get color palette by name.

        Args:
            name: Palette name

        Returns:
            List of colors
        """
        return self._color_palettes.get(name, self._color_palettes["default"])

    def create_gradient_palette(
        self, start_color: str, end_color: str, steps: int = 10
    ) -> list[str]:
        """Create gradient color palette.

        Args:
            start_color: Starting color (hex)
            end_color: Ending color (hex)
            steps: Number of gradient steps

        Returns:
            List of gradient colors
        """
        try:
            import matplotlib.colors as mcolors

            # Create colormap
            cmap = mcolors.LinearSegmentedColormap.from_list("gradient", [start_color, end_color])

            # Generate colors
            colors = [mcolors.rgb2hex(cmap(i / (steps - 1))) for i in range(steps)]
            return colors

        except ImportError:
            # Fallback without matplotlib
            return [start_color, end_color]

    def apply_style_preset(self, config: ChartConfig, preset: str) -> ChartConfig:
        """Apply style preset to chart configuration.

        Args:
            config: Chart configuration
            preset: Style preset name

        Returns:
            Updated chart configuration
        """
        presets = {
            "minimal": {
                "style": {"line_width": 1.0, "marker_size": 3.0, "opacity": 0.9},
                "layout": {
                    "show_legend": False,
                    "margin": {"l": 40, "r": 20, "t": 40, "b": 40},
                },
                "x_axis": {"show_grid": False},
                "y_axis": {"show_grid": True, "grid_color": "#f0f0f0"},
            },
            "bold": {
                "style": {"line_width": 4.0, "marker_size": 10.0, "opacity": 0.8},
                "layout": {"font_size": 14, "show_legend": True},
            },
            "scientific": {
                "style": {"line_width": 1.5, "marker_size": 4.0, "opacity": 0.85},
                "layout": {
                    "font_family": "Times New Roman",
                    "font_size": 12,
                    "background_color": "#ffffff",
                    "paper_color": "#ffffff",
                },
                "x_axis": {"show_grid": True, "grid_color": "#cccccc"},
                "y_axis": {"show_grid": True, "grid_color": "#cccccc"},
            },
        }

        if preset in presets:
            preset_config = presets[preset]

            # Apply style updates
            for section, updates in preset_config.items():
                if hasattr(config, section):
                    section_obj = getattr(config, section)
                    for key, value in updates.items():
                        if hasattr(section_obj, key):
                            setattr(section_obj, key, value)

        return config

    def list_color_palettes(self) -> list[str]:
        """List available color palettes.

        Returns:
            List of palette names
        """
        return list(self._color_palettes.keys())

    def list_style_presets(self) -> list[str]:
        """List available style presets.

        Returns:
            List of preset names
        """
        return ["minimal", "bold", "scientific"]
