"""Dashboard templates for different roles and use cases.

This module provides pre-built dashboard templates optimized for different
user roles and battery analysis scenarios, enabling quick dashboard creation.
"""

import logging
from datetime import datetime
from typing import Optional, Any

from frontend.visualization.models import ChartConfig, ChartData, ChartType

from .exceptions import TemplateError
from .models import (
    AlertConfig,
    ChartWidgetConfig,
    DashboardConfig,
    DashboardLayout,
    DashboardRole,
    DashboardTemplate,
    DashboardWidget,
    GaugeConfig,
    KPIConfig,
    LayoutType,
    MetricConfig,
    ProgressConfig,
    TableConfig,
    WidgetPosition,
    WidgetType,
)

logger = logging.getLogger(__name__)


class DashboardTemplateManager:
    """Manages dashboard templates and template operations."""

    def __init__(self):
        """Initialize template manager."""
        self._templates: dict[str, DashboardTemplate] = {}
        self._categories: dict[str, list[str]] = {}
        self._role_templates: dict[DashboardRole, list[str]] = {}

    def register_template(self, template: DashboardTemplate) -> None:
        """Register a dashboard template.

        Args:
            template: Template to register
        """
        self._templates[template.id] = template

        # Update category index
        if template.category not in self._categories:
            self._categories[template.category] = []
        self._categories[template.category].append(template.id)

        # Update role index
        if template.role:
            if template.role not in self._role_templates:
                self._role_templates[template.role] = []
            self._role_templates[template.role].append(template.id)

        logger.info(f"Registered template: {template.name}")

    def get_template(self, template_id: str) -> Optional[DashboardTemplate]:
        """Get template by ID.

        Args:
            template_id: Template ID

        Returns:
            Template or None if not found
        """
        return self._templates.get(template_id)

    def get_template_by_name(self, name: str) -> Optional[DashboardTemplate]:
        """Get template by name.

        Args:
            name: Template name

        Returns:
            Template or None if not found
        """
        for template in self._templates.values():
            if template.name == name:
                return template
        return None

    def list_templates(
        self,
        category: Optional[str] = None,
        role: Optional[DashboardRole] = None,
        tags: Optional[list[str]] = None,
    ) -> list[DashboardTemplate]:
        """List templates with optional filtering.

        Args:
            category: Filter by category
            role: Filter by role
            tags: Filter by tags

        Returns:
            List of matching templates
        """
        templates = list(self._templates.values())

        if category:
            templates = [t for t in templates if t.category == category]

        if role:
            templates = [t for t in templates if t.role == role]

        if tags:
            templates = [t for t in templates if any(tag in t.tags for tag in tags)]

        return sorted(templates, key=lambda t: t.name)

    def get_categories(self) -> list[str]:
        """Get all template categories.

        Returns:
            List of categories
        """
        return sorted(self._categories.keys())

    def get_templates_by_role(self, role: DashboardRole) -> list[DashboardTemplate]:
        """Get templates for specific role.

        Args:
            role: Dashboard role

        Returns:
            List of templates for role
        """
        template_ids = self._role_templates.get(role, [])
        return [self._templates[tid] for tid in template_ids if tid in self._templates]

    def create_dashboard_from_template(
        self,
        template_id: str,
        name: str,
        created_by: str,
        parameters: Optional[dict[str, Any]] = None,
    ) -> DashboardConfig:
        """Create dashboard from template.

        Args:
            template_id: Template ID
            name: Dashboard name
            created_by: User creating dashboard
            parameters: Template parameters

        Returns:
            Dashboard configuration

        Raises:
            TemplateError: If template not found or creation fails
        """
        template = self.get_template(template_id)
        if not template:
            raise TemplateError(f"Template {template_id} not found")

        try:
            # Copy template configuration
            config = template.config_template.copy(deep=True)

            # Update basic properties
            config.name = name
            config.created_by = created_by
            config.created_at = datetime.now()
            config.updated_at = datetime.now()

            # Apply parameters if provided
            if parameters:
                config = self._apply_template_parameters(config, template, parameters)

            # Increment template usage count
            template.usage_count += 1

            logger.info(f"Created dashboard from template {template.name}")
            return config

        except Exception as e:
            logger.error(f"Failed to create dashboard from template: {str(e)}")
            raise TemplateError(f"Dashboard creation failed: {str(e)}")

    def _apply_template_parameters(
        self,
        config: DashboardConfig,
        template: DashboardTemplate,
        parameters: dict[str, Any],
    ) -> DashboardConfig:
        """Apply template parameters to dashboard configuration.

        Args:
            config: Dashboard configuration
            template: Template
            parameters: Parameters to apply

        Returns:
            Updated configuration
        """
        # Apply basic parameters
        if "description" in parameters:
            config.description = parameters["description"]

        if "theme" in parameters:
            config.theme = parameters["theme"]

        # Apply widget-specific parameters
        for widget in config.widgets:
            widget_params = parameters.get(f"widget_{widget.id}", {})

            if "title" in widget_params:
                widget.config.title = widget_params["title"]

            if "data_source" in widget_params:
                widget.config.data_source = widget_params["data_source"]

        return config


class BatteryAnalysisTemplates:
    """Pre-built templates for battery data analysis."""

    @staticmethod
    def get_researcher_overview_template() -> DashboardTemplate:
        """Create researcher overview dashboard template."""

        # Create widgets
        widgets = []

        # KPI Cards Row
        widgets.extend(
            [
                DashboardWidget(
                    type=WidgetType.KPI,
                    position=WidgetPosition(x=0, y=0, width=3, height=2),
                    config=KPIConfig(
                        title="Active Experiments",
                        value=12,
                        unit="experiments",
                        trend=8.5,
                        show_trend=True,
                        color_scheme="blue",
                    ),
                ),
                DashboardWidget(
                    type=WidgetType.KPI,
                    position=WidgetPosition(x=3, y=0, width=3, height=2),
                    config=KPIConfig(
                        title="Avg Capacity",
                        value=2.45,
                        unit="Ah",
                        trend=-2.1,
                        show_trend=True,
                        color_scheme="green",
                    ),
                ),
                DashboardWidget(
                    type=WidgetType.KPI,
                    position=WidgetPosition(x=6, y=0, width=3, height=2),
                    config=KPIConfig(
                        title="Cycle Count",
                        value=1250,
                        unit="cycles",
                        trend=15.2,
                        show_trend=True,
                        color_scheme="orange",
                    ),
                ),
                DashboardWidget(
                    type=WidgetType.KPI,
                    position=WidgetPosition(x=9, y=0, width=3, height=2),
                    config=KPIConfig(
                        title="Success Rate",
                        value=94.2,
                        unit="%",
                        trend=1.8,
                        show_trend=True,
                        color_scheme="purple",
                    ),
                ),
            ]
        )

        # Charts Row
        widgets.extend(
            [
                DashboardWidget(
                    type=WidgetType.CHART,
                    position=WidgetPosition(x=0, y=2, width=6, height=4),
                    config=ChartWidgetConfig(
                        title="Capacity Fade Over Time",
                        chart_config=ChartConfig(
                            type=ChartType.LINE,
                            title="Capacity Fade Analysis",
                            data=ChartData(x=[1, 2, 3, 4, 5], y=[2.5, 2.4, 2.3, 2.2, 2.1]),
                        ),
                        export_enabled=True,
                        fullscreen_enabled=True,
                    ),
                ),
                DashboardWidget(
                    type=WidgetType.CHART,
                    position=WidgetPosition(x=6, y=2, width=6, height=4),
                    config=ChartWidgetConfig(
                        title="Voltage Profiles",
                        chart_config=ChartConfig(
                            type=ChartType.SCATTER,
                            title="Voltage vs Capacity",
                            data=ChartData(x=[0, 0.5, 1.0, 1.5, 2.0], y=[3.2, 3.4, 3.6, 3.8, 4.0]),
                        ),
                        export_enabled=True,
                        fullscreen_enabled=True,
                    ),
                ),
            ]
        )

        # Metrics and Gauges Row
        widgets.extend(
            [
                DashboardWidget(
                    type=WidgetType.GAUGE,
                    position=WidgetPosition(x=0, y=6, width=4, height=3),
                    config=GaugeConfig(
                        title="Battery Health",
                        value=87.5,
                        min_value=0,
                        max_value=100,
                        unit="%",
                        show_value=True,
                    ),
                ),
                DashboardWidget(
                    type=WidgetType.PROGRESS,
                    position=WidgetPosition(x=4, y=6, width=4, height=3),
                    config=ProgressConfig(
                        title="Experiment Progress",
                        value=750,
                        max_value=1000,
                        label="Cycles Completed",
                        show_percentage=True,
                        color="#28a745",
                    ),
                ),
                DashboardWidget(
                    type=WidgetType.METRIC,
                    position=WidgetPosition(x=8, y=6, width=4, height=3),
                    config=MetricConfig(
                        title="Temperature Monitor",
                        value=25.4,
                        label="Current Temp",
                        unit="°C",
                        color="#17a2b8",
                        precision=1,
                    ),
                ),
            ]
        )

        # Recent Experiments Table
        widgets.append(
            DashboardWidget(
                type=WidgetType.TABLE,
                position=WidgetPosition(x=0, y=9, width=12, height=4),
                config=TableConfig(
                    title="Recent Experiments",
                    columns=[
                        {"field": "id", "header": "ID", "sortable": True},
                        {"field": "name", "header": "Name", "sortable": True},
                        {"field": "status", "header": "Status", "sortable": True},
                        {"field": "cycles", "header": "Cycles", "sortable": True},
                        {
                            "field": "capacity",
                            "header": "Capacity (Ah)",
                            "sortable": True,
                        },
                        {"field": "started", "header": "Started", "sortable": True},
                    ],
                    sortable=True,
                    filterable=True,
                    searchable=True,
                    pagination=True,
                    page_size=10,
                ),
            )
        )

        # Create layout
        layout = DashboardLayout(
            type=LayoutType.GRID,
            grid_size=(12, 15),
            gap=10,
            padding=20,
            responsive=True,
        )

        # Create dashboard config
        config = DashboardConfig(
            name="Researcher Overview",
            description="Comprehensive overview dashboard for battery researchers",
            layout=layout,
            widgets=widgets,
            theme="scientific",
            auto_refresh=True,
            created_by="system",
        )

        return DashboardTemplate(
            name="Researcher Overview",
            description="Comprehensive dashboard for battery researchers with KPIs, charts, and experiment tracking",
            category="Research",
            role=DashboardRole.RESEARCHER,
            config_template=config,
            tags=["research", "overview", "experiments", "kpi"],
            is_system=True,
            created_by="system",
        )

    @staticmethod
    def get_manager_executive_template() -> DashboardTemplate:
        """Create manager/executive dashboard template."""

        widgets = []

        # Executive KPIs
        widgets.extend(
            [
                DashboardWidget(
                    type=WidgetType.KPI,
                    position=WidgetPosition(x=0, y=0, width=4, height=2),
                    config=KPIConfig(
                        title="Project Progress",
                        value=78,
                        unit="%",
                        trend=5.2,
                        target=85,
                        show_trend=True,
                        show_target=True,
                        color_scheme="blue",
                    ),
                ),
                DashboardWidget(
                    type=WidgetType.KPI,
                    position=WidgetPosition(x=4, y=0, width=4, height=2),
                    config=KPIConfig(
                        title="Budget Utilization",
                        value=65.4,
                        unit="%",
                        trend=-1.8,
                        target=70,
                        show_trend=True,
                        show_target=True,
                        color_scheme="green",
                    ),
                ),
                DashboardWidget(
                    type=WidgetType.KPI,
                    position=WidgetPosition(x=8, y=0, width=4, height=2),
                    config=KPIConfig(
                        title="Team Efficiency",
                        value=92.1,
                        unit="%",
                        trend=3.7,
                        target=90,
                        show_trend=True,
                        show_target=True,
                        color_scheme="purple",
                    ),
                ),
            ]
        )

        # High-level Charts
        widgets.extend(
            [
                DashboardWidget(
                    type=WidgetType.CHART,
                    position=WidgetPosition(x=0, y=2, width=8, height=4),
                    config=ChartWidgetConfig(
                        title="Project Timeline",
                        chart_config=ChartConfig(
                            type=ChartType.BAR,
                            title="Project Milestones",
                            data=ChartData(x=["Q1", "Q2", "Q3", "Q4"], y=[25, 45, 70, 85]),
                        ),
                    ),
                ),
                DashboardWidget(
                    type=WidgetType.GAUGE,
                    position=WidgetPosition(x=8, y=2, width=4, height=4),
                    config=GaugeConfig(
                        title="Overall Health Score",
                        value=84.5,
                        min_value=0,
                        max_value=100,
                        unit="%",
                        show_value=True,
                    ),
                ),
            ]
        )

        # Alerts and Status
        widgets.extend(
            [
                DashboardWidget(
                    type=WidgetType.ALERT,
                    position=WidgetPosition(x=0, y=6, width=6, height=2),
                    config=AlertConfig(
                        title="System Alerts",
                        message="2 experiments require attention",
                        alert_type="warning",
                        dismissible=True,
                    ),
                ),
                DashboardWidget(
                    type=WidgetType.PROGRESS,
                    position=WidgetPosition(x=6, y=6, width=6, height=2),
                    config=ProgressConfig(
                        title="Quarterly Goals",
                        value=68,
                        max_value=100,
                        label="Goal Achievement",
                        show_percentage=True,
                        color="#ffc107",
                    ),
                ),
            ]
        )

        layout = DashboardLayout(
            type=LayoutType.GRID,
            grid_size=(12, 10),
            gap=15,
            padding=25,
            responsive=True,
        )

        config = DashboardConfig(
            name="Executive Overview",
            description="High-level dashboard for managers and executives",
            layout=layout,
            widgets=widgets,
            theme="light",
            auto_refresh=True,
            created_by="system",
        )

        return DashboardTemplate(
            name="Executive Overview",
            description="High-level dashboard for managers with KPIs, project status, and alerts",
            category="Management",
            role=DashboardRole.MANAGER,
            config_template=config,
            tags=["executive", "management", "kpi", "overview"],
            is_system=True,
            created_by="system",
        )

    @staticmethod
    def get_analyst_detailed_template() -> DashboardTemplate:
        """Create analyst detailed dashboard template."""

        widgets = []

        # Detailed Analysis Charts
        widgets.extend(
            [
                DashboardWidget(
                    type=WidgetType.CHART,
                    position=WidgetPosition(x=0, y=0, width=6, height=5),
                    config=ChartWidgetConfig(
                        title="Capacity vs Cycle Analysis",
                        chart_config=ChartConfig(
                            type=ChartType.SCATTER,
                            title="Detailed Capacity Analysis",
                            data=ChartData(
                                x=list(range(100)),
                                y=[2.5 - i * 0.001 for i in range(100)],
                            ),
                        ),
                    ),
                ),
                DashboardWidget(
                    type=WidgetType.CHART,
                    position=WidgetPosition(x=6, y=0, width=6, height=5),
                    config=ChartWidgetConfig(
                        title="dQ/dV Analysis",
                        chart_config=ChartConfig(
                            type=ChartType.LINE,
                            title="Differential Capacity",
                            data=ChartData(
                                x=[3.0, 3.2, 3.4, 3.6, 3.8, 4.0],
                                y=[0.1, 0.8, 1.2, 0.9, 0.3, 0.1],
                            ),
                        ),
                    ),
                ),
            ]
        )

        # Statistical Metrics
        widgets.extend(
            [
                DashboardWidget(
                    type=WidgetType.METRIC,
                    position=WidgetPosition(x=0, y=5, width=3, height=2),
                    config=MetricConfig(
                        title="Statistical Analysis",
                        value=0.95,
                        label="R² Correlation",
                        precision=3,
                        color="#28a745",
                    ),
                ),
                DashboardWidget(
                    type=WidgetType.METRIC,
                    position=WidgetPosition(x=3, y=5, width=3, height=2),
                    config=MetricConfig(
                        title="Fade Rate",
                        value=0.025,
                        label="% per cycle",
                        unit="%",
                        precision=3,
                        color="#dc3545",
                    ),
                ),
                DashboardWidget(
                    type=WidgetType.METRIC,
                    position=WidgetPosition(x=6, y=5, width=3, height=2),
                    config=MetricConfig(
                        title="Efficiency",
                        value=98.7,
                        label="Coulombic Eff.",
                        unit="%",
                        precision=1,
                        color="#17a2b8",
                    ),
                ),
                DashboardWidget(
                    type=WidgetType.METRIC,
                    position=WidgetPosition(x=9, y=5, width=3, height=2),
                    config=MetricConfig(
                        title="Impedance",
                        value=45.2,
                        label="Internal R",
                        unit="mΩ",
                        precision=1,
                        color="#6f42c1",
                    ),
                ),
            ]
        )

        # Detailed Data Table
        widgets.append(
            DashboardWidget(
                type=WidgetType.TABLE,
                position=WidgetPosition(x=0, y=7, width=12, height=5),
                config=TableConfig(
                    title="Detailed Measurements",
                    columns=[
                        {"field": "cycle", "header": "Cycle", "sortable": True},
                        {"field": "voltage", "header": "Voltage (V)", "sortable": True},
                        {"field": "current", "header": "Current (A)", "sortable": True},
                        {
                            "field": "capacity",
                            "header": "Capacity (Ah)",
                            "sortable": True,
                        },
                        {"field": "energy", "header": "Energy (Wh)", "sortable": True},
                        {
                            "field": "temperature",
                            "header": "Temp (°C)",
                            "sortable": True,
                        },
                        {"field": "timestamp", "header": "Timestamp", "sortable": True},
                    ],
                    sortable=True,
                    filterable=True,
                    searchable=True,
                    pagination=True,
                    page_size=25,
                    export_enabled=True,
                ),
            )
        )

        layout = DashboardLayout(
            type=LayoutType.GRID, grid_size=(12, 12), gap=8, padding=15, responsive=True
        )

        config = DashboardConfig(
            name="Detailed Analysis",
            description="Comprehensive analytical dashboard with detailed charts and data",
            layout=layout,
            widgets=widgets,
            theme="scientific",
            auto_refresh=False,
            created_by="system",
        )

        return DashboardTemplate(
            name="Detailed Analysis",
            description="Comprehensive dashboard for analysts with detailed charts, statistics, and data tables",
            category="Analysis",
            role=DashboardRole.ANALYST,
            config_template=config,
            tags=["analysis", "detailed", "statistics", "data"],
            is_system=True,
            created_by="system",
        )


class RoleBasedTemplates:
    """Role-based template collections."""

    def __init__(self):
        """Initialize role-based templates."""
        self.template_manager = DashboardTemplateManager()
        self._register_all_templates()

    def _register_all_templates(self) -> None:
        """Register all built-in templates."""
        battery_templates = BatteryAnalysisTemplates()

        # Register battery analysis templates
        templates = [
            battery_templates.get_researcher_overview_template(),
            battery_templates.get_manager_executive_template(),
            battery_templates.get_analyst_detailed_template(),
        ]

        for template in templates:
            self.template_manager.register_template(template)

        logger.info(f"Registered {len(templates)} built-in templates")

    def get_templates_for_role(self, role: DashboardRole) -> list[DashboardTemplate]:
        """Get templates suitable for specific role.

        Args:
            role: Dashboard role

        Returns:
            List of suitable templates
        """
        return self.template_manager.get_templates_by_role(role)

    def get_recommended_template(self, role: DashboardRole) -> Optional[DashboardTemplate]:
        """Get recommended template for role.

        Args:
            role: Dashboard role

        Returns:
            Recommended template or None
        """
        templates = self.get_templates_for_role(role)

        if not templates:
            return None

        # Return most used template for role
        return max(templates, key=lambda t: t.usage_count)

    def create_custom_template(
        self,
        name: str,
        description: str,
        category: str,
        role: DashboardRole,
        config: DashboardConfig,
        created_by: str,
        tags: Optional[list[str]] = None,
    ) -> DashboardTemplate:
        """Create custom template.

        Args:
            name: Template name
            description: Template description
            category: Template category
            role: Target role
            config: Dashboard configuration
            created_by: Template creator
            tags: Template tags

        Returns:
            Created template
        """
        template = DashboardTemplate(
            name=name,
            description=description,
            category=category,
            role=role,
            config_template=config,
            tags=tags or [],
            is_system=False,
            created_by=created_by,
        )

        self.template_manager.register_template(template)

        logger.info(f"Created custom template: {name}")
        return template
