"""Visualization exceptions for error handling.

This module defines custom exceptions for visualization operations,
chart rendering, data formatting, and export failures.
"""

from typing import Any, Optional


class VisualizationError(Exception):
    """Base visualization exception."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class ChartRenderingError(VisualizationError):
    """Exception raised when chart rendering fails."""

    def __init__(
        self,
        message: str,
        chart_id: Optional[str] = None,
        chart_type: Optional[str] = None,
        rendering_stage: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, details)
        self.chart_id = chart_id
        self.chart_type = chart_type
        self.rendering_stage = rendering_stage


class DataFormatError(VisualizationError):
    """Exception raised when data format is invalid or incompatible."""

    def __init__(
        self,
        message: str,
        data_field: Optional[str] = None,
        expected_format: Optional[str] = None,
        actual_format: Optional[str] = None,
        data_sample: Optional[Any] = None,
        error_code: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, details)
        self.data_field = data_field
        self.expected_format = expected_format
        self.actual_format = actual_format
        self.data_sample = data_sample


class ConfigurationError(VisualizationError):
    """Exception raised when chart configuration is invalid."""

    def __init__(
        self,
        message: str,
        config_field: Optional[str] = None,
        config_value: Optional[Any] = None,
        valid_options: Optional[list[str]] = None,
        error_code: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, details)
        self.config_field = config_field
        self.config_value = config_value
        self.valid_options = valid_options or []


class ExportError(VisualizationError):
    """Exception raised when chart export fails."""

    def __init__(
        self,
        message: str,
        export_format: Optional[str] = None,
        file_path: Optional[str] = None,
        chart_id: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, details)
        self.export_format = export_format
        self.file_path = file_path
        self.chart_id = chart_id


class TemplateError(VisualizationError):
    """Exception raised when chart template processing fails."""

    def __init__(
        self,
        message: str,
        template_id: Optional[str] = None,
        template_name: Optional[str] = None,
        parameter_name: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, details)
        self.template_id = template_id
        self.template_name = template_name
        self.parameter_name = parameter_name


class ThemeError(VisualizationError):
    """Exception raised when theme processing fails."""

    def __init__(
        self,
        message: str,
        theme_id: Optional[str] = None,
        theme_name: Optional[str] = None,
        property_name: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, details)
        self.theme_id = theme_id
        self.theme_name = theme_name
        self.property_name = property_name


class DataStreamError(VisualizationError):
    """Exception raised when real-time data streaming fails."""

    def __init__(
        self,
        message: str,
        stream_id: Optional[str] = None,
        source_type: Optional[str] = None,
        connection_error: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, details)
        self.stream_id = stream_id
        self.source_type = source_type
        self.connection_error = connection_error


class InteractionError(VisualizationError):
    """Exception raised when chart interaction fails."""

    def __init__(
        self,
        message: str,
        interaction_type: Optional[str] = None,
        chart_id: Optional[str] = None,
        event_data: Optional[dict[str, Any]] = None,
        error_code: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, details)
        self.interaction_type = interaction_type
        self.chart_id = chart_id
        self.event_data = event_data


class DashboardError(VisualizationError):
    """Exception raised when dashboard operations fail."""

    def __init__(
        self,
        message: str,
        dashboard_id: Optional[str] = None,
        operation: Optional[str] = None,
        affected_charts: Optional[list[str]] = None,
        error_code: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, details)
        self.dashboard_id = dashboard_id
        self.operation = operation
        self.affected_charts = affected_charts or []


class ValidationError(VisualizationError):
    """Exception raised when chart validation fails."""

    def __init__(
        self,
        message: str,
        validation_type: Optional[str] = None,
        failed_checks: Optional[list[str]] = None,
        chart_config: Optional[dict[str, Any]] = None,
        error_code: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, details)
        self.validation_type = validation_type
        self.failed_checks = failed_checks or []
        self.chart_config = chart_config


class PerformanceError(VisualizationError):
    """Exception raised when performance limits are exceeded."""

    def __init__(
        self,
        message: str,
        metric_name: Optional[str] = None,
        current_value: Optional[float] = None,
        limit_value: Optional[float] = None,
        suggestion: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, details)
        self.metric_name = metric_name
        self.current_value = current_value
        self.limit_value = limit_value
        self.suggestion = suggestion


class AccessibilityError(VisualizationError):
    """Exception raised when accessibility requirements are not met."""

    def __init__(
        self,
        message: str,
        accessibility_rule: Optional[str] = None,
        severity: Optional[str] = None,
        element: Optional[str] = None,
        recommendation: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, details)
        self.accessibility_rule = accessibility_rule
        self.severity = severity
        self.element = element
        self.recommendation = recommendation


class ComponentError(VisualizationError):
    """Exception raised when chart component fails."""

    def __init__(
        self,
        message: str,
        component_name: Optional[str] = None,
        component_version: Optional[str] = None,
        initialization_error: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, details)
        self.component_name = component_name
        self.component_version = component_version
        self.initialization_error = initialization_error


class ResourceError(VisualizationError):
    """Exception raised when resource loading fails."""

    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        resource_path: Optional[str] = None,
        resource_url: Optional[str] = None,
        http_status: Optional[int] = None,
        error_code: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, details)
        self.resource_type = resource_type
        self.resource_path = resource_path
        self.resource_url = resource_url
        self.http_status = http_status


class SecurityError(VisualizationError):
    """Exception raised when security validation fails."""

    def __init__(
        self,
        message: str,
        security_rule: Optional[str] = None,
        violation_type: Optional[str] = None,
        user_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, details)
        self.security_rule = security_rule
        self.violation_type = violation_type
        self.user_id = user_id
        self.resource_id = resource_id


class CompatibilityError(VisualizationError):
    """Exception raised when compatibility issues are detected."""

    def __init__(
        self,
        message: str,
        browser_name: Optional[str] = None,
        browser_version: Optional[str] = None,
        feature_name: Optional[str] = None,
        minimum_version: Optional[str] = None,
        workaround: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, details)
        self.browser_name = browser_name
        self.browser_version = browser_version
        self.feature_name = feature_name
        self.minimum_version = minimum_version
        self.workaround = workaround
