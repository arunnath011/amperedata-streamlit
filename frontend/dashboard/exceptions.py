"""Dashboard-specific exception classes.

This module defines custom exception classes for various dashboard-related errors,
providing detailed error information for debugging and user feedback.
"""

from typing import Optional, Any


class DashboardError(Exception):
    """Base dashboard exception."""

    def __init__(
        self,
        message: str,
        dashboard_id: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.dashboard_id = dashboard_id
        self.error_code = error_code
        self.details = details or {}


class LayoutError(DashboardError):
    """Layout-related errors."""

    def __init__(
        self,
        message: str,
        layout_type: Optional[str] = None,
        position: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.layout_type = layout_type
        self.position = position


class WidgetError(DashboardError):
    """Widget-related errors."""

    def __init__(
        self,
        message: str,
        widget_id: Optional[str] = None,
        widget_type: Optional[str] = None,
        widget_config: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.widget_id = widget_id
        self.widget_type = widget_type
        self.widget_config = widget_config


class PermissionError(DashboardError):
    """Permission and access control errors."""

    def __init__(
        self,
        message: str,
        user_id: Optional[str] = None,
        required_permission: Optional[str] = None,
        current_permission: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.user_id = user_id
        self.required_permission = required_permission
        self.current_permission = current_permission


class VersionError(DashboardError):
    """Version control errors."""

    def __init__(
        self,
        message: str,
        version_number: Optional[int] = None,
        conflict_details: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.version_number = version_number
        self.conflict_details = conflict_details


class EmbedError(DashboardError):
    """Embedding and sharing errors."""

    def __init__(
        self,
        message: str,
        embed_id: Optional[str] = None,
        embed_type: Optional[str] = None,
        domain: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.embed_id = embed_id
        self.embed_type = embed_type
        self.domain = domain


class DataSourceError(DashboardError):
    """Data source connection and query errors."""

    def __init__(
        self,
        message: str,
        source_type: Optional[str] = None,
        connection_string: Optional[str] = None,
        query: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.source_type = source_type
        self.connection_string = connection_string
        self.query = query


class RenderingError(DashboardError):
    """Dashboard rendering errors."""

    def __init__(
        self,
        message: str,
        rendering_stage: Optional[str] = None,
        failed_widgets: Optional[list[str]] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.rendering_stage = rendering_stage
        self.failed_widgets = failed_widgets or []


class TemplateError(DashboardError):
    """Template-related errors."""

    def __init__(
        self,
        message: str,
        template_id: Optional[str] = None,
        template_name: Optional[str] = None,
        parameters: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.template_id = template_id
        self.template_name = template_name
        self.parameters = parameters


class StorageError(DashboardError):
    """Storage and persistence errors."""

    def __init__(
        self,
        message: str,
        storage_type: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.storage_type = storage_type
        self.operation = operation


class ValidationError(DashboardError):
    """Configuration validation errors."""

    def __init__(
        self,
        message: str,
        validation_field: Optional[str] = None,
        validation_rule: Optional[str] = None,
        provided_value: Optional[Any] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.validation_field = validation_field
        self.validation_rule = validation_rule
        self.provided_value = provided_value


class ScheduleError(DashboardError):
    """Scheduling and automation errors."""

    def __init__(
        self,
        message: str,
        schedule_id: Optional[str] = None,
        cron_expression: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.schedule_id = schedule_id
        self.cron_expression = cron_expression


class ExportError(DashboardError):
    """Export and snapshot errors."""

    def __init__(
        self,
        message: str,
        export_format: Optional[str] = None,
        file_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.export_format = export_format
        self.file_path = file_path
