"""Structured logging configuration using structlog."""

import logging
import sys
from typing import Any, Optional

import structlog
from structlog.types import FilteringBoundLogger

from shared.config.settings import settings


def configure_logging(
    log_level: Optional[str] = None,
    json_logs: bool = False,
    show_locals: bool = False,
) -> None:
    """Configure structured logging with structlog.

    Args:
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_logs: Whether to output logs in JSON format
        show_locals: Whether to show local variables in tracebacks
    """
    log_level = log_level or settings.log_level

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level),
    )

    # Determine if we're in development mode
    is_development = settings.environment.lower() in ("development", "dev", "local")

    # Configure processors based on environment
    processors = [
        # Add log level
        structlog.stdlib.add_log_level,
        # Add logger name
        structlog.stdlib.add_logger_name,
        # Add timestamp
        structlog.processors.TimeStamper(fmt="ISO"),
        # Add stack info for exceptions
        structlog.processors.StackInfoRenderer(),
        # Format exceptions
        structlog.processors.format_exc_info,
        # Add call site information (file, line, function)
        structlog.processors.CallsiteParameterAdder(
            parameters=[
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.LINENO,
                structlog.processors.CallsiteParameter.FUNC_NAME,
            ]
        ),
    ]

    # Add environment-specific processors
    if json_logs or not is_development:
        # Production: JSON output
        processors.append(structlog.processors.JSONRenderer())
    else:
        # Development: Pretty console output
        processors.extend(
            [
                structlog.dev.ConsoleRenderer(colors=True),
            ]
        )

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        context_class=dict,
        cache_logger_on_first_use=True,
    )

    # Configure specific loggers
    _configure_third_party_loggers(log_level)

    # Log configuration
    logger = get_logger(__name__)
    logger.info(
        "Logging configured",
        log_level=log_level,
        json_logs=json_logs,
        environment=settings.environment,
        show_locals=show_locals,
    )


def _configure_third_party_loggers(log_level: str) -> None:
    """Configure third-party library loggers."""
    # Reduce noise from third-party libraries
    third_party_loggers = {
        "uvicorn": "INFO",
        "uvicorn.error": "INFO",
        "uvicorn.access": "WARNING",
        "fastapi": "INFO",
        "sqlalchemy.engine": "WARNING",
        "sqlalchemy.pool": "WARNING",
        "alembic": "INFO",
        "celery": "INFO",
        "redis": "WARNING",
        "httpx": "WARNING",
        "asyncio": "WARNING",
    }

    for logger_name, level in third_party_loggers.items():
        logging.getLogger(logger_name).setLevel(getattr(logging, level))

    # Set root logger level
    logging.getLogger().setLevel(getattr(logging, log_level))


def get_logger(name: str) -> FilteringBoundLogger:
    """Get a configured logger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


def add_request_context(
    request_id: Optional[str] = None,
    user_id: Optional[str] = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Add request context to log messages.

    Args:
        request_id: Unique request identifier
        user_id: User identifier
        **kwargs: Additional context

    Returns:
        Context dictionary
    """
    context = {}

    if request_id:
        context["request_id"] = request_id
    if user_id:
        context["user_id"] = user_id

    context.update(kwargs)
    return context


def log_function_call(
    func_name: str,
    args: tuple = (),
    kwargs: Optional[dict[str, Any]] = None,
    logger: Optional[FilteringBoundLogger] = None,
) -> None:
    """Log function call details.

    Args:
        func_name: Name of the function being called
        args: Function positional arguments
        kwargs: Function keyword arguments
        logger: Logger instance (will create one if not provided)
    """
    if logger is None:
        logger = get_logger("function_calls")

    kwargs = kwargs or {}

    logger.debug(
        "Function called",
        function=func_name,
        args=args,
        kwargs=kwargs,
    )


def log_performance(
    operation: str,
    duration: float,
    success: bool = True,
    **context: Any,
) -> None:
    """Log performance metrics.

    Args:
        operation: Name of the operation
        duration: Duration in seconds
        success: Whether the operation was successful
        **context: Additional context
    """
    logger = get_logger("performance")

    logger.info(
        "Performance metric",
        operation=operation,
        duration_seconds=duration,
        success=success,
        **context,
    )
