"""Logging middleware for FastAPI applications."""

import time
import uuid
from collections.abc import Callable
from typing import Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from .config import get_logger


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log HTTP requests and responses."""

    def __init__(self, app: Any, logger_name: str = "http") -> None:
        super().__init__(app)
        self.logger = get_logger(logger_name)

    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Response:
        """Process HTTP request and log details."""
        # Generate request ID
        request_id = str(uuid.uuid4())

        # Add request ID to request state
        request.state.request_id = request_id

        # Start timing
        start_time = time.time()

        # Log request
        self.logger.info(
            "HTTP request started",
            request_id=request_id,
            method=request.method,
            url=str(request.url),
            user_agent=request.headers.get("user-agent"),
            client_ip=request.client.host if request.client else None,
        )

        # Process request
        try:
            response = await call_next(request)

            # Calculate duration
            duration = time.time() - start_time

            # Log successful response
            self.logger.info(
                "HTTP request completed",
                request_id=request_id,
                method=request.method,
                url=str(request.url),
                status_code=response.status_code,
                duration_seconds=duration,
            )

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id

            return response

        except Exception as exc:
            # Calculate duration
            duration = time.time() - start_time

            # Log error
            self.logger.error(
                "HTTP request failed",
                request_id=request_id,
                method=request.method,
                url=str(request.url),
                duration_seconds=duration,
                error=str(exc),
                exc_info=True,
            )

            raise
