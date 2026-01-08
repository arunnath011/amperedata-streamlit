"""ETL Pipeline Exceptions.

Custom exception hierarchy for ETL pipeline operations with detailed
error information and recovery suggestions.
"""

from typing import Any, Optional, Dict, List


class ETLError(Exception):
    """Base exception for ETL pipeline errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        recoverable: bool = True,
    ):
        """Initialize ETL error.

        Args:
            message: Human-readable error message
            error_code: Machine-readable error code
            details: Additional error details
            recoverable: Whether the error is recoverable with retry
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.recoverable = recoverable

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "recoverable": self.recoverable,
        }


class IngestionError(ETLError):
    """Exception raised during file ingestion phase."""

    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        parser_type: Optional[str] = None,
        **kwargs,
    ):
        """Initialize ingestion error.

        Args:
            message: Error message
            file_path: Path to the file that caused the error
            parser_type: Type of parser that failed
            **kwargs: Additional arguments for ETLError
        """
        details = kwargs.get("details", {})
        if file_path:
            details["file_path"] = file_path
        if parser_type:
            details["parser_type"] = parser_type

        kwargs["details"] = details
        super().__init__(message, **kwargs)


class ValidationError(ETLError):
    """Exception raised during data validation phase."""

    def __init__(
        self,
        message: str,
        validation_errors: Optional[List[str]] = None,
        field_errors: Optional[Dict[str, List[str]]] = None,
        **kwargs,
    ):
        """Initialize validation error.

        Args:
            message: Error message
            validation_errors: List of validation error messages
            field_errors: Field-specific validation errors
            **kwargs: Additional arguments for ETLError
        """
        details = kwargs.get("details", {})
        if validation_errors:
            details["validation_errors"] = validation_errors
        if field_errors:
            details["field_errors"] = field_errors

        kwargs["details"] = details
        super().__init__(message, **kwargs)


class TransformationError(ETLError):
    """Exception raised during data transformation phase."""

    def __init__(
        self,
        message: str,
        transformation_step: Optional[str] = None,
        failed_records: Optional[int] = None,
        **kwargs,
    ):
        """Initialize transformation error.

        Args:
            message: Error message
            transformation_step: Name of the transformation step that failed
            failed_records: Number of records that failed transformation
            **kwargs: Additional arguments for ETLError
        """
        details = kwargs.get("details", {})
        if transformation_step:
            details["transformation_step"] = transformation_step
        if failed_records is not None:
            details["failed_records"] = failed_records

        kwargs["details"] = details
        super().__init__(message, **kwargs)


class QualityCheckError(ETLError):
    """Exception raised during data quality checks."""

    def __init__(
        self,
        message: str,
        failed_checks: Optional[List[str]] = None,
        quality_score: Optional[float] = None,
        **kwargs,
    ):
        """Initialize quality check error.

        Args:
            message: Error message
            failed_checks: List of failed quality check names
            quality_score: Overall quality score (0-1)
            **kwargs: Additional arguments for ETLError
        """
        details = kwargs.get("details", {})
        if failed_checks:
            details["failed_checks"] = failed_checks
        if quality_score is not None:
            details["quality_score"] = quality_score

        kwargs["details"] = details
        super().__init__(message, **kwargs)


class LoadError(ETLError):
    """Exception raised during data loading phase."""

    def __init__(
        self,
        message: str,
        target_table: Optional[str] = None,
        failed_records: Optional[int] = None,
        **kwargs,
    ):
        """Initialize load error.

        Args:
            message: Error message
            target_table: Name of the target table/collection
            failed_records: Number of records that failed to load
            **kwargs: Additional arguments for ETLError
        """
        details = kwargs.get("details", {})
        if target_table:
            details["target_table"] = target_table
        if failed_records is not None:
            details["failed_records"] = failed_records

        kwargs["details"] = details
        super().__init__(message, **kwargs)


class ConfigurationError(ETLError):
    """Exception raised for ETL configuration errors."""

    def __init__(self, message: str, config_section: Optional[str] = None, **kwargs):
        """Initialize configuration error.

        Args:
            message: Error message
            config_section: Configuration section that has the error
            **kwargs: Additional arguments for ETLError
        """
        details = kwargs.get("details", {})
        if config_section:
            details["config_section"] = config_section

        kwargs["details"] = details
        kwargs["recoverable"] = False  # Config errors usually not recoverable
        super().__init__(message, **kwargs)


class ResourceError(ETLError):
    """Exception raised for resource-related errors (disk, memory, etc.)."""

    def __init__(self, message: str, resource_type: Optional[str] = None, **kwargs):
        """Initialize resource error.

        Args:
            message: Error message
            resource_type: Type of resource (disk, memory, network, etc.)
            **kwargs: Additional arguments for ETLError
        """
        details = kwargs.get("details", {})
        if resource_type:
            details["resource_type"] = resource_type

        kwargs["details"] = details
        super().__init__(message, **kwargs)
