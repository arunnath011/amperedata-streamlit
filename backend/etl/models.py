"""ETL Pipeline Data Models.

Pydantic models for ETL pipeline operations, job tracking, and results.
"""

import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Dict, List

from pydantic import BaseModel, Field, field_validator


class ETLJobStatus(str, Enum):
    """Status of ETL job execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class FileFormat(str, Enum):
    """Supported file formats for ingestion."""

    BIOLOGIC_MPT = "biologic_mpt"
    NEWARE_NDA = "neware_nda"
    NEWARE_NDAX = "neware_ndax"
    BATTERY_ARCHIVE_CSV = "battery_archive_csv"
    GENERIC_CSV = "generic_csv"
    AUTO_DETECT = "auto_detect"


class DataSource(BaseModel):
    """Information about the data source."""

    file_path: str = Field(description="Path to the source file")
    file_format: FileFormat = Field(description="Format of the source file")
    file_size_bytes: Optional[int] = Field(None, description="File size in bytes")
    checksum: Optional[str] = Field(None, description="File checksum for integrity")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @field_validator("file_path")
    @classmethod
    def validate_file_path(cls, v: str) -> str:
        """Validate file path exists."""
        path = Path(v)
        if not path.exists():
            raise ValueError(f"File does not exist: {v}")
        return str(path.absolute())


class IngestionRequest(BaseModel):
    """Request for file ingestion."""

    source: DataSource = Field(description="Data source information")
    experiment_id: Optional[str] = Field(None, description="Associated experiment ID")
    user_id: Optional[str] = Field(None, description="User who initiated the ingestion")
    priority: int = Field(default=5, description="Job priority (1-10, higher = more priority)")
    parser_config: Dict[str, Any] = Field(
        default_factory=dict, description="Parser-specific configuration"
    )
    validation_config: Dict[str, Any] = Field(
        default_factory=dict, description="Validation configuration"
    )
    transformation_config: Dict[str, Any] = Field(
        default_factory=dict, description="Transformation configuration"
    )

    @field_validator("priority")
    @classmethod
    def validate_priority(cls, v: int) -> int:
        """Validate priority is in valid range."""
        if not 1 <= v <= 10:
            raise ValueError("Priority must be between 1 and 10")
        return v


class IngestionResult(BaseModel):
    """Result of file ingestion phase."""

    success: bool = Field(description="Whether ingestion was successful")
    records_extracted: int = Field(description="Number of records extracted")
    columns_detected: List[str] = Field(description="List of detected columns")
    file_metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Extracted file metadata"
    )
    parser_used: str = Field(description="Parser that was used")
    parsing_duration_seconds: float = Field(description="Time taken for parsing")
    data_quality_score: Optional[float] = Field(None, description="Initial data quality score")
    warnings: List[str] = Field(default_factory=list, description="Non-fatal warnings")
    error_message: Optional[str] = Field(None, description="Error message if failed")


class ValidationRule(BaseModel):
    """Data validation rule definition."""

    name: str = Field(description="Name of the validation rule")
    description: str = Field(description="Description of what the rule checks")
    rule_type: str = Field(description="Type of rule (range, format, completeness, etc.)")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Rule parameters")
    severity: str = Field(default="error", description="Severity level (error, warning, info)")
    enabled: bool = Field(default=True, description="Whether the rule is enabled")


class ValidationResult(BaseModel):
    """Result of data validation phase."""

    success: bool = Field(description="Whether validation passed")
    rules_applied: List[str] = Field(description="List of validation rules applied")
    rules_passed: List[str] = Field(description="List of rules that passed")
    rules_failed: List[str] = Field(description="List of rules that failed")
    validation_errors: List[str] = Field(
        default_factory=list, description="Validation error messages"
    )
    validation_warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    field_errors: Dict[str, List[str]] = Field(
        default_factory=dict, description="Field-specific errors"
    )
    records_validated: int = Field(description="Number of records validated")
    records_passed: int = Field(description="Number of records that passed validation")
    validation_duration_seconds: float = Field(description="Time taken for validation")


class TransformationStep(BaseModel):
    """Definition of a transformation step."""

    name: str = Field(description="Name of the transformation step")
    transformer: str = Field(description="Transformer class/function name")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Transformation parameters"
    )
    enabled: bool = Field(default=True, description="Whether the step is enabled")
    order: int = Field(description="Execution order of the step")


class TransformationResult(BaseModel):
    """Result of data transformation phase."""

    success: bool = Field(description="Whether transformation was successful")
    steps_executed: List[str] = Field(description="List of transformation steps executed")
    steps_succeeded: List[str] = Field(description="List of steps that succeeded")
    steps_failed: List[str] = Field(description="List of steps that failed")
    records_input: int = Field(description="Number of input records")
    records_output: int = Field(description="Number of output records")
    columns_added: List[str] = Field(
        default_factory=list, description="Columns added during transformation"
    )
    columns_removed: List[str] = Field(
        default_factory=list, description="Columns removed during transformation"
    )
    columns_modified: List[str] = Field(
        default_factory=list, description="Columns modified during transformation"
    )
    transformation_duration_seconds: float = Field(description="Time taken for transformation")
    warnings: List[str] = Field(default_factory=list, description="Transformation warnings")
    error_message: Optional[str] = Field(None, description="Error message if failed")


class QualityCheck(BaseModel):
    """Definition of a data quality check."""

    name: str = Field(description="Name of the quality check")
    description: str = Field(description="Description of the quality check")
    check_type: str = Field(description="Type of check (completeness, accuracy, consistency, etc.)")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Check parameters")
    threshold: Optional[float] = Field(None, description="Pass/fail threshold")
    weight: float = Field(default=1.0, description="Weight in overall quality score")
    enabled: bool = Field(default=True, description="Whether the check is enabled")


class QualityCheckResult(BaseModel):
    """Result of data quality checks."""

    success: bool = Field(description="Whether quality checks passed")
    overall_score: float = Field(description="Overall quality score (0-1)")
    checks_executed: List[str] = Field(description="List of quality checks executed")
    checks_passed: List[str] = Field(description="List of checks that passed")
    checks_failed: List[str] = Field(description="List of checks that failed")
    check_results: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="Detailed check results"
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Quality improvement recommendations"
    )
    quality_duration_seconds: float = Field(description="Time taken for quality checks")


class ETLJob(BaseModel):
    """ETL job tracking and metadata."""

    job_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Unique job identifier"
    )
    request: IngestionRequest = Field(description="Original ingestion request")
    status: ETLJobStatus = Field(default=ETLJobStatus.PENDING, description="Current job status")

    # Timestamps
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Job creation timestamp"
    )
    started_at: Optional[datetime] = Field(None, description="Job start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Job completion timestamp")

    # Phase results
    ingestion_result: Optional[IngestionResult] = Field(None, description="Ingestion phase result")
    validation_result: Optional[ValidationResult] = Field(None, description="Validation phase result")
    transformation_result: Optional[TransformationResult] = Field(
        None, description="Transformation phase result"
    )
    quality_result: Optional[QualityCheckResult] = Field(None, description="Quality check result")

    # Error handling
    error_message: Optional[str] = Field(None, description="Error message if job failed")
    error_details: Dict[str, Any] = Field(
        default_factory=dict, description="Detailed error information"
    )
    retry_count: int = Field(default=0, description="Number of retry attempts")
    max_retries: int = Field(default=3, description="Maximum number of retries")

    # Progress tracking
    current_phase: str = Field(default="pending", description="Current processing phase")
    progress_percentage: float = Field(default=0.0, description="Job progress percentage")

    # Results
    output_location: Optional[str] = Field(None, description="Location of processed data")
    records_processed: int = Field(default=0, description="Total records processed")

    def duration_seconds(self) -> Optional[float]:
        """Calculate job duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        elif self.started_at:
            return (datetime.utcnow() - self.started_at).total_seconds()
        return None

    def is_complete(self) -> bool:
        """Check if job is in a terminal state."""
        return self.status in [
            ETLJobStatus.COMPLETED,
            ETLJobStatus.FAILED,
            ETLJobStatus.CANCELLED,
        ]

    def can_retry(self) -> bool:
        """Check if job can be retried."""
        return self.status == ETLJobStatus.FAILED and self.retry_count < self.max_retries


class ETLMetrics(BaseModel):
    """ETL pipeline metrics and statistics."""

    # Job statistics
    total_jobs: int = Field(description="Total number of jobs processed")
    successful_jobs: int = Field(description="Number of successful jobs")
    failed_jobs: int = Field(description="Number of failed jobs")

    # Performance metrics
    average_duration_seconds: float = Field(description="Average job duration")
    total_records_processed: int = Field(description="Total records processed")
    average_throughput_records_per_second: float = Field(
        description="Average processing throughput"
    )

    # Quality metrics
    average_quality_score: float = Field(description="Average data quality score")

    # Error statistics
    common_errors: Dict[str, int] = Field(
        default_factory=dict, description="Common error types and counts"
    )

    # Resource usage
    peak_memory_usage_mb: Optional[float] = Field(None, description="Peak memory usage in MB")
    total_cpu_time_seconds: Optional[float] = Field(None, description="Total CPU time used")

    # Time period
    period_start: datetime = Field(description="Start of metrics period")
    period_end: datetime = Field(description="End of metrics period")


class PipelineConfig(BaseModel):
    """Configuration for ETL pipeline."""

    # General settings
    max_concurrent_jobs: int = Field(default=5, description="Maximum concurrent ETL jobs")
    default_retry_attempts: int = Field(default=3, description="Default number of retry attempts")
    job_timeout_seconds: int = Field(default=3600, description="Job timeout in seconds")

    # Validation settings
    validation_rules: List[ValidationRule] = Field(
        default_factory=list, description="Validation rules"
    )
    strict_validation: bool = Field(default=False, description="Whether to use strict validation")

    # Transformation settings
    transformation_steps: List[TransformationStep] = Field(
        default_factory=list, description="Transformation steps"
    )
    enable_unit_conversion: bool = Field(
        default=True, description="Enable automatic unit conversion"
    )

    # Quality check settings
    quality_checks: List[QualityCheck] = Field(default_factory=list, description="Quality checks")
    minimum_quality_score: float = Field(
        default=0.8, description="Minimum acceptable quality score"
    )

    # Storage settings
    output_format: str = Field(default="parquet", description="Output data format")
    compression: str = Field(default="snappy", description="Data compression method")

    # Monitoring settings
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    metrics_retention_days: int = Field(default=30, description="Metrics retention period")

    class Config:
        arbitrary_types_allowed = True
