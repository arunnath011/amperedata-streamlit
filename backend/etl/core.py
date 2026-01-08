"""ETL Pipeline Core Orchestrator.

Main ETL pipeline class that coordinates the entire data processing workflow
from ingestion through transformation to loading.
"""

from datetime import datetime
from pathlib import Path
from typing import Union, Optional, Any

import structlog

try:
    from celery import chain

    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False
    chain = None
from pydantic import BaseModel

from backend.parsers import BiologicMPTParser, GenericCSVParser, NewareNDAParser

from .exceptions import ConfigurationError, ETLError
from .models import ETLJob, ETLJobStatus, FileFormat, IngestionRequest
from .tasks import (
    ingest_file_task,
    load_data_task,
    run_quality_checks_task,
    transform_data_task,
    validate_data_task,
)
from .transformers import DataNormalizer, DataTransformer, UnitConverter
from .validators import BatteryDataValidator, DataValidator

logger = structlog.get_logger(__name__)


class ETLConfig(BaseModel):
    """ETL Pipeline Configuration."""

    # Parser configurations
    parsers: dict[str, dict[str, Any]] = {
        "biologic": {"strict_validation": False},
        "neware": {"strict_validation": False},
        "generic_csv": {"template_dir": "templates/csv", "auto_detect": True},
    }

    # Validation configuration
    validation: dict[str, Any] = {
        "enabled": True,
        "strict_mode": False,
        "battery_data_validation": True,
        "custom_rules": [],
    }

    # Transformation configuration
    transformation: dict[str, Any] = {
        "enabled": True,
        "unit_conversion": True,
        "data_normalization": True,
        "custom_transformers": [],
    }

    # Quality check configuration
    quality_checks: dict[str, Any] = {
        "enabled": True,
        "minimum_score": 0.8,
        "fail_on_low_quality": False,
        "custom_checks": [],
    }

    # Pipeline settings
    pipeline: dict[str, Any] = {
        "max_retries": 3,
        "retry_delay_seconds": 60,
        "timeout_seconds": 3600,
        "enable_parallel_processing": True,
        "batch_size": 1000,
    }

    # Storage configuration
    storage: dict[str, Any] = {
        "output_format": "parquet",
        "compression": "snappy",
        "partition_by": ["experiment_id", "date"],
        "base_path": "./data/processed",
    }

    # Monitoring configuration
    monitoring: dict[str, Any] = {
        "enabled": True,
        "log_level": "INFO",
        "metrics_enabled": True,
        "alerts_enabled": False,
    }


class ETLPipeline:
    """Main ETL Pipeline orchestrator.

    Coordinates the entire ETL process from file ingestion through
    data transformation to final loading into the data warehouse.
    """

    def __init__(self, config: Optional[ETLConfig] = None):
        """Initialize ETL pipeline.

        Args:
            config: ETL configuration object
        """
        self.config = config or ETLConfig()
        self.logger = logger.bind(component="etl_pipeline")

        # Initialize parsers
        self._parsers = self._initialize_parsers()

        # Initialize validators
        self._validators = self._initialize_validators()

        # Initialize transformers
        self._transformers = self._initialize_transformers()

        # Job tracking
        self._active_jobs: dict[str, ETLJob] = {}

        self.logger.info("ETL Pipeline initialized", config=self.config.model_dump())

    def _initialize_parsers(self) -> dict[str, Any]:
        """Initialize file parsers based on configuration."""
        parsers = {}

        try:
            # BioLogic parser
            biologic_config = self.config.parsers.get("biologic", {})
            parsers["biologic"] = BiologicMPTParser(**biologic_config)

            # Neware parser
            neware_config = self.config.parsers.get("neware", {})
            parsers["neware"] = NewareNDAParser(**neware_config)

            # Generic CSV parser
            csv_config = self.config.parsers.get("generic_csv", {})
            parsers["generic_csv"] = GenericCSVParser(**csv_config)

            self.logger.info("Parsers initialized", parsers=list(parsers.keys()))
            return parsers

        except Exception as e:
            raise ConfigurationError(f"Failed to initialize parsers: {e}")

    def _initialize_validators(self) -> dict[str, Any]:
        """Initialize data validators based on configuration."""
        validators = {}

        try:
            validation_config = self.config.validation

            # Base data validator
            validators["base"] = DataValidator(
                strict_mode=validation_config.get("strict_mode", False)
            )

            # Battery-specific validator
            if validation_config.get("battery_data_validation", True):
                validators["battery"] = BatteryDataValidator(
                    strict_mode=validation_config.get("strict_mode", False)
                )

            self.logger.info("Validators initialized", validators=list(validators.keys()))
            return validators

        except Exception as e:
            raise ConfigurationError(f"Failed to initialize validators: {e}")

    def _initialize_transformers(self) -> dict[str, Any]:
        """Initialize data transformers based on configuration."""
        transformers = {}

        try:
            transform_config = self.config.transformation

            # Base data transformer
            transformers["base"] = DataTransformer()

            # Unit converter
            if transform_config.get("unit_conversion", True):
                transformers["unit_converter"] = UnitConverter()

            # Data normalizer
            if transform_config.get("data_normalization", True):
                transformers["normalizer"] = DataNormalizer()

            self.logger.info("Transformers initialized", transformers=list(transformers.keys()))
            return transformers

        except Exception as e:
            raise ConfigurationError(f"Failed to initialize transformers: {e}")

    def detect_file_format(self, file_path: Union[str, Path]) -> FileFormat:
        """Detect file format based on file extension and content.

        Args:
            file_path: Path to the file

        Returns:
            Detected file format
        """
        path = Path(file_path)
        suffix = path.suffix.lower()

        # Map file extensions to formats
        format_mapping = {
            ".mpt": FileFormat.BIOLOGIC_MPT,
            ".nda": FileFormat.NEWARE_NDA,
            ".ndax": FileFormat.NEWARE_NDAX,
            ".csv": FileFormat.GENERIC_CSV,
            ".txt": FileFormat.GENERIC_CSV,
        }

        detected_format = format_mapping.get(suffix, FileFormat.AUTO_DETECT)

        # Special handling for Battery Archive CSV files
        if detected_format == FileFormat.GENERIC_CSV and "timeseries" in path.name.lower():
            detected_format = FileFormat.BATTERY_ARCHIVE_CSV

        self.logger.info("File format detected", file_path=str(path), format=detected_format.value)

        return detected_format

    async def submit_job(self, request: IngestionRequest) -> str:
        """Submit an ETL job for processing.

        Args:
            request: Ingestion request with job parameters

        Returns:
            Job ID for tracking

        Raises:
            ETLError: If job submission fails
        """
        try:
            # Create ETL job
            job = ETLJob(request=request)
            job.status = ETLJobStatus.PENDING

            # Auto-detect file format if needed
            if request.source.file_format == FileFormat.AUTO_DETECT:
                request.source.file_format = self.detect_file_format(request.source.file_path)

            # Store job for tracking
            self._active_jobs[job.job_id] = job

            # Submit to Celery for async processing
            task_chain = self._create_task_chain(job.job_id)
            task_chain.apply_async()

            self.logger.info(
                "ETL job submitted",
                job_id=job.job_id,
                file_path=request.source.file_path,
                format=request.source.file_format.value,
            )

            return job.job_id

        except Exception as e:
            self.logger.error("Failed to submit ETL job", error=str(e))
            raise ETLError(f"Failed to submit ETL job: {e}")

    def _create_task_chain(self, job_id: str) -> chain:
        """Create Celery task chain for ETL processing.

        Args:
            job_id: Job identifier

        Returns:
            Celery task chain
        """
        # Create sequential task chain
        task_chain = chain(
            ingest_file_task.s(job_id),
            validate_data_task.s(job_id),
            transform_data_task.s(job_id),
            run_quality_checks_task.s(job_id),
            load_data_task.s(job_id),
        )

        return task_chain

    async def get_job_status(self, job_id: str) -> Optional[ETLJob]:
        """Get current status of an ETL job.

        Args:
            job_id: Job identifier

        Returns:
            ETL job object or None if not found
        """
        return self._active_jobs.get(job_id)

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running ETL job.

        Args:
            job_id: Job identifier

        Returns:
            True if job was cancelled, False otherwise
        """
        job = self._active_jobs.get(job_id)
        if not job:
            return False

        if job.status in [ETLJobStatus.PENDING, ETLJobStatus.RUNNING]:
            job.status = ETLJobStatus.CANCELLED
            job.completed_at = datetime.utcnow()

            self.logger.info("ETL job cancelled", job_id=job_id)
            return True

        return False

    async def retry_job(self, job_id: str) -> bool:
        """Retry a failed ETL job.

        Args:
            job_id: Job identifier

        Returns:
            True if job was retried, False otherwise
        """
        job = self._active_jobs.get(job_id)
        if not job or not job.can_retry():
            return False

        job.retry_count += 1
        job.status = ETLJobStatus.RETRYING
        job.error_message = None
        job.error_details = {}

        # Resubmit job
        task_chain = self._create_task_chain(job_id)
        task_chain.apply_async()

        self.logger.info("ETL job retried", job_id=job_id, retry_count=job.retry_count)

        return True

    def get_active_jobs(self) -> list[ETLJob]:
        """Get list of all active jobs."""
        return list(self._active_jobs.values())

    def get_job_metrics(self) -> dict[str, Any]:
        """Get ETL pipeline metrics and statistics."""
        jobs = list(self._active_jobs.values())

        if not jobs:
            return {
                "total_jobs": 0,
                "active_jobs": 0,
                "completed_jobs": 0,
                "failed_jobs": 0,
                "average_duration": 0,
                "success_rate": 0,
            }

        completed_jobs = [j for j in jobs if j.status == ETLJobStatus.COMPLETED]
        failed_jobs = [j for j in jobs if j.status == ETLJobStatus.FAILED]
        active_jobs = [j for j in jobs if j.status in [ETLJobStatus.PENDING, ETLJobStatus.RUNNING]]

        # Calculate average duration for completed jobs
        durations = [j.duration_seconds() for j in completed_jobs if j.duration_seconds()]
        avg_duration = sum(durations) / len(durations) if durations else 0

        return {
            "total_jobs": len(jobs),
            "active_jobs": len(active_jobs),
            "completed_jobs": len(completed_jobs),
            "failed_jobs": len(failed_jobs),
            "average_duration": avg_duration,
            "success_rate": len(completed_jobs) / len(jobs) if jobs else 0,
        }

    async def cleanup_completed_jobs(self, max_age_hours: int = 24) -> int:
        """Clean up completed jobs older than specified age.

        Args:
            max_age_hours: Maximum age in hours for completed jobs

        Returns:
            Number of jobs cleaned up
        """
        cutoff_time = datetime.utcnow().timestamp() - (max_age_hours * 3600)
        jobs_to_remove = []

        for job_id, job in self._active_jobs.items():
            if (
                job.is_complete()
                and job.completed_at
                and job.completed_at.timestamp() < cutoff_time
            ):
                jobs_to_remove.append(job_id)

        for job_id in jobs_to_remove:
            del self._active_jobs[job_id]

        if jobs_to_remove:
            self.logger.info(
                "Cleaned up completed jobs",
                count=len(jobs_to_remove),
                max_age_hours=max_age_hours,
            )

        return len(jobs_to_remove)

    def update_job_status(self, job_id: str, status: ETLJobStatus, **kwargs) -> None:
        """Update job status and metadata.

        Args:
            job_id: Job identifier
            status: New job status
            **kwargs: Additional fields to update
        """
        job = self._active_jobs.get(job_id)
        if not job:
            return

        job.status = status

        # Update timestamps
        if status == ETLJobStatus.RUNNING and not job.started_at:
            job.started_at = datetime.utcnow()
        elif status in [
            ETLJobStatus.COMPLETED,
            ETLJobStatus.FAILED,
            ETLJobStatus.CANCELLED,
        ]:
            job.completed_at = datetime.utcnow()

        # Update additional fields
        for key, value in kwargs.items():
            if hasattr(job, key):
                setattr(job, key, value)

        self.logger.info("Job status updated", job_id=job_id, status=status.value, **kwargs)
