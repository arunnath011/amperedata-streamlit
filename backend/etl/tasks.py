"""Celery Tasks for ETL Pipeline.

Asynchronous task definitions for ETL pipeline operations including
file ingestion, validation, transformation, and loading.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import structlog
from celery import Task

from backend.celery_app import celery_app
from backend.parsers import BiologicMPTParser, GenericCSVParser, NewareNDAParser

from .exceptions import (
    ETLError,
    IngestionError,
    LoadError,
    QualityCheckError,
    TransformationError,
    ValidationError,
)
from .models import ETLJobStatus, FileFormat, IngestionResult, TransformationResult
from .transformers import DataNormalizer, DataTransformer, UnitConverter
from .validators import BatteryDataValidator, DataValidator, QualityChecker

logger = structlog.get_logger(__name__)


class ETLTask(Task):
    """Base class for ETL tasks with common functionality."""

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure."""
        logger.error(
            "ETL task failed",
            task_id=task_id,
            task_name=self.name,
            exception=str(exc),
            traceback=einfo.traceback,
        )

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Handle task retry."""
        logger.warning(
            "ETL task retrying",
            task_id=task_id,
            task_name=self.name,
            exception=str(exc),
            retry_count=self.request.retries,
        )

    def on_success(self, retval, task_id, args, kwargs):
        """Handle task success."""
        logger.info("ETL task completed successfully", task_id=task_id, task_name=self.name)


@celery_app.task(base=ETLTask, bind=True, max_retries=3, default_retry_delay=60)
def ingest_file_task(self, job_id: str) -> dict[str, Any]:
    """Ingest file and extract data.

    Args:
        job_id: ETL job identifier

    Returns:
        Ingestion result dictionary

    Raises:
        IngestionError: If file ingestion fails
    """
    start_time = time.time()

    try:
        # Get job from pipeline (this would be from a job store in production)
        from .core import ETLPipeline

        pipeline = ETLPipeline()
        job = pipeline._active_jobs.get(job_id)

        if not job:
            raise IngestionError(f"Job {job_id} not found")

        # Update job status
        pipeline.update_job_status(job_id, ETLJobStatus.RUNNING, current_phase="ingestion")

        logger.info(
            "Starting file ingestion",
            job_id=job_id,
            file_path=job.request.source.file_path,
        )

        # Determine parser based on file format
        parser = _get_parser_for_format(job.request.source.file_format, job.request.parser_config)

        # Parse the file
        file_path = job.request.source.file_path

        if job.request.source.file_format == FileFormat.BIOLOGIC_MPT:
            parsed_data, metadata = parser.parse_file(file_path)
            df = parsed_data.to_dataframe()

        elif job.request.source.file_format in [
            FileFormat.NEWARE_NDA,
            FileFormat.NEWARE_NDAX,
        ]:
            parsed_data, metadata = parser.parse_file(file_path)
            df = parsed_data.to_dataframe()

        elif job.request.source.file_format in [
            FileFormat.BATTERY_ARCHIVE_CSV,
            FileFormat.GENERIC_CSV,
        ]:
            parsed_result = parser.parse_file(file_path)
            df = parsed_result.data
            metadata = parsed_result.metadata

        else:
            raise IngestionError(f"Unsupported file format: {job.request.source.file_format}")

        # Create ingestion result
        parsing_duration = time.time() - start_time

        result = IngestionResult(
            success=True,
            records_extracted=len(df),
            columns_detected=df.columns.tolist(),
            file_metadata=metadata if isinstance(metadata, dict) else {},
            parser_used=parser.__class__.__name__,
            parsing_duration_seconds=parsing_duration,
            data_quality_score=(
                getattr(parsed_result, "data_quality_score", None)
                if "parsed_result" in locals()
                else None
            ),
            warnings=[],
        )

        # Store parsed data temporarily (in production, this would go to a data store)
        _store_intermediate_data(job_id, "ingested", df)

        # Update job with ingestion result
        job.ingestion_result = result
        job.progress_percentage = 20.0

        logger.info(
            "File ingestion completed",
            job_id=job_id,
            records_extracted=result.records_extracted,
            duration=parsing_duration,
        )

        return result.model_dump()

    except Exception as e:
        logger.error("File ingestion failed", job_id=job_id, error=str(e))

        # Update job status
        if "pipeline" in locals():
            pipeline.update_job_status(
                job_id,
                ETLJobStatus.FAILED,
                error_message=str(e),
                current_phase="ingestion",
            )

        # Retry on recoverable errors
        if (
            isinstance(e, IngestionError)
            and e.recoverable
            and self.request.retries < self.max_retries
        ):
            raise self.retry(exc=e, countdown=60) from e

        raise IngestionError(f"File ingestion failed: {e}", recoverable=False) from e


@celery_app.task(base=ETLTask, bind=True, max_retries=3, default_retry_delay=60)
def validate_data_task(self, job_id: str, ingestion_result: dict[str, Any]) -> dict[str, Any]:
    """Validate ingested data.

    Args:
        job_id: ETL job identifier
        ingestion_result: Result from ingestion task

    Returns:
        Validation result dictionary

    Raises:
        ValidationError: If data validation fails
    """
    start_time = time.time()

    try:
        # Get job and data
        from .core import ETLPipeline

        pipeline = ETLPipeline()
        job = pipeline._active_jobs.get(job_id)

        if not job:
            raise ValidationError(f"Job {job_id} not found")

        # Update job status
        pipeline.update_job_status(job_id, ETLJobStatus.RUNNING, current_phase="validation")

        logger.info("Starting data validation", job_id=job_id)

        # Load intermediate data
        df = _load_intermediate_data(job_id, "ingested")

        # Get validation configuration
        validation_config = job.request.validation_config
        strict_mode = validation_config.get("strict_mode", False)

        # Choose appropriate validator
        if validation_config.get("battery_data_validation", True):
            validator = BatteryDataValidator(strict_mode=strict_mode)
        else:
            validator = DataValidator(strict_mode=strict_mode)

        # Run validation
        validation_result = validator.validate(df, **validation_config)

        # Store validated data
        _store_intermediate_data(job_id, "validated", df)

        # Update job with validation result
        job.validation_result = validation_result
        job.progress_percentage = 40.0

        # Check if validation passed
        if not validation_result.success and strict_mode:
            raise ValidationError(
                "Data validation failed in strict mode",
                validation_errors=validation_result.validation_errors,
                field_errors=validation_result.field_errors,
            )

        logger.info(
            "Data validation completed",
            job_id=job_id,
            success=validation_result.success,
            errors=len(validation_result.validation_errors),
            warnings=len(validation_result.validation_warnings),
        )

        return validation_result.model_dump()

    except Exception as e:
        logger.error("Data validation failed", job_id=job_id, error=str(e))

        # Update job status
        if "pipeline" in locals():
            pipeline.update_job_status(
                job_id,
                ETLJobStatus.FAILED,
                error_message=str(e),
                current_phase="validation",
            )

        # Retry on recoverable errors
        if (
            isinstance(e, ValidationError)
            and e.recoverable
            and self.request.retries < self.max_retries
        ):
            raise self.retry(exc=e, countdown=60) from e

        raise ValidationError(f"Data validation failed: {e}", recoverable=False) from e


@celery_app.task(base=ETLTask, bind=True, max_retries=3, default_retry_delay=60)
def transform_data_task(self, job_id: str, validation_result: dict[str, Any]) -> dict[str, Any]:
    """Transform and normalize data.

    Args:
        job_id: ETL job identifier
        validation_result: Result from validation task

    Returns:
        Transformation result dictionary

    Raises:
        TransformationError: If data transformation fails
    """
    start_time = time.time()

    try:
        # Get job and data
        from .core import ETLPipeline

        pipeline = ETLPipeline()
        job = pipeline._active_jobs.get(job_id)

        if not job:
            raise TransformationError(f"Job {job_id} not found")

        # Update job status
        pipeline.update_job_status(job_id, ETLJobStatus.RUNNING, current_phase="transformation")

        logger.info("Starting data transformation", job_id=job_id)

        # Load intermediate data
        df = _load_intermediate_data(job_id, "validated")

        # Get transformation configuration
        transform_config = job.request.transformation_config

        # Apply transformations in sequence
        transformed_df = df.copy()
        transformation_steps = []
        total_warnings = []

        # 1. Base data transformation
        if transform_config.get("enable_base_transformation", True):
            transformer = DataTransformer()
            transformed_df, metadata = transformer.transform(transformed_df, **transform_config)
            transformation_steps.append("base_transformation")
            total_warnings.extend(metadata.get("warnings", []))

        # 2. Unit conversion
        if transform_config.get("enable_unit_conversion", True):
            unit_converter = UnitConverter()
            transformed_df, metadata = unit_converter.transform(transformed_df, **transform_config)
            transformation_steps.append("unit_conversion")
            total_warnings.extend(metadata.get("warnings", []))

        # 3. Data normalization
        if transform_config.get("enable_normalization", False):
            normalizer = DataNormalizer()
            transformed_df, metadata = normalizer.transform(transformed_df, **transform_config)
            transformation_steps.append("normalization")
            total_warnings.extend(metadata.get("warnings", []))

        # Create transformation result
        transformation_duration = time.time() - start_time

        result = TransformationResult(
            success=True,
            steps_executed=transformation_steps,
            steps_succeeded=transformation_steps,
            steps_failed=[],
            records_input=len(df),
            records_output=len(transformed_df),
            columns_added=[],  # Would be populated by individual transformers
            columns_removed=[],
            columns_modified=[],
            transformation_duration_seconds=transformation_duration,
            warnings=total_warnings,
        )

        # Store transformed data
        _store_intermediate_data(job_id, "transformed", transformed_df)

        # Update job with transformation result
        job.transformation_result = result
        job.progress_percentage = 60.0

        logger.info(
            "Data transformation completed",
            job_id=job_id,
            steps_executed=len(transformation_steps),
            records_output=result.records_output,
            duration=transformation_duration,
        )

        return result.model_dump()

    except Exception as e:
        logger.error("Data transformation failed", job_id=job_id, error=str(e))

        # Update job status
        if "pipeline" in locals():
            pipeline.update_job_status(
                job_id,
                ETLJobStatus.FAILED,
                error_message=str(e),
                current_phase="transformation",
            )

        # Retry on recoverable errors
        if (
            isinstance(e, TransformationError)
            and e.recoverable
            and self.request.retries < self.max_retries
        ):
            raise self.retry(exc=e, countdown=60) from e

        raise TransformationError(f"Data transformation failed: {e}", recoverable=False) from e


@celery_app.task(base=ETLTask, bind=True, max_retries=3, default_retry_delay=60)
def run_quality_checks_task(
    self, job_id: str, transformation_result: dict[str, Any]
) -> dict[str, Any]:
    """Run data quality checks.

    Args:
        job_id: ETL job identifier
        transformation_result: Result from transformation task

    Returns:
        Quality check result dictionary

    Raises:
        QualityCheckError: If quality checks fail
    """
    start_time = time.time()

    try:
        # Get job and data
        from .core import ETLPipeline

        pipeline = ETLPipeline()
        job = pipeline._active_jobs.get(job_id)

        if not job:
            raise QualityCheckError(f"Job {job_id} not found")

        # Update job status
        pipeline.update_job_status(job_id, ETLJobStatus.RUNNING, current_phase="quality_checks")

        logger.info("Starting quality checks", job_id=job_id)

        # Load intermediate data
        df = _load_intermediate_data(job_id, "transformed")

        # Run quality assessment
        quality_checker = QualityChecker()
        quality_result = quality_checker.assess_quality(df, job.validation_result)

        # Update job with quality result
        job.quality_result = quality_result
        job.progress_percentage = 80.0

        # Check if quality meets minimum threshold
        min_quality_score = job.request.validation_config.get("minimum_quality_score", 0.8)
        if quality_result.overall_score < min_quality_score:
            if job.request.validation_config.get("fail_on_low_quality", False):
                raise QualityCheckError(
                    f"Data quality score {quality_result.overall_score:.3f} below minimum {min_quality_score}",
                    quality_score=quality_result.overall_score,
                    failed_checks=quality_result.checks_failed,
                )

        logger.info(
            "Quality checks completed",
            job_id=job_id,
            overall_score=quality_result.overall_score,
            checks_passed=len(quality_result.checks_passed),
            checks_failed=len(quality_result.checks_failed),
        )

        return quality_result.model_dump()

    except Exception as e:
        logger.error("Quality checks failed", job_id=job_id, error=str(e))

        # Update job status
        if "pipeline" in locals():
            pipeline.update_job_status(
                job_id,
                ETLJobStatus.FAILED,
                error_message=str(e),
                current_phase="quality_checks",
            )

        # Retry on recoverable errors
        if (
            isinstance(e, QualityCheckError)
            and e.recoverable
            and self.request.retries < self.max_retries
        ):
            raise self.retry(exc=e, countdown=60) from e

        raise QualityCheckError(f"Quality checks failed: {e}", recoverable=False) from e


@celery_app.task(base=ETLTask, bind=True, max_retries=3, default_retry_delay=60)
def load_data_task(self, job_id: str, quality_result: dict[str, Any]) -> dict[str, Any]:
    """Load processed data to final destination.

    Args:
        job_id: ETL job identifier
        quality_result: Result from quality checks task

    Returns:
        Load result dictionary

    Raises:
        LoadError: If data loading fails
    """
    start_time = time.time()

    try:
        # Get job and data
        from .core import ETLPipeline

        pipeline = ETLPipeline()
        job = pipeline._active_jobs.get(job_id)

        if not job:
            raise LoadError(f"Job {job_id} not found")

        # Update job status
        pipeline.update_job_status(job_id, ETLJobStatus.RUNNING, current_phase="loading")

        logger.info("Starting data loading", job_id=job_id)

        # Load final processed data
        df = _load_intermediate_data(job_id, "transformed")

        # Determine output location
        output_dir = Path("./data/processed") / job_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save in multiple formats for flexibility
        output_files = {}

        # Parquet format (primary)
        parquet_path = output_dir / "data.parquet"
        df.to_parquet(parquet_path, compression="snappy")
        output_files["parquet"] = str(parquet_path)

        # CSV format (for compatibility)
        csv_path = output_dir / "data.csv"
        df.to_csv(csv_path, index=False)
        output_files["csv"] = str(csv_path)

        # Save metadata
        metadata_path = output_dir / "metadata.json"
        metadata = {
            "job_id": job_id,
            "ingestion_result": job.ingestion_result.model_dump() if job.ingestion_result else None,
            "validation_result": (
                job.validation_result.model_dump() if job.validation_result else None
            ),
            "transformation_result": (
                job.transformation_result.model_dump() if job.transformation_result else None
            ),
            "quality_result": job.quality_result.model_dump() if job.quality_result else None,
            "output_files": output_files,
            "processing_completed_at": datetime.utcnow().isoformat(),
        }

        import json

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Update job completion
        job.output_location = str(output_dir)
        job.records_processed = len(df)
        job.progress_percentage = 100.0

        # Mark job as completed
        pipeline.update_job_status(job_id, ETLJobStatus.COMPLETED)

        loading_duration = time.time() - start_time

        logger.info(
            "Data loading completed",
            job_id=job_id,
            output_location=job.output_location,
            records_processed=job.records_processed,
            duration=loading_duration,
        )

        # Clean up intermediate data
        _cleanup_intermediate_data(job_id)

        return {
            "success": True,
            "output_location": job.output_location,
            "records_processed": job.records_processed,
            "output_files": output_files,
            "loading_duration_seconds": loading_duration,
        }

    except Exception as e:
        logger.error("Data loading failed", job_id=job_id, error=str(e))

        # Update job status
        if "pipeline" in locals():
            pipeline.update_job_status(
                job_id,
                ETLJobStatus.FAILED,
                error_message=str(e),
                current_phase="loading",
            )

        # Retry on recoverable errors
        if isinstance(e, LoadError) and e.recoverable and self.request.retries < self.max_retries:
            raise self.retry(exc=e, countdown=60) from e

        raise LoadError(f"Data loading failed: {e}", recoverable=False) from e


# Helper functions


def _get_parser_for_format(file_format: FileFormat, config: dict[str, Any]) -> Any:
    """Get appropriate parser for file format."""
    if file_format == FileFormat.BIOLOGIC_MPT:
        return BiologicMPTParser(**config.get("biologic", {}))
    elif file_format in [FileFormat.NEWARE_NDA, FileFormat.NEWARE_NDAX]:
        return NewareNDAParser(**config.get("neware", {}))
    elif file_format in [FileFormat.BATTERY_ARCHIVE_CSV, FileFormat.GENERIC_CSV]:
        return GenericCSVParser(**config.get("generic_csv", {}))
    else:
        raise IngestionError(f"No parser available for format: {file_format}")


def _store_intermediate_data(job_id: str, stage: str, data: pd.DataFrame) -> None:
    """Store intermediate data during ETL processing."""
    # In production, this would store to a distributed cache or database
    # For now, store to local filesystem
    temp_dir = Path("./temp/etl") / job_id
    temp_dir.mkdir(parents=True, exist_ok=True)

    file_path = temp_dir / f"{stage}.parquet"
    data.to_parquet(file_path)

    logger.debug("Stored intermediate data", job_id=job_id, stage=stage, file_path=str(file_path))


def _load_intermediate_data(job_id: str, stage: str) -> pd.DataFrame:
    """Load intermediate data during ETL processing."""
    temp_dir = Path("./temp/etl") / job_id
    file_path = temp_dir / f"{stage}.parquet"

    if not file_path.exists():
        raise ETLError(f"Intermediate data not found: {file_path}")

    data = pd.read_parquet(file_path)
    logger.debug("Loaded intermediate data", job_id=job_id, stage=stage, records=len(data))

    return data


def _cleanup_intermediate_data(job_id: str) -> None:
    """Clean up intermediate data files."""
    import shutil

    temp_dir = Path("./temp/etl") / job_id
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
        logger.debug("Cleaned up intermediate data", job_id=job_id)
