"""ETL Pipeline Core for Battery Data Processing.

This module provides a comprehensive ETL (Extract, Transform, Load) pipeline
for processing battery testing data from various equipment manufacturers.

Key Components:
- Pipeline orchestration using Celery
- File ingestion and validation
- Data transformation and normalization
- Quality checks and validation rules
- Error handling and retry logic
- Monitoring and logging

Supported Data Sources:
- BioLogic .mpt files
- Neware .nda/.ndax files
- Generic CSV/TXT files
- Battery Archive format
"""

from .core import ETLConfig, ETLPipeline
from .exceptions import (
    ETLError,
    IngestionError,
    QualityCheckError,
    TransformationError,
    ValidationError,
)
from .models import (
    DataSource,
    ETLJob,
    ETLJobStatus,
    FileFormat,
    IngestionRequest,
    IngestionResult,
    QualityCheckResult,
    TransformationResult,
    ValidationResult,
)
from .tasks import (
    ingest_file_task,
    load_data_task,
    run_quality_checks_task,
    transform_data_task,
    validate_data_task,
)
from .transformers import DataNormalizer, DataTransformer, UnitConverter
from .validators import BatteryDataValidator, DataValidator, QualityChecker

__all__ = [
    # Core pipeline
    "ETLPipeline",
    "ETLConfig",
    # Celery tasks
    "ingest_file_task",
    "validate_data_task",
    "transform_data_task",
    "load_data_task",
    "run_quality_checks_task",
    # Data models
    "DataSource",
    "FileFormat",
    "IngestionRequest",
    "IngestionResult",
    "ValidationResult",
    "TransformationResult",
    "QualityCheckResult",
    "ETLJobStatus",
    "ETLJob",
    # Validators and transformers
    "DataValidator",
    "BatteryDataValidator",
    "QualityChecker",
    "DataTransformer",
    "UnitConverter",
    "DataNormalizer",
    # Exceptions
    "ETLError",
    "IngestionError",
    "ValidationError",
    "TransformationError",
    "QualityCheckError",
]
