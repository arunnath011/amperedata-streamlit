"""Electrochemical visualization exception classes.

This module defines custom exception classes for various electrochemical
analysis and visualization errors.
"""

from typing import Any, Optional


class ElectrochemicalError(Exception):
    """Base electrochemical analysis exception."""

    def __init__(
        self,
        message: str,
        data_id: Optional[str] = None,
        analysis_type: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.data_id = data_id
        self.analysis_type = analysis_type
        self.error_code = error_code
        self.details = details or {}


class DataProcessingError(ElectrochemicalError):
    """Data processing and validation errors."""

    def __init__(
        self,
        message: str,
        processing_step: Optional[str] = None,
        invalid_data_points: Optional[list[int]] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.processing_step = processing_step
        self.invalid_data_points = invalid_data_points or []


class PlottingError(ElectrochemicalError):
    """Plotting and visualization errors."""

    def __init__(
        self,
        message: str,
        plot_type: Optional[str] = None,
        rendering_stage: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.plot_type = plot_type
        self.rendering_stage = rendering_stage


class ComparisonError(ElectrochemicalError):
    """Batch comparison and statistical analysis errors."""

    def __init__(
        self,
        message: str,
        comparison_type: Optional[str] = None,
        dataset_ids: Optional[list[str]] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.comparison_type = comparison_type
        self.dataset_ids = dataset_ids or []


class DataValidationError(DataProcessingError):
    """Data validation specific errors."""

    def __init__(
        self,
        message: str,
        validation_rule: Optional[str] = None,
        field_name: Optional[str] = None,
        expected_value: Optional[Any] = None,
        actual_value: Optional[Any] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.validation_rule = validation_rule
        self.field_name = field_name
        self.expected_value = expected_value
        self.actual_value = actual_value


class SmoothingError(DataProcessingError):
    """Data smoothing and filtering errors."""

    def __init__(
        self,
        message: str,
        smoothing_method: Optional[str] = None,
        window_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.smoothing_method = smoothing_method
        self.window_size = window_size


class FittingError(DataProcessingError):
    """Model fitting and curve fitting errors."""

    def __init__(
        self,
        message: str,
        model_type: Optional[str] = None,
        convergence_error: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.model_type = model_type
        self.convergence_error = convergence_error


class EISAnalysisError(ElectrochemicalError):
    """EIS-specific analysis errors."""

    def __init__(
        self,
        message: str,
        frequency_range: Optional[tuple] = None,
        circuit_model: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.frequency_range = frequency_range
        self.circuit_model = circuit_model


class DifferentialAnalysisError(ElectrochemicalError):
    """Differential analysis specific errors."""

    def __init__(
        self,
        message: str,
        differential_type: Optional[str] = None,
        voltage_range: Optional[tuple] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.differential_type = differential_type
        self.voltage_range = voltage_range


class CycleAnalysisError(ElectrochemicalError):
    """Cycle analysis specific errors."""

    def __init__(
        self,
        message: str,
        cycle_number: Optional[int] = None,
        charge_state: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.cycle_number = cycle_number
        self.charge_state = charge_state


class RateAnalysisError(ElectrochemicalError):
    """Rate capability analysis errors."""

    def __init__(
        self,
        message: str,
        c_rate: Optional[float] = None,
        rate_range: Optional[tuple] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.c_rate = c_rate
        self.rate_range = rate_range


class AgingAnalysisError(ElectrochemicalError):
    """Aging analysis specific errors."""

    def __init__(
        self,
        message: str,
        aging_type: Optional[str] = None,
        time_range: Optional[tuple] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.aging_type = aging_type
        self.time_range = time_range


class StatisticalAnalysisError(ComparisonError):
    """Statistical analysis errors."""

    def __init__(
        self,
        message: str,
        statistical_test: Optional[str] = None,
        sample_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.statistical_test = statistical_test
        self.sample_size = sample_size


class ConfigurationError(ElectrochemicalError):
    """Configuration and parameter errors."""

    def __init__(
        self,
        message: str,
        parameter_name: Optional[str] = None,
        parameter_value: Optional[Any] = None,
        valid_range: Optional[tuple] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.parameter_name = parameter_name
        self.parameter_value = parameter_value
        self.valid_range = valid_range


class ExportError(ElectrochemicalError):
    """Plot export and file I/O errors."""

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
