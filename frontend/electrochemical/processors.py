"""Electrochemical data processors for battery analysis.

This module provides data processing and analysis functions for various
types of electrochemical data including cycle analysis, differential analysis,
EIS processing, and statistical comparisons.
"""
from typing import Optional

import logging
from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np

# Try to import scipy for advanced processing
try:
    import scipy.optimize
    import scipy.signal
    import scipy.stats
    from scipy.interpolate import interp1d

    SCIPY_AVAILABLE = True
except ImportError:
    scipy = None
    SCIPY_AVAILABLE = False

from .exceptions import (
    AgingAnalysisError,
    CycleAnalysisError,
    DifferentialAnalysisError,
    EISAnalysisError,
    RateAnalysisError,
    SmoothingError,
    StatisticalAnalysisError,
)
from .models import (
    AgingData,
    AgingResult,
    AnalysisResult,
    AnalysisType,
    CapacityFadeResult,
    ComparisonData,
    ComparisonResult,
    CycleData,
    DataQuality,
    DifferentialData,
    DifferentialResult,
    EISData,
    EISResult,
    ElectrochemicalData,
    ProcessingParameters,
    RateCapabilityData,
    RateCapabilityResult,
)

logger = logging.getLogger(__name__)


class ElectrochemicalProcessor(ABC):
    """Abstract base class for electrochemical data processors."""

    def __init__(self, parameters: Optional[ProcessingParameters] = None):
        """Initialize processor with parameters.

        Args:
            parameters: Processing parameters
        """
        self.parameters = parameters or ProcessingParameters()
        self._processing_start_time = None

    @abstractmethod
    def process(self, data: ElectrochemicalData) -> AnalysisResult:
        """Process electrochemical data.

        Args:
            data: Input electrochemical data

        Returns:
            Analysis result
        """

    def validate_data(self, data: ElectrochemicalData) -> DataQuality:
        """Validate input data quality.

        Args:
            data: Data to validate

        Returns:
            Data quality assessment
        """
        try:
            issues = []
            recommendations = []

            # Check basic data structure
            if not hasattr(data, "id") or not data.id:
                issues.append("Missing data ID")
                recommendations.append("Provide unique data identifier")

            # Check for required fields based on data type
            if isinstance(data, CycleData):
                required_fields = ["time", "voltage", "current", "capacity"]
                for field in required_fields:
                    if not hasattr(data, field) or not getattr(data, field):
                        issues.append(f"Missing required field: {field}")
                        recommendations.append(f"Provide {field} data")

                # Check data array lengths
                if hasattr(data, "time") and data.time:
                    time_length = len(data.time)
                    for field in ["voltage", "current", "capacity"]:
                        if hasattr(data, field) and getattr(data, field):
                            if len(getattr(data, field)) != time_length:
                                issues.append(f"Data length mismatch: {field}")
                                recommendations.append(f"Ensure {field} has same length as time")

            elif isinstance(data, EISData):
                required_fields = ["frequency", "impedance_real", "impedance_imag"]
                for field in required_fields:
                    if not hasattr(data, field) or not getattr(data, field):
                        issues.append(f"Missing required field: {field}")
                        recommendations.append(f"Provide {field} data")

            # Calculate quality metrics
            completeness = 1.0 - (len(issues) / 10.0)  # Normalize to 0-1
            completeness = max(0.0, min(1.0, completeness))

            # Check for outliers and missing values
            outliers_detected = 0
            missing_points = 0

            if isinstance(data, CycleData) and data.voltage:
                voltage_array = np.array(data.voltage)

                # Count missing values (NaN, inf)
                missing_points = np.sum(~np.isfinite(voltage_array))

                # Simple outlier detection using IQR
                if len(voltage_array) > 4:
                    q1, q3 = np.percentile(voltage_array[np.isfinite(voltage_array)], [25, 75])
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    outliers_detected = np.sum(
                        (voltage_array < lower_bound) | (voltage_array > upper_bound)
                    )

            # Calculate consistency (based on data smoothness)
            consistency = 0.8  # Default value
            if isinstance(data, CycleData) and len(data.voltage) > 10:
                voltage_diff = np.diff(data.voltage)
                consistency = 1.0 - min(1.0, np.std(voltage_diff) / np.mean(np.abs(voltage_diff)))

            # Calculate accuracy (placeholder - would need reference data)
            accuracy = 0.9  # Default value

            # Overall quality score
            quality_score = (completeness + consistency + accuracy) / 3.0

            return DataQuality(
                completeness=completeness,
                consistency=consistency,
                accuracy=accuracy,
                outliers_detected=outliers_detected,
                missing_points=missing_points,
                quality_score=quality_score,
                issues=issues,
                recommendations=recommendations,
            )

        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}")
            return DataQuality(
                completeness=0.0,
                consistency=0.0,
                accuracy=0.0,
                outliers_detected=0,
                missing_points=0,
                quality_score=0.0,
                issues=[f"Validation error: {str(e)}"],
                recommendations=["Check data format and content"],
            )

    def smooth_data(self, data: np.ndarray, method: str = "savgol", window: int = 51) -> np.ndarray:
        """Smooth data using specified method.

        Args:
            data: Input data array
            method: Smoothing method
            window: Window size

        Returns:
            Smoothed data array

        Raises:
            SmoothingError: If smoothing fails
        """
        try:
            if not SCIPY_AVAILABLE:
                # Simple moving average fallback
                if len(data) < window:
                    return data

                smoothed = np.convolve(data, np.ones(window) / window, mode="same")
                return smoothed

            if method == "savgol":
                if window >= len(data):
                    window = len(data) - 1 if len(data) % 2 == 0 else len(data) - 2
                if window < 3:
                    return data
                if window % 2 == 0:
                    window -= 1

                return scipy.signal.savgol_filter(data, window, 3)

            elif method == "gaussian":
                sigma = window / 6.0  # Convert window to sigma
                return scipy.signal.gaussian_filter1d(data, sigma)

            elif method == "median":
                if window >= len(data):
                    window = len(data) - 1 if len(data) % 2 == 0 else len(data) - 2
                if window < 3:
                    return data
                if window % 2 == 0:
                    window -= 1

                return scipy.signal.medfilt(data, window)

            elif method == "moving_average":
                if len(data) < window:
                    return data
                return np.convolve(data, np.ones(window) / window, mode="same")

            else:
                raise SmoothingError(f"Unknown smoothing method: {method}")

        except Exception as e:
            logger.error(f"Data smoothing failed: {str(e)}")
            raise SmoothingError(
                f"Smoothing failed: {str(e)}",
                smoothing_method=method,
                window_size=window,
            )

    def remove_outliers(
        self, data: np.ndarray, threshold: float = 3.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Remove outliers from data.

        Args:
            data: Input data array
            threshold: Outlier threshold in standard deviations

        Returns:
            Tuple of (cleaned_data, outlier_mask)
        """
        try:
            if len(data) < 4:
                return data, np.zeros(len(data), dtype=bool)

            # Z-score method
            z_scores = np.abs((data - np.mean(data)) / np.std(data))
            outlier_mask = z_scores > threshold

            cleaned_data = data.copy()
            if np.any(outlier_mask):
                # Replace outliers with interpolated values
                valid_indices = ~outlier_mask
                if np.sum(valid_indices) > 1:
                    interp_func = interp1d(
                        np.where(valid_indices)[0],
                        data[valid_indices],
                        kind="linear",
                        fill_value="extrapolate",
                    )
                    cleaned_data[outlier_mask] = interp_func(np.where(outlier_mask)[0])

            return cleaned_data, outlier_mask

        except Exception as e:
            logger.error(f"Outlier removal failed: {str(e)}")
            return data, np.zeros(len(data), dtype=bool)

    def _start_processing(self):
        """Mark start of processing for timing."""
        self._processing_start_time = datetime.now()

    def _get_processing_time(self) -> float:
        """Get processing time in seconds."""
        if self._processing_start_time:
            return (datetime.now() - self._processing_start_time).total_seconds()
        return 0.0


class CycleAnalyzer(ElectrochemicalProcessor):
    """Analyzer for cycle data and capacity fade."""

    def process(self, data: CycleData) -> CapacityFadeResult:
        """Process cycle data for capacity fade analysis.

        Args:
            data: Cycle data

        Returns:
            Capacity fade analysis result
        """
        self._start_processing()

        try:
            # Validate input data
            quality = self.validate_data(data)

            if quality.quality_score < 0.5:
                raise CycleAnalysisError("Data quality too low for analysis")

            # Extract capacity data
            if not data.capacity:
                raise CycleAnalysisError("No capacity data available")

            capacity_array = np.array(data.capacity)

            # Simple capacity fade analysis
            if len(capacity_array) < 2:
                raise CycleAnalysisError("Insufficient data points for fade analysis")

            # Calculate fade rate (simple linear fit)
            cycles = np.arange(len(capacity_array))

            if SCIPY_AVAILABLE:
                # Linear regression
                slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
                    cycles, capacity_array
                )
                fade_rate_per_cycle = abs(slope / intercept) * 100 if intercept != 0 else 0.0
                r_squared = r_value**2
            else:
                # Simple linear fit fallback
                slope = (capacity_array[-1] - capacity_array[0]) / (len(capacity_array) - 1)
                intercept = capacity_array[0]
                fade_rate_per_cycle = abs(slope / intercept) * 100 if intercept != 0 else 0.0
                r_squared = 0.8  # Placeholder

            # Predict end-of-life (80% capacity retention)
            eol_threshold = 0.8
            current_retention = (
                capacity_array[-1] / capacity_array[0] if capacity_array[0] != 0 else 0.0
            )

            if (
                current_retention > eol_threshold
                and fade_rate_per_cycle > 0
                and not np.isnan(fade_rate_per_cycle)
                and not np.isnan(current_retention)
            ):
                remaining_retention = current_retention - eol_threshold
                fade_rate_decimal = fade_rate_per_cycle / 100
                if fade_rate_decimal > 0:
                    eol_cycle = int(remaining_retention / fade_rate_decimal)
                else:
                    eol_cycle = None
            else:
                eol_cycle = None

            # Create result
            result = CapacityFadeResult(
                analysis_type=AnalysisType.CAPACITY_FADE,
                input_data_id=data.id,
                parameters=self.parameters,
                results={
                    "initial_capacity": float(capacity_array[0]),
                    "final_capacity": float(capacity_array[-1]),
                    "capacity_retention": float(current_retention * 100),
                    "total_cycles": len(capacity_array),
                },
                quality_assessment=quality,
                processing_time=self._get_processing_time(),
                fade_rate_per_cycle=fade_rate_per_cycle,
                fade_rate_per_day=None,  # Would need time data
                eol_cycle=eol_cycle,
                eol_days=None,
                model_r_squared=r_squared,
                model_parameters={"slope": slope, "intercept": intercept},
            )

            return result

        except Exception as e:
            logger.error(f"Cycle analysis failed: {str(e)}")
            raise CycleAnalysisError(f"Analysis failed: {str(e)}")


class DifferentialAnalyzer(ElectrochemicalProcessor):
    """Analyzer for differential capacity (dQ/dV) and voltage (dV/dQ) analysis."""

    def process(self, data: DifferentialData) -> DifferentialResult:
        """Process data for differential analysis.

        Args:
            data: Differential data

        Returns:
            Differential analysis result
        """
        self._start_processing()

        try:
            # Validate input data
            quality = self.validate_data(data)

            if not data.voltage or not data.capacity:
                raise DifferentialAnalysisError("Missing voltage or capacity data")

            voltage = np.array(data.voltage)
            capacity = np.array(data.capacity)

            if len(voltage) != len(capacity):
                raise DifferentialAnalysisError("Voltage and capacity arrays must have same length")

            # Calculate differential
            if len(voltage) < 3:
                raise DifferentialAnalysisError(
                    "Insufficient data points for differential analysis"
                )

            # Smooth data if requested
            if self.parameters.smoothing_enabled:
                voltage = self.smooth_data(
                    voltage,
                    self.parameters.smoothing_method,
                    self.parameters.smoothing_window,
                )
                capacity = self.smooth_data(
                    capacity,
                    self.parameters.smoothing_method,
                    self.parameters.smoothing_window,
                )

            # Calculate dQ/dV
            dv = np.diff(voltage)
            dq = np.diff(capacity)

            # Avoid division by zero
            valid_mask = np.abs(dv) > 1e-6
            dq_dv = np.zeros_like(dv)
            dq_dv[valid_mask] = dq[valid_mask] / dv[valid_mask]

            # Calculate dV/dQ
            valid_mask_q = np.abs(dq) > 1e-6
            dv_dq = np.zeros_like(dq)
            dv_dq[valid_mask_q] = dv[valid_mask_q] / dq[valid_mask_q]

            # Peak detection (simple method)
            peaks_detected = []
            peak_positions = []
            peak_intensities = []

            if SCIPY_AVAILABLE and len(dq_dv) > 10:
                # Find peaks in dQ/dV
                peaks, properties = scipy.signal.find_peaks(np.abs(dq_dv), height=0.1, distance=5)

                for peak_idx in peaks:
                    if peak_idx < len(voltage) - 1:
                        peak_voltage = (voltage[peak_idx] + voltage[peak_idx + 1]) / 2
                        peak_intensity = abs(dq_dv[peak_idx])

                        peaks_detected.append(
                            {
                                "voltage": float(peak_voltage),
                                "intensity": float(peak_intensity),
                                "type": "dq_dv_peak",
                            }
                        )
                        peak_positions.append(float(peak_voltage))
                        peak_intensities.append(float(peak_intensity))

            # Create result
            result = DifferentialResult(
                analysis_type=AnalysisType.DIFFERENTIAL,
                input_data_id=data.id,
                parameters=self.parameters,
                results={
                    "dq_dv_data": dq_dv.tolist(),
                    "dv_dq_data": dv_dq.tolist(),
                    "voltage_points": voltage[:-1].tolist(),  # Differential has one less point
                    "capacity_points": capacity[:-1].tolist(),
                },
                quality_assessment=quality,
                processing_time=self._get_processing_time(),
                peaks_detected=peaks_detected,
                peak_positions=peak_positions,
                peak_intensities=peak_intensities,
                phase_transitions=[],  # Would need more sophisticated analysis
                smoothing_quality=0.8,  # Placeholder
            )

            return result

        except Exception as e:
            logger.error(f"Differential analysis failed: {str(e)}")
            raise DifferentialAnalysisError(f"Analysis failed: {str(e)}")


class EISAnalyzer(ElectrochemicalProcessor):
    """Analyzer for Electrochemical Impedance Spectroscopy data."""

    def process(self, data: EISData) -> EISResult:
        """Process EIS data.

        Args:
            data: EIS data

        Returns:
            EIS analysis result
        """
        self._start_processing()

        try:
            # Validate input data
            quality = self.validate_data(data)

            if not data.frequency or not data.impedance_real or not data.impedance_imag:
                raise EISAnalysisError("Missing required EIS data")

            frequency = np.array(data.frequency)
            z_real = np.array(data.impedance_real)
            z_imag = np.array(data.impedance_imag)

            # Calculate magnitude and phase if not provided
            z_magnitude = np.sqrt(z_real**2 + z_imag**2)
            phase_angle = (
                np.arctan2(-z_imag, z_real) * 180 / np.pi
            )  # Negative for impedance convention

            # Find characteristic frequencies
            characteristic_frequencies = []

            if len(frequency) > 3:
                # Find frequency at maximum imaginary impedance (characteristic frequency)
                max_imag_idx = np.argmax(np.abs(z_imag))
                if max_imag_idx < len(frequency):
                    characteristic_frequencies.append(float(frequency[max_imag_idx]))

            # Extract resistance values
            resistance_values = {}

            if len(z_real) > 0:
                # High frequency resistance (first point, typically)
                resistance_values["r_hf"] = float(z_real[0])

                # Low frequency resistance (last point, typically)
                resistance_values["r_lf"] = float(z_real[-1])

                # Series resistance (minimum real impedance)
                resistance_values["r_series"] = float(np.min(z_real))

            # Simple equivalent circuit fitting (placeholder)
            circuit_parameters = {}
            fit_quality = {"r_squared": 0.85, "chi_squared": 0.01}  # Placeholder values

            # Create result
            result = EISResult(
                analysis_type=AnalysisType.EIS_NYQUIST,
                input_data_id=data.id,
                parameters=self.parameters,
                results={
                    "frequency": frequency.tolist(),
                    "z_real": z_real.tolist(),
                    "z_imag": z_imag.tolist(),
                    "z_magnitude": z_magnitude.tolist(),
                    "phase_angle": phase_angle.tolist(),
                },
                quality_assessment=quality,
                processing_time=self._get_processing_time(),
                equivalent_circuit="R(RC)(RC)",  # Placeholder
                circuit_parameters=circuit_parameters,
                fit_quality=fit_quality,
                characteristic_frequencies=characteristic_frequencies,
                resistance_values=resistance_values,
            )

            return result

        except Exception as e:
            logger.error(f"EIS analysis failed: {str(e)}")
            raise EISAnalysisError(f"Analysis failed: {str(e)}")


class RateAnalyzer(ElectrochemicalProcessor):
    """Analyzer for rate capability data."""

    def process(self, data: RateCapabilityData) -> RateCapabilityResult:
        """Process rate capability data.

        Args:
            data: Rate capability data

        Returns:
            Rate capability analysis result
        """
        self._start_processing()

        try:
            # Validate input data
            quality = self.validate_data(data)

            if not data.c_rates or not data.discharge_capacity:
                raise RateAnalysisError("Missing C-rate or capacity data")

            c_rates = np.array(data.c_rates)
            discharge_capacity = np.array(data.discharge_capacity)

            if len(c_rates) != len(discharge_capacity):
                raise RateAnalysisError("C-rate and capacity arrays must have same length")

            # Find maximum C-rate (where capacity > 50% of initial)
            if len(discharge_capacity) > 0:
                initial_capacity = discharge_capacity[0]  # Assume first point is lowest C-rate
                capacity_threshold = initial_capacity * 0.5

                valid_rates = c_rates[discharge_capacity >= capacity_threshold]
                max_c_rate = (
                    float(np.max(valid_rates)) if len(valid_rates) > 0 else float(c_rates[0])
                )
            else:
                max_c_rate = 0.0

            # Calculate capacity retention at specific C-rates
            capacity_retention_1c = 100.0  # Default
            capacity_retention_5c = None

            if len(c_rates) > 1:
                initial_capacity = discharge_capacity[0]

                # Find closest to 1C
                idx_1c = np.argmin(np.abs(c_rates - 1.0))
                if idx_1c < len(discharge_capacity):
                    capacity_retention_1c = (discharge_capacity[idx_1c] / initial_capacity) * 100

                # Find closest to 5C
                if np.max(c_rates) >= 5.0:
                    idx_5c = np.argmin(np.abs(c_rates - 5.0))
                    if idx_5c < len(discharge_capacity):
                        capacity_retention_5c = (
                            discharge_capacity[idx_5c] / initial_capacity
                        ) * 100

            # Calculate power and energy density (placeholder values)
            power_density_max = 1000.0  # W/kg (placeholder)
            energy_density_1c = 150.0  # Wh/kg (placeholder)

            # Calculate overall rate capability score
            rate_capability_score = min(1.0, capacity_retention_1c / 100.0 * max_c_rate / 5.0)

            # Create result
            result = RateCapabilityResult(
                analysis_type=AnalysisType.RATE_CAPABILITY,
                input_data_id=data.id,
                parameters=self.parameters,
                results={
                    "c_rates": c_rates.tolist(),
                    "discharge_capacity": discharge_capacity.tolist(),
                    "capacity_retention": (
                        (discharge_capacity / discharge_capacity[0]) * 100
                    ).tolist(),
                },
                quality_assessment=quality,
                processing_time=self._get_processing_time(),
                max_c_rate=max_c_rate,
                capacity_retention_at_1c=capacity_retention_1c,
                capacity_retention_at_5c=capacity_retention_5c,
                power_density_max=power_density_max,
                energy_density_at_1c=energy_density_1c,
                rate_capability_score=rate_capability_score,
            )

            return result

        except Exception as e:
            logger.error(f"Rate analysis failed: {str(e)}")
            raise RateAnalysisError(f"Analysis failed: {str(e)}")


class AgingAnalyzer(ElectrochemicalProcessor):
    """Analyzer for aging and degradation data."""

    def process(self, data: AgingData) -> AgingResult:
        """Process aging data.

        Args:
            data: Aging data

        Returns:
            Aging analysis result
        """
        self._start_processing()

        try:
            # Validate input data
            quality = self.validate_data(data)

            if not data.time_days or not data.capacity_retention:
                raise AgingAnalysisError("Missing time or capacity retention data")

            time_days = np.array(data.time_days)
            capacity_retention = np.array(data.capacity_retention)

            if len(time_days) != len(capacity_retention):
                raise AgingAnalysisError("Time and capacity arrays must have same length")

            # Calculate aging rate
            if len(time_days) > 1:
                # Simple linear aging rate
                time_span = time_days[-1] - time_days[0]
                capacity_change = capacity_retention[-1] - capacity_retention[0]
                aging_rate = abs(capacity_change / time_span) if time_span > 0 else 0.0
            else:
                aging_rate = 0.0

            # Predict calendar life (time to reach 80% capacity)
            calendar_life_prediction = None
            if aging_rate > 0 and len(capacity_retention) > 0:
                current_retention = capacity_retention[-1]
                if current_retention > 80.0:
                    remaining_retention = current_retention - 80.0
                    days_to_eol = remaining_retention / aging_rate
                    calendar_life_prediction = days_to_eol / 365.25  # Convert to years

            # Determine dominant aging mechanism (placeholder)
            dominant_aging_mechanism = "SEI growth"  # Would need more sophisticated analysis

            # Create result
            result = AgingResult(
                analysis_type=AnalysisType.CALENDAR_AGING,
                input_data_id=data.id,
                parameters=self.parameters,
                results={
                    "time_days": time_days.tolist(),
                    "capacity_retention": capacity_retention.tolist(),
                    "aging_curve_fit": {},  # Placeholder for fitted curve
                },
                quality_assessment=quality,
                processing_time=self._get_processing_time(),
                aging_rate=aging_rate,
                calendar_life_prediction=calendar_life_prediction,
                cycle_life_prediction=None,  # Would need cycle data
                dominant_aging_mechanism=dominant_aging_mechanism,
                aging_acceleration_factors={},
                model_confidence=0.8,  # Placeholder
            )

            return result

        except Exception as e:
            logger.error(f"Aging analysis failed: {str(e)}")
            raise AgingAnalysisError(f"Analysis failed: {str(e)}")


class ComparisonAnalyzer(ElectrochemicalProcessor):
    """Analyzer for batch comparison and statistical analysis."""

    def process(self, data: ComparisonData) -> ComparisonResult:
        """Process comparison data.

        Args:
            data: Comparison data

        Returns:
            Comparison analysis result
        """
        self._start_processing()

        try:
            if not data.datasets:
                raise StatisticalAnalysisError("No datasets provided for comparison")

            # Extract data for comparison
            dataset_values = {}

            for dataset in data.datasets:
                if isinstance(dataset, CycleData) and dataset.capacity:
                    # Use final capacity for comparison
                    dataset_values[dataset.id] = dataset.capacity[-1]
                elif hasattr(dataset, "value"):
                    dataset_values[dataset.id] = dataset.value

            if not dataset_values:
                raise StatisticalAnalysisError("No comparable values found in datasets")

            # Calculate statistical summary
            values = list(dataset_values.values())
            statistical_summary = {
                "all": {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "median": float(np.median(values)),
                    "count": len(values),
                }
            }

            # Find best and worst performing samples
            best_performing_sample = max(dataset_values.keys(), key=lambda k: dataset_values[k])
            worst_performing_sample = min(dataset_values.keys(), key=lambda k: dataset_values[k])

            # Calculate variability metrics
            variability_metrics = {
                "coefficient_of_variation": float(np.std(values) / np.mean(values))
                if np.mean(values) != 0
                else 0.0,
                "range": float(np.max(values) - np.min(values)),
                "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
            }

            # Placeholder for more sophisticated statistical analysis
            significant_differences = []

            # Create result
            result = ComparisonResult(
                analysis_type=AnalysisType.BATCH_COMPARISON,
                input_data_id=data.id,
                parameters=self.parameters,
                results={
                    "dataset_values": dataset_values,
                    "comparison_type": data.comparison_type,
                    "sample_count": len(data.datasets),
                },
                quality_assessment=DataQuality(
                    completeness=1.0,
                    consistency=0.9,
                    accuracy=0.9,
                    outliers_detected=0,
                    missing_points=0,
                    quality_score=0.9,
                    issues=[],
                    recommendations=[],
                ),
                processing_time=self._get_processing_time(),
                statistical_summary=statistical_summary,
                significant_differences=significant_differences,
                best_performing_sample=best_performing_sample,
                worst_performing_sample=worst_performing_sample,
                variability_metrics=variability_metrics,
                correlation_matrix=None,  # Would need more data
            )

            return result

        except Exception as e:
            logger.error(f"Comparison analysis failed: {str(e)}")
            raise StatisticalAnalysisError(f"Analysis failed: {str(e)}")
