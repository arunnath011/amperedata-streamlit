"""Data Validation Components for ETL Pipeline.

Comprehensive data validation system for battery testing data with
configurable rules, quality scoring, and detailed error reporting.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd
import structlog

from .models import QualityCheck, QualityCheckResult, ValidationResult, ValidationRule

logger = structlog.get_logger(__name__)


class BaseValidator(ABC):
    """Abstract base class for data validators."""

    def __init__(self, strict_mode: bool = False):
        """Initialize validator.

        Args:
            strict_mode: Whether to use strict validation (fail on warnings)
        """
        self.strict_mode = strict_mode
        self.logger = logger.bind(component=self.__class__.__name__)

    @abstractmethod
    def validate(self, data: pd.DataFrame, **kwargs) -> ValidationResult:
        """Validate data and return results.

        Args:
            data: DataFrame to validate
            **kwargs: Additional validation parameters

        Returns:
            Validation result with detailed information
        """

    def _create_validation_result(
        self,
        success: bool,
        rules_applied: list[str],
        rules_passed: list[str],
        rules_failed: list[str],
        errors: list[str],
        warnings: list[str],
        field_errors: dict[str, list[str]],
        records_validated: int,
        records_passed: int,
        duration: float,
    ) -> ValidationResult:
        """Create validation result object."""
        return ValidationResult(
            success=success,
            rules_applied=rules_applied,
            rules_passed=rules_passed,
            rules_failed=rules_failed,
            validation_errors=errors,
            validation_warnings=warnings,
            field_errors=field_errors,
            records_validated=records_validated,
            records_passed=records_passed,
            validation_duration_seconds=duration,
        )


class DataValidator(BaseValidator):
    """General-purpose data validator with configurable rules."""

    def __init__(
        self,
        strict_mode: bool = False,
        custom_rules: Optional[list[ValidationRule]] = None,
    ):
        """Initialize data validator.

        Args:
            strict_mode: Whether to use strict validation
            custom_rules: Custom validation rules to apply
        """
        super().__init__(strict_mode)
        self.custom_rules = custom_rules or []
        self.default_rules = self._create_default_rules()

    def _create_default_rules(self) -> list[ValidationRule]:
        """Create default validation rules."""
        return [
            ValidationRule(
                name="no_empty_dataframe",
                description="DataFrame must not be empty",
                rule_type="completeness",
                severity="error",
            ),
            ValidationRule(
                name="no_all_null_columns",
                description="Columns must not be entirely null",
                rule_type="completeness",
                severity="error",
            ),
            ValidationRule(
                name="numeric_columns_finite",
                description="Numeric columns should contain finite values",
                rule_type="format",
                severity="warning",
            ),
            ValidationRule(
                name="datetime_columns_valid",
                description="DateTime columns should contain valid dates",
                rule_type="format",
                severity="error",
            ),
        ]

    def validate(self, data: pd.DataFrame, **kwargs) -> ValidationResult:
        """Validate DataFrame using configured rules.

        Args:
            data: DataFrame to validate
            **kwargs: Additional validation parameters

        Returns:
            Validation result
        """
        start_time = datetime.now()

        rules_to_apply = self.default_rules + self.custom_rules
        rules_applied = []
        rules_passed = []
        rules_failed = []
        errors = []
        warnings = []
        field_errors = {}

        self.logger.info("Starting data validation", rules_count=len(rules_to_apply))

        for rule in rules_to_apply:
            if not rule.enabled:
                continue

            rules_applied.append(rule.name)

            try:
                passed, messages, field_msgs = self._apply_rule(data, rule)

                if passed:
                    rules_passed.append(rule.name)
                else:
                    rules_failed.append(rule.name)

                    if rule.severity == "error":
                        errors.extend(messages)
                    elif rule.severity == "warning":
                        warnings.extend(messages)

                    # Add field-specific errors
                    for field, msgs in field_msgs.items():
                        if field not in field_errors:
                            field_errors[field] = []
                        field_errors[field].extend(msgs)

            except Exception as e:
                self.logger.error(f"Error applying rule {rule.name}", error=str(e))
                rules_failed.append(rule.name)
                errors.append(f"Rule {rule.name} failed to execute: {e}")

        # Calculate validation success
        has_errors = len(errors) > 0
        has_warnings = len(warnings) > 0
        success = not has_errors and (not self.strict_mode or not has_warnings)

        # Count valid records (simplified - assumes all records are validated)
        records_validated = len(data)
        records_passed = records_validated if success else 0

        duration = (datetime.now() - start_time).total_seconds()

        result = self._create_validation_result(
            success=success,
            rules_applied=rules_applied,
            rules_passed=rules_passed,
            rules_failed=rules_failed,
            errors=errors,
            warnings=warnings,
            field_errors=field_errors,
            records_validated=records_validated,
            records_passed=records_passed,
            duration=duration,
        )

        self.logger.info(
            "Data validation completed",
            success=success,
            rules_passed=len(rules_passed),
            rules_failed=len(rules_failed),
            errors=len(errors),
            warnings=len(warnings),
        )

        return result

    def _apply_rule(
        self, data: pd.DataFrame, rule: ValidationRule
    ) -> tuple[bool, list[str], dict[str, list[str]]]:
        """Apply a single validation rule.

        Args:
            data: DataFrame to validate
            rule: Validation rule to apply

        Returns:
            Tuple of (passed, messages, field_messages)
        """
        if rule.rule_type == "completeness":
            return self._apply_completeness_rule(data, rule)
        elif rule.rule_type == "format":
            return self._apply_format_rule(data, rule)
        elif rule.rule_type == "range":
            return self._apply_range_rule(data, rule)
        elif rule.rule_type == "consistency":
            return self._apply_consistency_rule(data, rule)
        else:
            return False, [f"Unknown rule type: {rule.rule_type}"], {}

    def _apply_completeness_rule(
        self, data: pd.DataFrame, rule: ValidationRule
    ) -> tuple[bool, list[str], dict[str, list[str]]]:
        """Apply completeness validation rules."""
        if rule.name == "no_empty_dataframe":
            if data.empty:
                return False, ["DataFrame is empty"], {}
            return True, [], {}

        elif rule.name == "no_all_null_columns":
            all_null_cols = data.columns[data.isnull().all()].tolist()
            if all_null_cols:
                field_errors = {col: ["Column is entirely null"] for col in all_null_cols}
                return (
                    False,
                    [f"Columns with all null values: {all_null_cols}"],
                    field_errors,
                )
            return True, [], {}

        return True, [], {}

    def _apply_format_rule(
        self, data: pd.DataFrame, rule: ValidationRule
    ) -> tuple[bool, list[str], dict[str, list[str]]]:
        """Apply format validation rules."""
        if rule.name == "numeric_columns_finite":
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            issues = []
            field_errors = {}

            for col in numeric_cols:
                infinite_count = np.isinf(data[col]).sum()
                if infinite_count > 0:
                    issues.append(f"Column {col} has {infinite_count} infinite values")
                    field_errors[col] = [f"{infinite_count} infinite values"]

            return len(issues) == 0, issues, field_errors

        elif rule.name == "datetime_columns_valid":
            datetime_cols = data.select_dtypes(include=["datetime64"]).columns
            issues = []
            field_errors = {}

            for col in datetime_cols:
                null_count = data[col].isnull().sum()
                if null_count > 0:
                    issues.append(f"Column {col} has {null_count} invalid datetime values")
                    field_errors[col] = [f"{null_count} invalid datetime values"]

            return len(issues) == 0, issues, field_errors

        return True, [], {}

    def _apply_range_rule(
        self, data: pd.DataFrame, rule: ValidationRule
    ) -> tuple[bool, list[str], dict[str, list[str]]]:
        """Apply range validation rules."""
        column = rule.parameters.get("column")
        min_val = rule.parameters.get("min_value")
        max_val = rule.parameters.get("max_value")

        if not column or column not in data.columns:
            return False, [f"Column {column} not found"], {}

        issues = []
        field_errors = {}

        if min_val is not None:
            below_min = (data[column] < min_val).sum()
            if below_min > 0:
                issues.append(f"Column {column} has {below_min} values below minimum {min_val}")
                field_errors[column] = field_errors.get(column, []) + [
                    f"{below_min} values below minimum"
                ]

        if max_val is not None:
            above_max = (data[column] > max_val).sum()
            if above_max > 0:
                issues.append(f"Column {column} has {above_max} values above maximum {max_val}")
                field_errors[column] = field_errors.get(column, []) + [
                    f"{above_max} values above maximum"
                ]

        return len(issues) == 0, issues, field_errors

    def _apply_consistency_rule(
        self, data: pd.DataFrame, rule: ValidationRule
    ) -> tuple[bool, list[str], dict[str, list[str]]]:
        """Apply consistency validation rules."""
        # Placeholder for consistency rules (e.g., time series monotonicity)
        return True, [], {}


class BatteryDataValidator(DataValidator):
    """Specialized validator for battery testing data."""

    def __init__(self, strict_mode: bool = False):
        """Initialize battery data validator."""
        super().__init__(strict_mode)
        self.battery_rules = self._create_battery_rules()

    def _create_battery_rules(self) -> list[ValidationRule]:
        """Create battery-specific validation rules."""
        return [
            # Voltage validation
            ValidationRule(
                name="voltage_range",
                description="Voltage should be within reasonable range",
                rule_type="range",
                parameters={"column": "voltage_v", "min_value": 0, "max_value": 10},
                severity="error",
            ),
            # Current validation
            ValidationRule(
                name="current_range",
                description="Current should be within reasonable range",
                rule_type="range",
                parameters={
                    "column": "current_a",
                    "min_value": -1000,
                    "max_value": 1000,
                },
                severity="warning",
            ),
            # Temperature validation
            ValidationRule(
                name="temperature_range",
                description="Temperature should be within reasonable range",
                rule_type="range",
                parameters={
                    "column": "temperature_c",
                    "min_value": -50,
                    "max_value": 100,
                },
                severity="error",
            ),
            # Capacity validation
            ValidationRule(
                name="capacity_positive",
                description="Capacity values should be non-negative",
                rule_type="range",
                parameters={"column": "capacity_ah", "min_value": 0},
                severity="error",
            ),
            # Time series validation
            ValidationRule(
                name="time_monotonic",
                description="Time series should be monotonically increasing",
                rule_type="consistency",
                parameters={"column": "time_s"},
                severity="warning",
            ),
        ]

    def validate(self, data: pd.DataFrame, **kwargs) -> ValidationResult:
        """Validate battery data with specialized rules.

        Args:
            data: Battery data DataFrame
            **kwargs: Additional validation parameters

        Returns:
            Validation result with battery-specific checks
        """
        # Add battery-specific rules to validation
        original_custom_rules = self.custom_rules
        self.custom_rules = self.custom_rules + self.battery_rules

        try:
            result = super().validate(data, **kwargs)

            # Add battery-specific validation logic
            result = self._validate_battery_specific(data, result)

            return result
        finally:
            # Restore original custom rules
            self.custom_rules = original_custom_rules

    def _validate_battery_specific(
        self, data: pd.DataFrame, result: ValidationResult
    ) -> ValidationResult:
        """Add battery-specific validation checks."""

        # Check for required battery columns
        required_columns = ["voltage_v", "current_a"]
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            result.validation_errors.append(f"Missing required battery columns: {missing_columns}")
            result.success = False

        # Check cycle consistency
        if "cycle_index" in data.columns:
            cycle_gaps = self._check_cycle_consistency(data)
            if cycle_gaps:
                result.validation_warnings.append(f"Cycle index gaps detected: {cycle_gaps}")

        # Check energy balance (if energy columns exist)
        if all(col in data.columns for col in ["charge_energy_wh", "discharge_energy_wh"]):
            energy_issues = self._check_energy_balance(data)
            if energy_issues:
                result.validation_warnings.extend(energy_issues)

        return result

    def _check_cycle_consistency(self, data: pd.DataFrame) -> list[int]:
        """Check for gaps in cycle indexing."""
        if "cycle_index" not in data.columns:
            return []

        cycles = data["cycle_index"].unique()
        cycles = sorted(cycles)

        gaps = []
        for i in range(len(cycles) - 1):
            if cycles[i + 1] - cycles[i] > 1:
                gaps.extend(range(cycles[i] + 1, cycles[i + 1]))

        return gaps

    def _check_energy_balance(self, data: pd.DataFrame) -> list[str]:
        """Check energy balance consistency."""
        issues = []

        # Check if discharge energy exceeds charge energy significantly
        if "charge_energy_wh" in data.columns and "discharge_energy_wh" in data.columns:
            charge_energy = data["charge_energy_wh"].max()
            discharge_energy = data["discharge_energy_wh"].max()

            if discharge_energy > charge_energy * 1.1:  # Allow 10% tolerance
                issues.append(
                    f"Discharge energy ({discharge_energy:.2f} Wh) significantly exceeds "
                    f"charge energy ({charge_energy:.2f} Wh)"
                )

        return issues


class QualityChecker:
    """Data quality assessment and scoring system."""

    def __init__(self, custom_checks: Optional[list[QualityCheck]] = None):
        """Initialize quality checker.

        Args:
            custom_checks: Custom quality checks to apply
        """
        self.custom_checks = custom_checks or []
        self.default_checks = self._create_default_checks()
        self.logger = logger.bind(component="quality_checker")

    def _create_default_checks(self) -> list[QualityCheck]:
        """Create default quality checks."""
        return [
            QualityCheck(
                name="completeness",
                description="Data completeness (non-null values)",
                check_type="completeness",
                threshold=0.95,
                weight=0.3,
            ),
            QualityCheck(
                name="consistency",
                description="Data consistency and format",
                check_type="consistency",
                threshold=0.90,
                weight=0.2,
            ),
            QualityCheck(
                name="accuracy",
                description="Data accuracy and range validation",
                check_type="accuracy",
                threshold=0.85,
                weight=0.3,
            ),
            QualityCheck(
                name="timeliness",
                description="Data timeliness and temporal consistency",
                check_type="timeliness",
                threshold=0.80,
                weight=0.2,
            ),
        ]

    def assess_quality(
        self, data: pd.DataFrame, validation_result: ValidationResult
    ) -> QualityCheckResult:
        """Assess overall data quality and generate score.

        Args:
            data: DataFrame to assess
            validation_result: Previous validation results

        Returns:
            Quality check result with detailed scoring
        """
        start_time = datetime.now()

        checks_to_run = self.default_checks + self.custom_checks
        checks_executed = []
        checks_passed = []
        checks_failed = []
        check_results = {}
        recommendations = []

        self.logger.info("Starting quality assessment", checks_count=len(checks_to_run))

        total_weighted_score = 0.0
        total_weight = 0.0

        for check in checks_to_run:
            if not check.enabled:
                continue

            checks_executed.append(check.name)

            try:
                score, details, recs = self._run_quality_check(data, check, validation_result)

                check_results[check.name] = {
                    "score": score,
                    "threshold": check.threshold,
                    "weight": check.weight,
                    "details": details,
                    "passed": score >= check.threshold,
                }

                if score >= check.threshold:
                    checks_passed.append(check.name)
                else:
                    checks_failed.append(check.name)

                recommendations.extend(recs)

                # Calculate weighted score
                total_weighted_score += score * check.weight
                total_weight += check.weight

            except Exception as e:
                self.logger.error(f"Error running quality check {check.name}", error=str(e))
                checks_failed.append(check.name)
                check_results[check.name] = {"score": 0.0, "error": str(e)}

        # Calculate overall score
        overall_score = total_weighted_score / total_weight if total_weight > 0 else 0.0
        success = len(checks_failed) == 0

        duration = (datetime.now() - start_time).total_seconds()

        result = QualityCheckResult(
            success=success,
            overall_score=overall_score,
            checks_executed=checks_executed,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            check_results=check_results,
            recommendations=recommendations,
            quality_duration_seconds=duration,
        )

        self.logger.info(
            "Quality assessment completed",
            overall_score=overall_score,
            checks_passed=len(checks_passed),
            checks_failed=len(checks_failed),
        )

        return result

    def _run_quality_check(
        self,
        data: pd.DataFrame,
        check: QualityCheck,
        validation_result: ValidationResult,
    ) -> tuple[float, dict[str, Any], list[str]]:
        """Run a single quality check.

        Args:
            data: DataFrame to check
            check: Quality check definition
            validation_result: Previous validation results

        Returns:
            Tuple of (score, details, recommendations)
        """
        if check.check_type == "completeness":
            return self._check_completeness(data, check)
        elif check.check_type == "consistency":
            return self._check_consistency(data, check, validation_result)
        elif check.check_type == "accuracy":
            return self._check_accuracy(data, check, validation_result)
        elif check.check_type == "timeliness":
            return self._check_timeliness(data, check)
        else:
            return 0.0, {"error": f"Unknown check type: {check.check_type}"}, []

    def _check_completeness(
        self, data: pd.DataFrame, check: QualityCheck
    ) -> tuple[float, dict[str, Any], list[str]]:
        """Check data completeness."""
        if data.empty:
            return 0.0, {"reason": "Empty dataset"}, ["Provide non-empty dataset"]

        total_cells = data.size
        non_null_cells = total_cells - data.isnull().sum().sum()
        completeness_score = non_null_cells / total_cells

        details = {
            "total_cells": total_cells,
            "non_null_cells": non_null_cells,
            "completeness_percentage": completeness_score * 100,
        }

        recommendations = []
        if completeness_score < 0.95:
            recommendations.append("Consider data cleaning to handle missing values")

        return completeness_score, details, recommendations

    def _check_consistency(
        self,
        data: pd.DataFrame,
        check: QualityCheck,
        validation_result: ValidationResult,
    ) -> tuple[float, dict[str, Any], list[str]]:
        """Check data consistency."""
        # Base consistency on validation results
        total_rules = len(validation_result.rules_applied)
        passed_rules = len(validation_result.rules_passed)

        consistency_score = passed_rules / total_rules if total_rules > 0 else 1.0

        details = {
            "rules_applied": total_rules,
            "rules_passed": passed_rules,
            "validation_errors": len(validation_result.validation_errors),
            "validation_warnings": len(validation_result.validation_warnings),
        }

        recommendations = []
        if consistency_score < 0.9:
            recommendations.append("Review and fix data validation errors")

        return consistency_score, details, recommendations

    def _check_accuracy(
        self,
        data: pd.DataFrame,
        check: QualityCheck,
        validation_result: ValidationResult,
    ) -> tuple[float, dict[str, Any], list[str]]:
        """Check data accuracy."""
        # Simple accuracy check based on outliers and range violations
        numeric_cols = data.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            return 1.0, {"reason": "No numeric columns to check"}, []

        total_values = 0
        accurate_values = 0

        for col in numeric_cols:
            col_data = data[col].dropna()
            if len(col_data) == 0:
                continue

            # Simple outlier detection using IQR
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = ((col_data < lower_bound) | (col_data > upper_bound)).sum()

            total_values += len(col_data)
            accurate_values += len(col_data) - outliers

        accuracy_score = accurate_values / total_values if total_values > 0 else 1.0

        details = {
            "total_values": total_values,
            "accurate_values": accurate_values,
            "outlier_percentage": (1 - accuracy_score) * 100,
        }

        recommendations = []
        if accuracy_score < 0.85:
            recommendations.append("Investigate and handle outlier values")

        return accuracy_score, details, recommendations

    def _check_timeliness(
        self, data: pd.DataFrame, check: QualityCheck
    ) -> tuple[float, dict[str, Any], list[str]]:
        """Check data timeliness."""
        # Look for time-related columns
        time_cols = []
        for col in data.columns:
            if any(time_word in col.lower() for time_word in ["time", "date", "timestamp"]):
                time_cols.append(col)

        if not time_cols:
            return 1.0, {"reason": "No time columns found"}, []

        # Check for temporal consistency (monotonic time series)
        timeliness_scores = []

        for col in time_cols:
            if pd.api.types.is_datetime64_any_dtype(data[col]):
                # Check for monotonic increasing timestamps
                time_data = data[col].dropna().sort_values()
                if len(time_data) > 1:
                    is_monotonic = time_data.is_monotonic_increasing
                    timeliness_scores.append(1.0 if is_monotonic else 0.5)
            elif pd.api.types.is_numeric_dtype(data[col]):
                # Check for monotonic increasing numeric time
                time_data = data[col].dropna()
                if len(time_data) > 1:
                    is_monotonic = time_data.is_monotonic_increasing
                    timeliness_scores.append(1.0 if is_monotonic else 0.5)

        timeliness_score = np.mean(timeliness_scores) if timeliness_scores else 1.0

        details = {"time_columns": time_cols, "monotonic_score": timeliness_score}

        recommendations = []
        if timeliness_score < 0.8:
            recommendations.append("Check time series ordering and consistency")

        return timeliness_score, details, recommendations
