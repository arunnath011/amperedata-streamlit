"""Data Transformation Components for ETL Pipeline.

Comprehensive data transformation system for battery testing data including
unit conversion, normalization, and custom transformations.
"""

import re
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd
import structlog
from pydantic import BaseModel

from .exceptions import TransformationError

logger = structlog.get_logger(__name__)


class UnitConversionRule(BaseModel):
    """Unit conversion rule definition."""

    source_unit: str
    target_unit: str
    conversion_factor: float
    conversion_offset: float = 0.0
    description: Optional[str] = None


class BaseTransformer(ABC):
    """Abstract base class for data transformers."""

    def __init__(self):
        """Initialize transformer."""
        self.logger = logger.bind(component=self.__class__.__name__)

    @abstractmethod
    def transform(self, data: pd.DataFrame, **kwargs) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Transform data and return results.

        Args:
            data: DataFrame to transform
            **kwargs: Additional transformation parameters

        Returns:
            Tuple of (transformed_data, transformation_metadata)
        """

    def _create_transformation_metadata(
        self,
        records_input: int,
        records_output: int,
        columns_added: list[str] = None,
        columns_removed: list[str] = None,
        columns_modified: list[str] = None,
        warnings: list[str] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Create transformation metadata."""
        return {
            "records_input": records_input,
            "records_output": records_output,
            "columns_added": columns_added or [],
            "columns_removed": columns_removed or [],
            "columns_modified": columns_modified or [],
            "warnings": warnings or [],
            **kwargs,
        }


class DataTransformer(BaseTransformer):
    """General-purpose data transformer with common operations."""

    def __init__(self):
        """Initialize data transformer."""
        super().__init__()

    def transform(self, data: pd.DataFrame, **kwargs) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Apply general data transformations.

        Args:
            data: DataFrame to transform
            **kwargs: Transformation parameters

        Returns:
            Tuple of (transformed_data, metadata)
        """
        start_time = datetime.now()
        original_shape = data.shape
        transformed_data = data.copy()

        columns_added = []
        columns_modified = []
        warnings = []

        self.logger.info("Starting data transformation", original_shape=original_shape)

        try:
            # Apply standard transformations

            # 1. Clean column names
            if kwargs.get("clean_column_names", True):
                transformed_data, cleaned_cols = self._clean_column_names(transformed_data)
                if cleaned_cols:
                    columns_modified.extend(cleaned_cols)

            # 2. Handle missing values
            if kwargs.get("handle_missing", True):
                transformed_data, missing_info = self._handle_missing_values(
                    transformed_data, **kwargs
                )
                if missing_info.get("warnings"):
                    warnings.extend(missing_info["warnings"])

            # 3. Standardize data types
            if kwargs.get("standardize_types", True):
                transformed_data, type_info = self._standardize_data_types(transformed_data)
                if type_info.get("modified_columns"):
                    columns_modified.extend(type_info["modified_columns"])

            # 4. Add derived columns
            if kwargs.get("add_derived_columns", True):
                transformed_data, derived_cols = self._add_derived_columns(transformed_data)
                columns_added.extend(derived_cols)

            # 5. Remove duplicate records
            if kwargs.get("remove_duplicates", True):
                original_count = len(transformed_data)
                transformed_data = transformed_data.drop_duplicates()
                duplicates_removed = original_count - len(transformed_data)
                if duplicates_removed > 0:
                    warnings.append(f"Removed {duplicates_removed} duplicate records")

            duration = (datetime.now() - start_time).total_seconds()

            metadata = self._create_transformation_metadata(
                records_input=original_shape[0],
                records_output=len(transformed_data),
                columns_added=columns_added,
                columns_modified=columns_modified,
                warnings=warnings,
                transformation_duration=duration,
            )

            self.logger.info(
                "Data transformation completed",
                input_shape=original_shape,
                output_shape=transformed_data.shape,
                duration=duration,
            )

            return transformed_data, metadata

        except Exception as e:
            self.logger.error("Data transformation failed", error=str(e))
            raise TransformationError(f"Data transformation failed: {e}") from e

    def _clean_column_names(self, data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Clean and standardize column names."""
        original_columns = data.columns.tolist()
        cleaned_columns = []

        for col in original_columns:
            # Convert to lowercase and replace spaces/special chars with underscores
            cleaned = re.sub(r"[^\w\s]", "", str(col).lower())
            cleaned = re.sub(r"\s+", "_", cleaned.strip())
            cleaned = re.sub(r"_+", "_", cleaned)  # Remove multiple underscores
            cleaned = cleaned.strip("_")  # Remove leading/trailing underscores

            cleaned_columns.append(cleaned)

        data.columns = cleaned_columns

        # Return columns that were actually changed
        changed_columns = [
            new_col
            for orig_col, new_col in zip(original_columns, cleaned_columns)
            if orig_col != new_col
        ]

        return data, changed_columns

    def _handle_missing_values(
        self, data: pd.DataFrame, **kwargs
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Handle missing values in the dataset."""
        strategy = kwargs.get("missing_value_strategy", "drop_rows")
        fill_value = kwargs.get("missing_fill_value", 0)

        original_nulls = data.isnull().sum().sum()
        warnings = []

        if original_nulls == 0:
            return data, {"warnings": []}

        if strategy == "drop_rows":
            # Drop rows with any missing values
            data_cleaned = data.dropna()
            rows_dropped = len(data) - len(data_cleaned)
            if rows_dropped > 0:
                warnings.append(f"Dropped {rows_dropped} rows with missing values")

        elif strategy == "drop_columns":
            # Drop columns with missing values
            data_cleaned = data.dropna(axis=1)
            cols_dropped = len(data.columns) - len(data_cleaned.columns)
            if cols_dropped > 0:
                warnings.append(f"Dropped {cols_dropped} columns with missing values")

        elif strategy == "fill_forward":
            # Forward fill missing values
            data_cleaned = data.fillna(method="ffill")
            warnings.append(f"Forward filled {original_nulls} missing values")

        elif strategy == "fill_backward":
            # Backward fill missing values
            data_cleaned = data.fillna(method="bfill")
            warnings.append(f"Backward filled {original_nulls} missing values")

        elif strategy == "fill_value":
            # Fill with specific value
            data_cleaned = data.fillna(fill_value)
            warnings.append(f"Filled {original_nulls} missing values with {fill_value}")

        elif strategy == "interpolate":
            # Interpolate numeric columns
            data_cleaned = data.copy()
            numeric_cols = data_cleaned.select_dtypes(include=[np.number]).columns
            data_cleaned[numeric_cols] = data_cleaned[numeric_cols].interpolate()
            warnings.append(f"Interpolated missing values in {len(numeric_cols)} numeric columns")

        else:
            # No handling - return as is
            data_cleaned = data
            warnings.append(
                f"No missing value handling applied ({original_nulls} missing values remain)"
            )

        return data_cleaned, {"warnings": warnings}

    def _standardize_data_types(self, data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Standardize data types based on column names and content."""
        modified_columns = []

        for col in data.columns:
            # Infer appropriate data type based on column name
            if any(keyword in col.lower() for keyword in ["time", "date", "timestamp"]):
                if not pd.api.types.is_datetime64_any_dtype(data[col]):
                    try:
                        data[col] = pd.to_datetime(data[col], errors="coerce")
                        modified_columns.append(col)
                    except Exception:
                        pass

            elif any(
                keyword in col.lower() for keyword in ["index", "number", "count", "cycle", "step"]
            ):
                if not pd.api.types.is_integer_dtype(data[col]):
                    try:
                        data[col] = pd.to_numeric(data[col], errors="coerce").astype("Int64")
                        modified_columns.append(col)
                    except Exception:
                        pass

            elif any(
                keyword in col.lower()
                for keyword in [
                    "voltage",
                    "current",
                    "capacity",
                    "energy",
                    "temperature",
                    "resistance",
                ]
            ):
                if not pd.api.types.is_numeric_dtype(data[col]):
                    try:
                        data[col] = pd.to_numeric(data[col], errors="coerce")
                        modified_columns.append(col)
                    except Exception:
                        pass

        return data, {"modified_columns": modified_columns}

    def _add_derived_columns(self, data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Add derived columns based on existing data."""
        derived_columns = []

        # Add power column if voltage and current exist
        if "voltage_v" in data.columns and "current_a" in data.columns:
            data["power_w"] = data["voltage_v"] * data["current_a"]
            derived_columns.append("power_w")

        # Add energy efficiency if charge and discharge energy exist
        if "charge_energy_wh" in data.columns and "discharge_energy_wh" in data.columns:
            # Avoid division by zero
            charge_energy = data["charge_energy_wh"].replace(0, np.nan)
            data["energy_efficiency"] = data["discharge_energy_wh"] / charge_energy
            derived_columns.append("energy_efficiency")

        # Add coulombic efficiency if charge and discharge capacity exist
        if "charge_capacity_ah" in data.columns and "discharge_capacity_ah" in data.columns:
            charge_capacity = data["charge_capacity_ah"].replace(0, np.nan)
            data["coulombic_efficiency"] = data["discharge_capacity_ah"] / charge_capacity
            derived_columns.append("coulombic_efficiency")

        # Add time differences if time column exists
        time_cols = [col for col in data.columns if "time" in col.lower()]
        if time_cols:
            time_col = time_cols[0]  # Use first time column
            if pd.api.types.is_datetime64_any_dtype(data[time_col]):
                data["time_delta_s"] = data[time_col].diff().dt.total_seconds()
                derived_columns.append("time_delta_s")
            elif pd.api.types.is_numeric_dtype(data[time_col]):
                data["time_delta_s"] = data[time_col].diff()
                derived_columns.append("time_delta_s")

        return data, derived_columns


class UnitConverter(BaseTransformer):
    """Unit conversion transformer for battery data."""

    def __init__(self):
        """Initialize unit converter."""
        super().__init__()
        self.conversion_rules = self._create_conversion_rules()

    def _create_conversion_rules(self) -> dict[str, UnitConversionRule]:
        """Create unit conversion rules."""
        return {
            # Current conversions
            "ma_to_a": UnitConversionRule(
                source_unit="mA",
                target_unit="A",
                conversion_factor=0.001,
                description="Milliamps to Amps",
            ),
            "a_to_ma": UnitConversionRule(
                source_unit="A",
                target_unit="mA",
                conversion_factor=1000.0,
                description="Amps to Milliamps",
            ),
            # Capacity conversions
            "mah_to_ah": UnitConversionRule(
                source_unit="mAh",
                target_unit="Ah",
                conversion_factor=0.001,
                description="Milliamp-hours to Amp-hours",
            ),
            "ah_to_mah": UnitConversionRule(
                source_unit="Ah",
                target_unit="mAh",
                conversion_factor=1000.0,
                description="Amp-hours to Milliamp-hours",
            ),
            # Energy conversions
            "mwh_to_wh": UnitConversionRule(
                source_unit="mWh",
                target_unit="Wh",
                conversion_factor=0.001,
                description="Milliwatt-hours to Watt-hours",
            ),
            "wh_to_mwh": UnitConversionRule(
                source_unit="Wh",
                target_unit="mWh",
                conversion_factor=1000.0,
                description="Watt-hours to Milliwatt-hours",
            ),
            # Temperature conversions
            "c_to_k": UnitConversionRule(
                source_unit="°C",
                target_unit="K",
                conversion_factor=1.0,
                conversion_offset=273.15,
                description="Celsius to Kelvin",
            ),
            "k_to_c": UnitConversionRule(
                source_unit="K",
                target_unit="°C",
                conversion_factor=1.0,
                conversion_offset=-273.15,
                description="Kelvin to Celsius",
            ),
            "f_to_c": UnitConversionRule(
                source_unit="°F",
                target_unit="°C",
                conversion_factor=5 / 9,
                conversion_offset=-32 * 5 / 9,
                description="Fahrenheit to Celsius",
            ),
            # Resistance conversions
            "mohm_to_ohm": UnitConversionRule(
                source_unit="mΩ",
                target_unit="Ω",
                conversion_factor=0.001,
                description="Milliohms to Ohms",
            ),
        }

    def transform(self, data: pd.DataFrame, **kwargs) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Apply unit conversions to data.

        Args:
            data: DataFrame to transform
            **kwargs: Conversion parameters including 'conversions' list

        Returns:
            Tuple of (transformed_data, metadata)
        """
        start_time = datetime.now()
        transformed_data = data.copy()

        # Get conversion specifications
        conversions = kwargs.get("conversions", [])
        auto_convert = kwargs.get("auto_convert", True)

        columns_modified = []
        warnings = []
        conversions_applied = []

        self.logger.info("Starting unit conversions", conversions_count=len(conversions))

        try:
            # Apply explicit conversions
            for conversion in conversions:
                column = conversion.get("column")
                rule_name = conversion.get("rule")

                if column not in transformed_data.columns:
                    warnings.append(f"Column {column} not found for conversion")
                    continue

                if rule_name not in self.conversion_rules:
                    warnings.append(f"Conversion rule {rule_name} not found")
                    continue

                rule = self.conversion_rules[rule_name]
                transformed_data[column] = self._apply_conversion(transformed_data[column], rule)
                columns_modified.append(column)
                conversions_applied.append(f"{column}: {rule.description}")

            # Apply automatic conversions based on column names
            if auto_convert:
                auto_conversions = self._detect_auto_conversions(transformed_data)
                for column, rule_name in auto_conversions.items():
                    if column not in columns_modified:  # Don't double-convert
                        rule = self.conversion_rules[rule_name]
                        transformed_data[column] = self._apply_conversion(
                            transformed_data[column], rule
                        )
                        columns_modified.append(column)
                        conversions_applied.append(f"{column}: {rule.description} (auto)")

            duration = (datetime.now() - start_time).total_seconds()

            metadata = self._create_transformation_metadata(
                records_input=len(data),
                records_output=len(transformed_data),
                columns_modified=columns_modified,
                warnings=warnings,
                conversions_applied=conversions_applied,
                transformation_duration=duration,
            )

            self.logger.info(
                "Unit conversions completed",
                conversions_applied=len(conversions_applied),
                columns_modified=len(columns_modified),
            )

            return transformed_data, metadata

        except Exception as e:
            self.logger.error("Unit conversion failed", error=str(e))
            raise TransformationError(f"Unit conversion failed: {e}") from e

    def _apply_conversion(self, series: pd.Series, rule: UnitConversionRule) -> pd.Series:
        """Apply unit conversion rule to a pandas Series."""
        # Handle non-numeric data
        if not pd.api.types.is_numeric_dtype(series):
            try:
                series = pd.to_numeric(series, errors="coerce")
            except Exception:
                return series

        # Apply conversion: new_value = (old_value * factor) + offset
        converted = (series * rule.conversion_factor) + rule.conversion_offset

        return converted

    def _detect_auto_conversions(self, data: pd.DataFrame) -> dict[str, str]:
        """Detect columns that should be auto-converted based on naming patterns."""
        auto_conversions = {}

        for column in data.columns:
            col_lower = column.lower()

            # Current conversions (mA -> A)
            if "current" in col_lower and ("ma" in col_lower or "milliamp" in col_lower):
                auto_conversions[column] = "ma_to_a"

            # Capacity conversions (mAh -> Ah)
            elif "capacity" in col_lower and ("mah" in col_lower or "milliamp" in col_lower):
                auto_conversions[column] = "mah_to_ah"

            # Energy conversions (mWh -> Wh)
            elif "energy" in col_lower and ("mwh" in col_lower or "milliwatt" in col_lower):
                auto_conversions[column] = "mwh_to_wh"

            # Resistance conversions (mΩ -> Ω)
            elif "resistance" in col_lower and ("mohm" in col_lower or "milliohm" in col_lower):
                auto_conversions[column] = "mohm_to_ohm"

        return auto_conversions


class DataNormalizer(BaseTransformer):
    """Data normalization and scaling transformer."""

    def __init__(self):
        """Initialize data normalizer."""
        super().__init__()

    def transform(self, data: pd.DataFrame, **kwargs) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Apply data normalization and scaling.

        Args:
            data: DataFrame to normalize
            **kwargs: Normalization parameters

        Returns:
            Tuple of (normalized_data, metadata)
        """
        start_time = datetime.now()
        transformed_data = data.copy()

        normalization_method = kwargs.get("method", "none")
        columns_to_normalize = kwargs.get("columns", [])

        columns_modified = []
        warnings = []
        normalization_stats = {}

        if normalization_method == "none":
            return transformed_data, self._create_transformation_metadata(
                records_input=len(data), records_output=len(transformed_data)
            )

        self.logger.info("Starting data normalization", method=normalization_method)

        try:
            # Auto-select numeric columns if none specified
            if not columns_to_normalize:
                columns_to_normalize = transformed_data.select_dtypes(
                    include=[np.number]
                ).columns.tolist()

            # Filter to existing columns
            columns_to_normalize = [
                col for col in columns_to_normalize if col in transformed_data.columns
            ]

            if normalization_method == "min_max":
                transformed_data, stats = self._apply_min_max_scaling(
                    transformed_data, columns_to_normalize
                )
                normalization_stats.update(stats)
                columns_modified.extend(columns_to_normalize)

            elif normalization_method == "z_score":
                transformed_data, stats = self._apply_z_score_normalization(
                    transformed_data, columns_to_normalize
                )
                normalization_stats.update(stats)
                columns_modified.extend(columns_to_normalize)

            elif normalization_method == "robust":
                transformed_data, stats = self._apply_robust_scaling(
                    transformed_data, columns_to_normalize
                )
                normalization_stats.update(stats)
                columns_modified.extend(columns_to_normalize)

            else:
                warnings.append(f"Unknown normalization method: {normalization_method}")

            duration = (datetime.now() - start_time).total_seconds()

            metadata = self._create_transformation_metadata(
                records_input=len(data),
                records_output=len(transformed_data),
                columns_modified=columns_modified,
                warnings=warnings,
                normalization_method=normalization_method,
                normalization_stats=normalization_stats,
                transformation_duration=duration,
            )

            self.logger.info(
                "Data normalization completed",
                method=normalization_method,
                columns_normalized=len(columns_modified),
            )

            return transformed_data, metadata

        except Exception as e:
            self.logger.error("Data normalization failed", error=str(e))
            raise TransformationError(f"Data normalization failed: {e}") from e

    def _apply_min_max_scaling(
        self, data: pd.DataFrame, columns: list[str]
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Apply min-max scaling (0-1 normalization)."""
        stats = {}

        for col in columns:
            if col not in data.columns:
                continue

            col_data = data[col]
            min_val = col_data.min()
            max_val = col_data.max()

            if max_val == min_val:
                # Constant column - set to 0
                data[col] = 0.0
                stats[col] = {"min": min_val, "max": max_val, "constant": True}
            else:
                data[col] = (col_data - min_val) / (max_val - min_val)
                stats[col] = {"min": min_val, "max": max_val, "constant": False}

        return data, stats

    def _apply_z_score_normalization(
        self, data: pd.DataFrame, columns: list[str]
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Apply z-score normalization (mean=0, std=1)."""
        stats = {}

        for col in columns:
            if col not in data.columns:
                continue

            col_data = data[col]
            mean_val = col_data.mean()
            std_val = col_data.std()

            if std_val == 0:
                # Constant column - set to 0
                data[col] = 0.0
                stats[col] = {"mean": mean_val, "std": std_val, "constant": True}
            else:
                data[col] = (col_data - mean_val) / std_val
                stats[col] = {"mean": mean_val, "std": std_val, "constant": False}

        return data, stats

    def _apply_robust_scaling(
        self, data: pd.DataFrame, columns: list[str]
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Apply robust scaling using median and IQR."""
        stats = {}

        for col in columns:
            if col not in data.columns:
                continue

            col_data = data[col]
            median_val = col_data.median()
            q25 = col_data.quantile(0.25)
            q75 = col_data.quantile(0.75)
            iqr = q75 - q25

            if iqr == 0:
                # No variability - set to 0
                data[col] = 0.0
                stats[col] = {"median": median_val, "iqr": iqr, "constant": True}
            else:
                data[col] = (col_data - median_val) / iqr
                stats[col] = {"median": median_val, "iqr": iqr, "constant": False}

        return data, stats
