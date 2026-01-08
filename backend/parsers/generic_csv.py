"""Generic CSV/TXT parser with configurable templates and auto-detection.

This module provides a flexible parser for various CSV/TXT formats from different
battery testing equipment manufacturers. It supports:

- Configurable column mapping via YAML templates
- Auto-detection of delimiters, encoding, and headers
- Data validation and type conversion
- Units extraction and normalization
- Extensible template system for new equipment formats

Supported formats:
- Battery Archive CSV files (*timeseries*.csv)
- Generic delimited text files (CSV, TSV, pipe-separated, etc.)
- Files with headers, units rows, and metadata sections
"""

import csv
import re
from pathlib import Path
from typing import Any, Optional, Dict, List, Union, Tuple

import chardet
import pandas as pd
import structlog
import yaml
from pydantic import BaseModel, Field, field_validator

logger = structlog.get_logger(__name__)


class CSVParsingError(Exception):
    """Base exception for CSV parsing errors."""


class CSVFileNotFoundError(CSVParsingError):
    """Raised when CSV file is not found."""


class CSVFormatError(CSVParsingError):
    """Raised when CSV format is invalid or unsupported."""


class CSVDataValidationError(CSVParsingError):
    """Raised when parsed CSV data fails validation."""


class CSVTemplateError(CSVParsingError):
    """Raised when template configuration is invalid."""


class ColumnMapping(BaseModel):
    """Configuration for mapping source columns to standardized names."""

    source_name: str = Field(description="Original column name in the file")
    target_name: str = Field(description="Standardized column name")
    data_type: str = Field(
        default="float", description="Expected data type (int, float, str, datetime)"
    )
    unit: Optional[str] = Field(None, description="Physical unit (V, A, Ah, etc.)")
    required: bool = Field(default=False, description="Whether this column is required")
    validation_range: Optional[Tuple[float, float]] = Field(
        None, description="Valid range for numeric data"
    )
    default_value: Optional[Any] = Field(None, description="Default value if missing")

    @field_validator("data_type")
    @classmethod
    def validate_data_type(cls, v: str) -> str:
        """Validate data type specification."""
        valid_types = ["int", "float", "str", "datetime", "bool"]
        if v not in valid_types:
            raise ValueError(f"Data type must be one of: {valid_types}")
        return v


class ParsingTemplate(BaseModel):
    """Template configuration for parsing specific CSV formats."""

    name: str = Field(description="Template name")
    description: Optional[str] = Field(None, description="Template description")
    file_patterns: List[str] = Field(description="File name patterns this template applies to")

    # File format settings
    delimiter: Optional[str] = Field(None, description="Column delimiter (auto-detect if None)")
    encoding: Optional[str] = Field(None, description="File encoding (auto-detect if None)")
    header_row: Optional[int] = Field(0, description="Row index containing headers (0-based)")
    units_row: Optional[int] = Field(None, description="Row index containing units")
    data_start_row: Optional[int] = Field(None, description="Row index where data starts")
    skip_rows: List[int] = Field(default_factory=list, description="Row indices to skip")

    # Column configuration
    column_mappings: List[ColumnMapping] = Field(description="Column mapping configurations")
    case_sensitive: bool = Field(default=False, description="Case sensitive column matching")

    # Data validation
    min_rows: int = Field(default=1, description="Minimum number of data rows required")
    max_rows: Optional[int] = Field(None, description="Maximum number of data rows allowed")

    # Metadata extraction
    metadata_patterns: Dict[str, str] = Field(
        default_factory=dict, description="Regex patterns to extract metadata from file"
    )

    def get_column_mapping_dict(self, case_sensitive: bool = None) -> Dict[str, ColumnMapping]:
        """Get column mappings as a dictionary for lookup."""
        if case_sensitive is None:
            case_sensitive = self.case_sensitive

        if case_sensitive:
            return {mapping.source_name: mapping for mapping in self.column_mappings}
        else:
            return {mapping.source_name.lower(): mapping for mapping in self.column_mappings}


class ParsedCSVData(BaseModel):
    """Container for parsed CSV data and metadata."""

    # Core data
    data: pd.DataFrame = Field(description="Parsed and validated data")
    original_columns: List[str] = Field(description="Original column names from file")
    mapped_columns: List[str] = Field(description="Standardized column names")

    # File information
    file_path: Optional[str] = Field(None, description="Source file path")
    template_name: Optional[str] = Field(None, description="Template used for parsing")
    encoding: Optional[str] = Field(None, description="Detected file encoding")
    delimiter: Optional[str] = Field(None, description="Detected delimiter")

    # Parsing statistics
    total_rows: int = Field(description="Total number of data rows")
    valid_rows: int = Field(description="Number of valid data rows after cleaning")
    skipped_rows: int = Field(default=0, description="Number of rows skipped")

    # Extracted metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Extracted file metadata")
    units: Dict[str, str] = Field(default_factory=dict, description="Column units")

    # Validation results
    validation_errors: List[str] = Field(default_factory=list, description="Data validation errors")
    data_quality_score: float = Field(default=1.0, description="Data quality score (0-1)")

    class Config:
        arbitrary_types_allowed = True


class GenericCSVParser:
    """Generic CSV parser with configurable templates and auto-detection."""

    def __init__(
        self,
        template_dir: Union[str, Path, None] = None,
        strict_validation: bool = True,
        auto_detect: bool = True,
    ):
        """Initialize the parser.

        Args:
            template_dir: Directory containing YAML template files
            strict_validation: Whether to enforce strict data validation
            auto_detect: Whether to auto-detect file format parameters
        """
        self.template_dir = Path(template_dir) if template_dir else None
        self.strict_validation = strict_validation
        self.auto_detect = auto_detect
        self.logger = logger.bind(parser="generic_csv")

        # Load templates
        self.templates: Dict[str, ParsingTemplate] = {}
        self._load_templates()

        # Common delimiters for auto-detection
        self.common_delimiters = [",", "\t", ";", "|", " "]

        # Default column mappings for common battery data
        self.default_mappings = self._create_default_mappings()

    def _load_templates(self) -> None:
        """Load parsing templates from YAML files."""
        if not self.template_dir or not self.template_dir.exists():
            self.logger.info(
                "No template directory specified or found, using built-in templates only"
            )
            self._create_builtin_templates()
            return

        try:
            for template_file in self.template_dir.glob("*.yaml"):
                with open(template_file, encoding="utf-8") as f:
                    template_data = yaml.safe_load(f)
                    template = ParsingTemplate(**template_data)
                    self.templates[template.name] = template
                    self.logger.info(f"Loaded template: {template.name}")
        except Exception as e:
            self.logger.warning(f"Failed to load templates: {e}")

        # Always include built-in templates
        self._create_builtin_templates()

    def _create_builtin_templates(self) -> None:
        """Create built-in templates for common formats."""
        # Battery Archive template
        battery_archive_template = ParsingTemplate(
            name="battery_archive",
            description="Battery Archive CSV format (*timeseries*.csv)",
            file_patterns=["*timeseries*.csv", "*battery_archive*.csv"],
            delimiter=",",
            header_row=0,
            case_sensitive=False,
            column_mappings=[
                ColumnMapping(
                    source_name="Cycle_Index",
                    target_name="cycle_index",
                    data_type="int",
                    required=True,
                ),
                ColumnMapping(
                    source_name="Current (A)",
                    target_name="current_a",
                    data_type="float",
                    unit="A",
                    required=True,
                ),
                ColumnMapping(
                    source_name="Voltage (V)",
                    target_name="voltage_v",
                    data_type="float",
                    unit="V",
                    required=True,
                ),
                ColumnMapping(
                    source_name="Charge_Capacity (Ah)",
                    target_name="charge_capacity_ah",
                    data_type="float",
                    unit="Ah",
                    required=True,
                ),
                ColumnMapping(
                    source_name="Discharge_Capacity (Ah)",
                    target_name="discharge_capacity_ah",
                    data_type="float",
                    unit="Ah",
                    required=True,
                ),
                ColumnMapping(
                    source_name="Charge_Energy (Wh)",
                    target_name="charge_energy_wh",
                    data_type="float",
                    unit="Wh",
                    required=True,
                ),
                ColumnMapping(
                    source_name="Discharge_Energy (Wh)",
                    target_name="discharge_energy_wh",
                    data_type="float",
                    unit="Wh",
                    required=True,
                ),
                ColumnMapping(
                    source_name="Cell_Temperature (C)",
                    target_name="cell_temperature_c",
                    data_type="float",
                    unit="°C",
                    required=True,
                ),
                ColumnMapping(
                    source_name="Environmental_Temperature (C)",
                    target_name="env_temperature_c",
                    data_type="float",
                    unit="°C",
                    required=False,
                ),
                ColumnMapping(
                    source_name="Test_Time (s)",
                    target_name="test_time_s",
                    data_type="float",
                    unit="s",
                    required=True,
                ),
                ColumnMapping(
                    source_name="Date_Time",
                    target_name="date_time",
                    data_type="datetime",
                    required=True,
                ),
            ],
            min_rows=1,
        )
        self.templates["battery_archive"] = battery_archive_template

        # Generic battery data template
        generic_template = ParsingTemplate(
            name="generic_battery",
            description="Generic battery testing CSV format",
            file_patterns=["*.csv", "*.txt"],
            case_sensitive=False,
            column_mappings=[
                ColumnMapping(
                    source_name="time",
                    target_name="time_s",
                    data_type="float",
                    unit="s",
                ),
                ColumnMapping(
                    source_name="voltage",
                    target_name="voltage_v",
                    data_type="float",
                    unit="V",
                ),
                ColumnMapping(
                    source_name="current",
                    target_name="current_a",
                    data_type="float",
                    unit="A",
                ),
                ColumnMapping(
                    source_name="capacity",
                    target_name="capacity_ah",
                    data_type="float",
                    unit="Ah",
                ),
                ColumnMapping(
                    source_name="energy",
                    target_name="energy_wh",
                    data_type="float",
                    unit="Wh",
                ),
                ColumnMapping(
                    source_name="temperature",
                    target_name="temperature_c",
                    data_type="float",
                    unit="°C",
                ),
                ColumnMapping(source_name="cycle", target_name="cycle_index", data_type="int"),
                ColumnMapping(source_name="step", target_name="step_index", data_type="int"),
            ],
            min_rows=1,
        )
        self.templates["generic_battery"] = generic_template

    def _create_default_mappings(self) -> Dict[str, str]:
        """Create default column name mappings for common variations."""
        return {
            # Time variations
            "time": "time_s",
            "time_s": "time_s",
            "time(s)": "time_s",
            "test_time": "test_time_s",
            "test_time_s": "test_time_s",
            "test_time_(s)": "test_time_s",
            "elapsed_time": "time_s",
            # Voltage variations
            "voltage": "voltage_v",
            "voltage_v": "voltage_v",
            "voltage(v)": "voltage_v",
            "v": "voltage_v",
            "cell_voltage": "voltage_v",
            "ecell": "voltage_v",
            "ewe": "voltage_v",
            # Current variations
            "current": "current_a",
            "current_a": "current_a",
            "current(a)": "current_a",
            "i": "current_a",
            "cell_current": "current_a",
            # Capacity variations
            "capacity": "capacity_ah",
            "capacity_ah": "capacity_ah",
            "capacity(ah)": "capacity_ah",
            "charge_capacity": "charge_capacity_ah",
            "discharge_capacity": "discharge_capacity_ah",
            # Energy variations
            "energy": "energy_wh",
            "energy_wh": "energy_wh",
            "energy(wh)": "energy_wh",
            "charge_energy": "charge_energy_wh",
            "discharge_energy": "discharge_energy_wh",
            # Temperature variations
            "temperature": "temperature_c",
            "temperature_c": "temperature_c",
            "temperature(c)": "temperature_c",
            "temp": "temperature_c",
            "cell_temperature": "cell_temperature_c",
            "env_temperature": "env_temperature_c",
            # Cycle/Step variations
            "cycle": "cycle_index",
            "cycle_index": "cycle_index",
            "cycle_number": "cycle_index",
            "step": "step_index",
            "step_index": "step_index",
            "step_number": "step_index",
        }

    def detect_encoding(self, file_path: Union[str, Path]) -> str:
        """Detect file encoding using chardet."""
        try:
            with open(file_path, "rb") as f:
                # Read first 10KB for detection
                raw_data = f.read(10240)
                result = chardet.detect(raw_data)
                encoding = result["encoding"]
                confidence = result["confidence"]

                self.logger.info(
                    f"Detected encoding: {encoding} (confidence: {confidence:.2f})",
                    file_path=str(file_path),
                )

                # Fallback to common encodings if confidence is low
                if confidence < 0.7:
                    for fallback in ["utf-8", "latin-1", "cp1252"]:
                        try:
                            with open(file_path, encoding=fallback) as test_f:
                                test_f.read(1024)  # Try to read a bit
                            self.logger.info(f"Using fallback encoding: {fallback}")
                            return fallback
                        except UnicodeDecodeError:
                            continue

                return encoding or "utf-8"
        except Exception as e:
            self.logger.warning(f"Encoding detection failed: {e}, using utf-8")
            return "utf-8"

    def detect_delimiter(self, file_path: Union[str, Path], encoding: str) -> str:
        """Detect CSV delimiter by analyzing the file."""
        try:
            with open(file_path, encoding=encoding) as f:
                # Read first few lines
                sample_lines = []
                for i, line in enumerate(f):
                    if i >= 10:  # Analyze first 10 lines
                        break
                    sample_lines.append(line.strip())

            if not sample_lines:
                return ","

            # Use csv.Sniffer to detect delimiter
            sample_text = "\n".join(sample_lines)
            try:
                sniffer = csv.Sniffer()
                delimiter = sniffer.sniff(sample_text, delimiters=",\t;| ").delimiter
                self.logger.info(f"Detected delimiter: '{delimiter}'")
                return delimiter
            except csv.Error:
                pass

            # Fallback: count occurrences of common delimiters
            delimiter_counts = {}
            for delimiter in self.common_delimiters:
                count = sum(line.count(delimiter) for line in sample_lines)
                if count > 0:
                    delimiter_counts[delimiter] = count

            if delimiter_counts:
                best_delimiter = max(delimiter_counts, key=delimiter_counts.get)
                self.logger.info(f"Detected delimiter by counting: '{best_delimiter}'")
                return best_delimiter

            # Default to comma
            return ","

        except Exception as e:
            self.logger.warning(f"Delimiter detection failed: {e}, using comma")
            return ","

    def find_matching_template(self, file_path: Union[str, Path]) -> Optional[ParsingTemplate]:
        """Find the best matching template for a file."""
        file_name = Path(file_path).name.lower()

        # Check each template's file patterns
        for template in self.templates.values():
            for pattern in template.file_patterns:
                if self._matches_pattern(file_name, pattern.lower()):
                    self.logger.info(f"Matched template '{template.name}' for file: {file_name}")
                    return template

        self.logger.info(f"No specific template matched for {file_name}, will use generic approach")
        return None

    def _matches_pattern(self, filename: str, pattern: str) -> bool:
        """Check if filename matches a pattern (supports * wildcards)."""
        import fnmatch

        return fnmatch.fnmatch(filename, pattern)

    def parse_file(
        self, file_path: Union[str, Path], template_name: Optional[str] = None, **kwargs
    ) -> ParsedCSVData:
        """Parse a CSV/TXT file.

        Args:
            file_path: Path to the file to parse
            template_name: Specific template to use (auto-detect if None)
            **kwargs: Override parameters (delimiter, encoding, etc.)

        Returns:
            ParsedCSVData object containing parsed data and metadata

        Raises:
            CSVFileNotFoundError: If file doesn't exist
            CSVFormatError: If file format is invalid
            CSVDataValidationError: If data validation fails
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise CSVFileNotFoundError(f"File not found: {file_path}")

        self.logger.info("Parsing CSV file", file_path=str(file_path))

        try:
            # Find or use specified template
            template = None
            if template_name:
                template = self.templates.get(template_name)
                if not template:
                    raise CSVTemplateError(f"Template '{template_name}' not found")
            else:
                template = self.find_matching_template(file_path)

            # Auto-detect file parameters if needed
            encoding = kwargs.get("encoding") or (template.encoding if template else None)
            if not encoding and self.auto_detect:
                encoding = self.detect_encoding(file_path)
            encoding = encoding or "utf-8"

            delimiter = kwargs.get("delimiter") or (template.delimiter if template else None)
            if not delimiter and self.auto_detect:
                delimiter = self.detect_delimiter(file_path, encoding)
            delimiter = delimiter or ","

            # Read the file
            df = self._read_csv_file(file_path, encoding, delimiter, template, **kwargs)

            # Apply template mappings if available
            if template:
                df, metadata, units = self._apply_template(df, template, file_path)
            else:
                df, metadata, units = self._apply_generic_mapping(df, file_path)

            # Validate data
            validation_errors = []
            if self.strict_validation:
                validation_errors = self._validate_data(df, template)

            # Calculate data quality score
            quality_score = self._calculate_quality_score(df, validation_errors)

            # Create result object
            result = ParsedCSVData(
                data=df,
                original_columns=list(df.columns),
                mapped_columns=list(df.columns),
                file_path=str(file_path),
                template_name=template.name if template else "auto_detected",
                encoding=encoding,
                delimiter=delimiter,
                total_rows=len(df),
                valid_rows=len(df.dropna()),
                metadata=metadata,
                units=units,
                validation_errors=validation_errors,
                data_quality_score=quality_score,
            )

            self.logger.info(
                "Successfully parsed CSV file",
                file_path=str(file_path),
                rows=len(df),
                columns=len(df.columns),
                template=template.name if template else "generic",
            )

            return result

        except Exception as e:
            self.logger.error("Failed to parse CSV file", file_path=str(file_path), error=str(e))
            if isinstance(
                e,
                (
                    CSVFileNotFoundError,
                    CSVFormatError,
                    CSVDataValidationError,
                    CSVTemplateError,
                ),
            ):
                raise
            else:
                raise CSVFormatError(f"Failed to parse CSV file: {e}") from e

    def _read_csv_file(
        self,
        file_path: Path,
        encoding: str,
        delimiter: str,
        template: Optional[ParsingTemplate],
        **kwargs,
    ) -> pd.DataFrame:
        """Read CSV file with specified parameters."""
        read_kwargs = {
            "encoding": encoding,
            "sep": delimiter,
        }

        # Apply template-specific settings
        if template:
            if template.header_row is not None:
                read_kwargs["header"] = template.header_row
            if template.skip_rows:
                read_kwargs["skiprows"] = template.skip_rows

        # Override with explicit kwargs
        read_kwargs.update(kwargs)

        try:
            df = pd.read_csv(file_path, **read_kwargs)
            self.logger.info(f"Read CSV: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except pd.errors.ParserError as e:
            # Try to handle malformed CSV by being more lenient
            self.logger.warning(f"CSV parsing error, trying with error_bad_lines=False: {e}")
            try:
                # Try with on_bad_lines='skip' for pandas >= 1.3
                read_kwargs["on_bad_lines"] = "skip"
                df = pd.read_csv(file_path, **read_kwargs)
                self.logger.info(
                    f"Read CSV with skipped bad lines: {df.shape[0]} rows, {df.shape[1]} columns"
                )
                return df
            except Exception:
                # Fallback: try with warn
                try:
                    read_kwargs["on_bad_lines"] = "warn"
                    df = pd.read_csv(file_path, **read_kwargs)
                    self.logger.info(
                        f"Read CSV with warnings: {df.shape[0]} rows, {df.shape[1]} columns"
                    )
                    return df
                except Exception:
                    pass
            raise CSVFormatError(f"Failed to read CSV file: {e}") from e
        except Exception as e:
            raise CSVFormatError(f"Failed to read CSV file: {e}") from e

    def _apply_template(
        self, df: pd.DataFrame, template: ParsingTemplate, file_path: Path
    ) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, str]]:
        """Apply template mappings and validation."""
        df.columns.tolist()
        column_mapping = template.get_column_mapping_dict()

        # Create mapping for renaming columns
        rename_mapping = {}
        units = {}

        # Find matching columns
        for col in df.columns:
            col_key = col if template.case_sensitive else col.lower()
            if col_key in column_mapping:
                mapping = column_mapping[col_key]
                rename_mapping[col] = mapping.target_name
                if mapping.unit:
                    units[mapping.target_name] = mapping.unit

        # Check for required columns
        required_mappings = {m.source_name: m for m in template.column_mappings if m.required}
        missing_required = []

        for req_col, mapping in required_mappings.items():
            col_key = req_col if template.case_sensitive else req_col.lower()
            found = any(
                (c if template.case_sensitive else c.lower()) == col_key for c in df.columns
            )
            if not found:
                missing_required.append(req_col)

        if missing_required and self.strict_validation:
            raise CSVDataValidationError(f"Missing required columns: {missing_required}")

        # Rename columns
        df = df.rename(columns=rename_mapping)

        # Apply data type conversions
        for col in df.columns:
            # Find the mapping for this column
            mapping = None
            for m in template.column_mappings:
                if m.target_name == col:
                    mapping = m
                    break

            if mapping:
                df = self._convert_column_type(df, col, mapping)

        # Extract metadata
        metadata = self._extract_metadata(file_path, template)

        return df, metadata, units

    def _apply_generic_mapping(
        self, df: pd.DataFrame, file_path: Path
    ) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, str]]:
        """Apply generic column mappings when no template matches."""
        rename_mapping = {}
        units = {}

        # Try to map columns using default mappings
        for col in df.columns:
            col_lower = col.lower().strip()
            # Remove common prefixes/suffixes and special characters
            cleaned_col = re.sub(r"[^\w\s]", "", col_lower).strip()

            if cleaned_col in self.default_mappings:
                target_name = self.default_mappings[cleaned_col]
                rename_mapping[col] = target_name

                # Infer units from column names
                if "voltage" in cleaned_col or "v" in cleaned_col:
                    units[target_name] = "V"
                elif "current" in cleaned_col or "i" in cleaned_col:
                    units[target_name] = "A"
                elif "capacity" in cleaned_col and "ah" in cleaned_col:
                    units[target_name] = "Ah"
                elif "energy" in cleaned_col and "wh" in cleaned_col:
                    units[target_name] = "Wh"
                elif "temperature" in cleaned_col or "temp" in cleaned_col:
                    units[target_name] = "°C"
                elif "time" in cleaned_col:
                    units[target_name] = "s"

        # Rename columns
        df = df.rename(columns=rename_mapping)

        # Generic type inference
        for col in df.columns:
            if col.endswith("_c"):  # Temperature
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif col.endswith(("_v", "_a", "_ah", "_wh", "_s")):  # Numeric columns
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif "index" in col or "number" in col:  # Integer columns
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            elif "date" in col or "time" in col:  # Datetime columns
                df[col] = pd.to_datetime(df[col], errors="coerce")

        metadata = {"file_path": str(file_path), "parsing_method": "generic"}
        return df, metadata, units

    def _convert_column_type(
        self, df: pd.DataFrame, col: str, mapping: ColumnMapping
    ) -> pd.DataFrame:
        """Convert column to specified data type."""
        if col not in df.columns:
            return df

        try:
            if mapping.data_type == "int":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            elif mapping.data_type == "float":
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif mapping.data_type == "datetime":
                df[col] = pd.to_datetime(df[col], errors="coerce")
            elif mapping.data_type == "bool":
                df[col] = df[col].astype("bool")
            else:  # str
                df[col] = df[col].astype(str)
        except Exception as e:
            self.logger.warning(f"Failed to convert column {col} to {mapping.data_type}: {e}")

        return df

    def _extract_metadata(self, file_path: Path, template: ParsingTemplate) -> Dict[str, Any]:
        """Extract metadata using template patterns."""
        metadata = {"file_path": str(file_path)}

        if not template.metadata_patterns:
            return metadata

        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            for key, pattern in template.metadata_patterns.items():
                match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
                if match:
                    metadata[key] = match.group(1) if match.groups() else match.group(0)
        except Exception as e:
            self.logger.warning(f"Failed to extract metadata: {e}")

        return metadata

    def _validate_data(self, df: pd.DataFrame, template: Optional[ParsingTemplate]) -> List[str]:
        """Validate parsed data against template requirements."""
        errors = []

        if template:
            # Check minimum rows
            if len(df) < template.min_rows:
                errors.append(f"Insufficient data: {len(df)} rows < {template.min_rows} required")

            # Check maximum rows
            if template.max_rows and len(df) > template.max_rows:
                errors.append(f"Too much data: {len(df)} rows > {template.max_rows} allowed")

            # Validate column ranges
            for mapping in template.column_mappings:
                if mapping.target_name in df.columns and mapping.validation_range:
                    min_val, max_val = mapping.validation_range
                    col_data = df[mapping.target_name].dropna()
                    if len(col_data) > 0:
                        if col_data.min() < min_val or col_data.max() > max_val:
                            errors.append(
                                f"Column {mapping.target_name} values outside valid range "
                                f"[{min_val}, {max_val}]"
                            )

        # Generic validation
        numeric_cols = df.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            if df[col].isna().all():
                errors.append(f"Column {col} contains no valid data")

        return errors

    def _calculate_quality_score(self, df: pd.DataFrame, validation_errors: List[str]) -> float:
        """Calculate data quality score (0-1)."""
        if len(validation_errors) > 0:
            # Deduct points for validation errors
            error_penalty = min(len(validation_errors) * 0.1, 0.5)
        else:
            error_penalty = 0

        # Check completeness
        total_cells = df.size
        missing_cells = df.isna().sum().sum()
        completeness = 1 - (missing_cells / total_cells) if total_cells > 0 else 0

        # Combined score
        quality_score = max(0, completeness - error_penalty)
        return round(quality_score, 3)

    def create_template(
        self, name: str, description: str, sample_file: Union[str, Path], **kwargs
    ) -> ParsingTemplate:
        """Create a new parsing template by analyzing a sample file."""
        sample_path = Path(sample_file)
        if not sample_path.exists():
            raise CSVFileNotFoundError(f"Sample file not found: {sample_path}")

        # Auto-detect parameters
        encoding = self.detect_encoding(sample_path)
        delimiter = self.detect_delimiter(sample_path, encoding)

        # Read sample to analyze structure
        df = pd.read_csv(sample_path, encoding=encoding, sep=delimiter, nrows=100)

        # Create column mappings based on detected columns
        column_mappings = []
        for col in df.columns:
            col_lower = col.lower().strip()
            cleaned_col = re.sub(r"[^\w\s]", "", col_lower).strip()

            # Infer data type
            data_type = "str"
            if df[col].dtype in ["int64", "int32"]:
                data_type = "int"
            elif df[col].dtype in ["float64", "float32"]:
                data_type = "float"

            # Check if it's a required column (heuristic)
            required = col_lower in ["voltage", "current", "time", "cycle"]

            mapping = ColumnMapping(
                source_name=col,
                target_name=self.default_mappings.get(cleaned_col, col_lower.replace(" ", "_")),
                data_type=data_type,
                required=required,
            )
            column_mappings.append(mapping)

        # Create template
        template = ParsingTemplate(
            name=name,
            description=description,
            file_patterns=[f"*{sample_path.suffix}"],
            delimiter=delimiter,
            encoding=encoding,
            header_row=0,
            column_mappings=column_mappings,
            **kwargs,
        )

        return template

    def save_template(self, template: ParsingTemplate, file_path: Union[str, Path]) -> None:
        """Save a template to a YAML file."""
        template_data = template.model_dump()

        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(template_data, f, default_flow_style=False, indent=2)

        self.logger.info(f"Saved template '{template.name}' to {file_path}")

    def list_templates(self) -> List[str]:
        """List available template names."""
        return list(self.templates.keys())

    def get_template(self, name: str) -> Optional[ParsingTemplate]:
        """Get a template by name."""
        return self.templates.get(name)
