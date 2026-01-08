"""BioLogic .mpt file parser for electrochemical data.

This module provides functionality to parse BioLogic .mpt files from EC-Lab software,
extracting both metadata and time-series measurement data with comprehensive error handling.

Supported experiment types:
- CV (Cyclic Voltammetry)
- EIS (Electrochemical Impedance Spectroscopy)
- GITT (Galvanostatic Intermittent Titration Technique)
- Galvanostatic (constant current)
- PITT (Potentiostatic Intermittent Titration Technique)
- OCV (Open Circuit Voltage)

File format specifications:
- ASCII text files with .mpt extension
- Header section with metadata and settings
- Data section with tab-separated columns
- Matching .mpl log files (optional)
"""

import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import structlog
from pydantic import BaseModel, Field, field_validator

logger = structlog.get_logger(__name__)


class BiologicParsingError(Exception):
    """Base exception for BioLogic file parsing errors."""


class BiologicFileNotFoundError(BiologicParsingError):
    """Raised when .mpt file is not found."""


class BiologicFormatError(BiologicParsingError):
    """Raised when .mpt file format is invalid or corrupted."""


class BiologicDataValidationError(BiologicParsingError):
    """Raised when parsed data fails validation."""


class BiologicMetadata(BaseModel):
    """Metadata extracted from BioLogic .mpt file header."""

    # File information
    file_version: Optional[str] = Field(None, description="EC-Lab file format version")
    acquisition_date: Optional[datetime] = Field(None, description="Date of data acquisition")

    # Instrument settings
    instrument_model: Optional[str] = Field(None, description="BioLogic instrument model")
    instrument_serial: Optional[str] = Field(None, description="Instrument serial number")
    channel_number: Optional[int] = Field(None, description="Channel number used")

    # Experiment parameters
    experiment_type: Optional[str] = Field(None, description="Type of electrochemical experiment")
    technique_name: Optional[str] = Field(None, description="EC-Lab technique name")
    comments: Optional[str] = Field(None, description="User comments")

    # Cell configuration
    electrode_material: Optional[str] = Field(None, description="Electrode material")
    electrolyte: Optional[str] = Field(None, description="Electrolyte composition")
    reference_electrode: Optional[str] = Field(None, description="Reference electrode type")
    temperature: Optional[float] = Field(None, description="Temperature in Celsius")

    # Measurement parameters
    voltage_range: Optional[Tuple[float, float]] = Field(
        None, description="Voltage range (min, max) in V"
    )
    current_range: Optional[float] = Field(None, description="Current range in A")
    scan_rate: Optional[float] = Field(None, description="Scan rate in V/s for CV")
    frequency_range: Optional[Tuple[float, float]] = Field(
        None, description="Frequency range for EIS"
    )

    # Raw header data
    raw_header: Dict[str, Any] = Field(
        default_factory=dict, description="Raw header key-value pairs"
    )
    column_names: List[str] = Field(default_factory=list, description="Data column names")
    column_units: List[str] = Field(default_factory=list, description="Data column units")

    @field_validator("acquisition_date", mode="before")
    @classmethod
    def parse_date(cls, v: Any) -> Optional[datetime]:
        """Parse acquisition date from various formats."""
        if v is None or v == "":
            return None
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            # Try common date formats used by EC-Lab
            date_formats = [
                "%d/%m/%Y %H:%M:%S",
                "%m/%d/%Y %H:%M:%S",
                "%Y-%m-%d %H:%M:%S",
                "%d/%m/%Y",
                "%m/%d/%Y",
                "%Y-%m-%d",
            ]
            for fmt in date_formats:
                try:
                    return datetime.strptime(v, fmt)
                except ValueError:
                    continue
        return None


class BiologicData(BaseModel):
    """Parsed data from BioLogic .mpt file."""

    # Required columns based on your specifications
    cycle_number: List[int] = Field(description="Cycle index")
    half_cycle: List[int] = Field(description="Half cycle index")
    voltage_v: List[float] = Field(description="Cell potential in Volts")
    current_ma: List[float] = Field(description="Cell current in mA")

    # Capacity and energy data
    q_discharge_mah: List[float] = Field(description="Discharge capacity in mAh")
    q_charge_mah: List[float] = Field(description="Charge capacity in mAh")
    energy_charge_wh: List[float] = Field(description="Charge energy in Wh")
    energy_discharge_wh: List[float] = Field(description="Discharge energy in Wh")

    # Time data
    time_s: Optional[List[float]] = Field(None, description="Time in seconds")

    # Additional common columns
    step_number: Optional[List[int]] = Field(None, description="Step number in protocol")
    temperature_c: Optional[List[float]] = Field(None, description="Temperature in Celsius")

    # Raw data for additional columns
    additional_data: Dict[str, List[Any]] = Field(
        default_factory=dict, description="Additional columns"
    )

    def to_dataframe(self) -> pd.DataFrame:
        """Convert parsed data to pandas DataFrame."""
        data_dict = {}

        # Add required columns
        data_dict["cycle_number"] = self.cycle_number
        data_dict["half_cycle"] = self.half_cycle
        data_dict["Ecell/V"] = self.voltage_v
        data_dict["I/mA"] = self.current_ma
        data_dict["Q discharge/mA.h"] = self.q_discharge_mah
        data_dict["Q charge/mA.h"] = self.q_charge_mah
        data_dict["Energy charge/W.h"] = self.energy_charge_wh
        data_dict["Energy discharge/W.h"] = self.energy_discharge_wh

        # Add optional columns if present
        if self.time_s is not None:
            data_dict["time/s"] = self.time_s
        if self.step_number is not None:
            data_dict["step_number"] = self.step_number
        if self.temperature_c is not None:
            data_dict["temperature/°C"] = self.temperature_c

        # Add additional columns
        data_dict.update(self.additional_data)

        return pd.DataFrame(data_dict)


class BiologicMPTParser:
    """Parser for BioLogic .mpt files with comprehensive error handling."""

    def __init__(self, encoding: str = "utf-8", strict_validation: bool = True):
        """Initialize the parser.

        Args:
            encoding: File encoding to use when reading .mpt files
            strict_validation: Whether to enforce strict data validation
        """
        self.encoding = encoding
        self.strict_validation = strict_validation
        self.logger = logger.bind(parser="biologic_mpt")

        # Common column name mappings (EC-Lab can use various names)
        self.column_mappings = {
            # Voltage columns
            "ewe/v": "Ecell/V",
            "ecell/v": "Ecell/V",
            "voltage/v": "Ecell/V",
            "e/v": "Ecell/V",
            "control/v": "control_voltage",
            # Current columns
            "i/ma": "I/mA",
            "current/ma": "I/mA",
            "<i>/ma": "I/mA",
            # Time columns
            "time/s": "time/s",
            "t/s": "time/s",
            # Capacity columns
            "q discharge/ma.h": "Q discharge/mA.h",
            "q charge/ma.h": "Q charge/mA.h",
            "capacity/mah": "Q discharge/mA.h",
            "(q-qo)/c": "charge_capacity",
            # Energy columns
            "energy charge/w.h": "Energy charge/W.h",
            "energy discharge/w.h": "Energy discharge/W.h",
            "p/w": "power_w",
            # Cycle columns
            "cycle number": "cycle number",
            "half cycle": "half cycle",
            "ns": "step_number",
            # Additional BioLogic columns
            "mode": "mode",
            "ox/red": "ox_red",
            "error": "error_flag",
            "control changes": "control_changes",
            "counter inc.": "counter_inc",
        }

    def parse_file(self, file_path: Union[str, Path]) -> Tuple[BiologicData, BiologicMetadata]:
        """Parse a BioLogic .mpt file.

        Args:
            file_path: Path to the .mpt file

        Returns:
            Tuple of (parsed_data, metadata)

        Raises:
            BiologicFileNotFoundError: If file doesn't exist
            BiologicFormatError: If file format is invalid
            BiologicDataValidationError: If data validation fails
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise BiologicFileNotFoundError(f"File not found: {file_path}")

        if not file_path.suffix.lower() == ".mpt":
            warnings.warn(f"File does not have .mpt extension: {file_path}")

        self.logger.info("Parsing BioLogic .mpt file", file_path=str(file_path))

        try:
            with open(file_path, encoding=self.encoding) as file:
                content = file.read()

            # Parse header and data sections
            header_lines, data_lines = self._split_header_data(content)

            # Parse metadata from header
            metadata = self._parse_header(header_lines, file_path)

            # Parse data section
            data = self._parse_data(data_lines, metadata)

            self.logger.info(
                "Successfully parsed .mpt file",
                file_path=str(file_path),
                data_rows=len(data.cycle_number),
                columns=len(metadata.column_names),
            )

            return data, metadata

        except UnicodeDecodeError as e:
            # Try different encodings
            for alt_encoding in ["latin1", "cp1252", "iso-8859-1"]:
                try:
                    with open(file_path, encoding=alt_encoding) as file:
                        content = file.read()
                    self.logger.warning(
                        "Used alternative encoding",
                        file_path=str(file_path),
                        encoding=alt_encoding,
                    )

                    # Parse header and data sections with alternative encoding
                    header_lines, data_lines = self._split_header_data(content)

                    # Parse metadata from header
                    metadata = self._parse_header(header_lines, file_path)

                    # Parse data section
                    data = self._parse_data(data_lines, metadata)

                    self.logger.info(
                        "Successfully parsed .mpt file",
                        file_path=str(file_path),
                        data_rows=len(data.cycle_number),
                        columns=len(metadata.column_names),
                    )

                    return data, metadata

                except UnicodeDecodeError:
                    continue
                except Exception as parse_error:
                    self.logger.error(
                        "Failed to parse with alternative encoding",
                        encoding=alt_encoding,
                        error=str(parse_error),
                    )
                    continue
            else:
                raise BiologicFormatError(f"Could not decode file with any encoding: {e}")

        except Exception as e:
            self.logger.error("Failed to parse .mpt file", file_path=str(file_path), error=str(e))
            raise BiologicFormatError(f"Failed to parse .mpt file: {e}") from e

    def _split_header_data(self, content: str) -> Tuple[List[str], List[str]]:
        """Split file content into header and data sections."""
        lines = content.split("\n")

        # Find the data start marker (usually "Nb header lines")
        data_start_idx = 0
        for i, line in enumerate(lines):
            if line.strip().lower().startswith("nb header lines"):
                try:
                    # Extract number of header lines
                    nb_header = int(re.search(r"\d+", line).group())
                    data_start_idx = nb_header
                    break
                except (AttributeError, ValueError):
                    # Fallback: look for column headers
                    pass

        # If no header line count found, look for typical data column headers
        if data_start_idx == 0:
            for i, line in enumerate(lines):
                if any(col in line.lower() for col in ["time/s", "ewe/v", "i/ma", "cycle"]):
                    data_start_idx = i
                    break

        if data_start_idx == 0:
            raise BiologicFormatError("Could not identify data section start")

        # The column header is typically the last line of the header section
        # So we include it in the data section for easier parsing
        header_lines = lines[: data_start_idx - 1]  # Exclude the column header line
        data_lines = lines[data_start_idx - 1 :]  # Include the column header line

        return header_lines, data_lines

    def _parse_header(self, header_lines: List[str], file_path: Path) -> BiologicMetadata:
        """Parse metadata from header lines."""
        raw_header = {}

        for line in header_lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Parse key-value pairs
            if ":" in line:
                key, value = line.split(":", 1)
                raw_header[key.strip()] = value.strip()
            elif "=" in line:
                key, value = line.split("=", 1)
                raw_header[key.strip()] = value.strip()

        # Extract specific metadata fields
        metadata_dict = {
            "raw_header": raw_header,
            "file_version": raw_header.get("EC-Lab ASCII file"),
            "acquisition_date": raw_header.get("Created")
            or raw_header.get("Acquisition started on"),
            "instrument_model": raw_header.get("Instrument"),
            "technique_name": raw_header.get("Technique"),
            "comments": raw_header.get("Comments"),
        }

        # Parse numeric fields with error handling
        try:
            if "Channel number" in raw_header:
                metadata_dict["channel_number"] = int(raw_header["Channel number"])
        except (ValueError, TypeError):
            pass

        try:
            if "Temperature" in raw_header:
                temp_str = raw_header["Temperature"].replace("°C", "").strip()
                metadata_dict["temperature"] = float(temp_str)
        except (ValueError, TypeError):
            pass

        return BiologicMetadata(**metadata_dict)

    def _parse_data(self, data_lines: List[str], metadata: BiologicMetadata) -> BiologicData:
        """Parse data section into structured format."""
        if not data_lines:
            raise BiologicFormatError("No data section found")

        # Find column header line - look for specific column names that indicate headers
        header_line_idx = 0
        for i, line in enumerate(data_lines):
            line = line.strip()
            if line and not line.startswith("#"):
                # Check if this looks like a header line (contains column names, not numeric data)
                if any(
                    col in line.lower()
                    for col in ["time/s", "ewe/v", "mode", "<i>/ma", "cycle number"]
                ):
                    # Additional check: make sure it's not a data line by checking if it contains mostly numbers
                    parts = line.split("\t")
                    non_numeric_count = 0
                    for part in parts:
                        try:
                            float(part.replace("E+", "E").replace("E-", "E"))
                        except ValueError:
                            non_numeric_count += 1

                    # If more than half the parts are non-numeric, it's likely a header
                    if non_numeric_count > len(parts) / 2:
                        header_line_idx = i
                        break

        if header_line_idx >= len(data_lines):
            raise BiologicFormatError("Could not find data column headers")

        # Parse column headers
        header_line = data_lines[header_line_idx].strip()
        raw_columns = [col.strip() for col in header_line.split("\t")]

        # Normalize column names using mappings
        normalized_columns = []
        for col in raw_columns:
            normalized = self.column_mappings.get(col.lower(), col)
            normalized_columns.append(normalized)

        metadata.column_names = normalized_columns

        # Parse data rows
        data_rows = []
        for line in data_lines[header_line_idx + 1 :]:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            values = line.split("\t")
            if len(values) != len(normalized_columns):
                # Skip incomplete rows but warn
                self.logger.warning(
                    "Skipping incomplete data row",
                    expected_columns=len(normalized_columns),
                    actual_values=len(values),
                )
                continue

            data_rows.append(values)

        if not data_rows:
            raise BiologicFormatError("No valid data rows found")

        # Convert to structured data
        return self._convert_to_structured_data(data_rows, normalized_columns)

    def _convert_to_structured_data(
        self, data_rows: List[List[str]], columns: List[str]
    ) -> BiologicData:
        """Convert raw data rows to structured BiologicData."""
        # Initialize data containers
        data_dict = {}
        for col in columns:
            data_dict[col] = []

        # Process each row
        for row in data_rows:
            for i, (col, value) in enumerate(zip(columns, row)):
                try:
                    # Convert based on column type
                    if col in ["cycle number", "half cycle", "step_number"]:
                        converted_value = int(float(value)) if value.strip() else 0
                    elif col in [
                        "Ecell/V",
                        "I/mA",
                        "time/s",
                        "Q discharge/mA.h",
                        "Q charge/mA.h",
                        "Energy charge/W.h",
                        "Energy discharge/W.h",
                    ]:
                        converted_value = float(value) if value.strip() else 0.0
                    else:
                        # Try float first, then keep as string
                        try:
                            converted_value = float(value)
                        except ValueError:
                            converted_value = value

                    data_dict[col].append(converted_value)

                except (ValueError, TypeError) as e:
                    if self.strict_validation:
                        raise BiologicDataValidationError(
                            f"Failed to convert value '{value}' in column '{col}': {e}"
                        )
                    else:
                        # Use default value and warn
                        if col in ["cycle number", "half cycle", "step_number"]:
                            data_dict[col].append(0)
                        else:
                            data_dict[col].append(0.0)
                        self.logger.warning(
                            "Using default value for invalid data",
                            column=col,
                            value=value,
                            error=str(e),
                        )

        # Build BiologicData with required fields
        required_fields = {
            "cycle_number": data_dict.get("cycle number", [0] * len(data_rows)),
            "half_cycle": data_dict.get("half cycle", [0] * len(data_rows)),
            "voltage_v": data_dict.get("Ecell/V", [0.0] * len(data_rows)),
            "current_ma": data_dict.get("I/mA", [0.0] * len(data_rows)),
            "q_discharge_mah": data_dict.get("Q discharge/mA.h", [0.0] * len(data_rows)),
            "q_charge_mah": data_dict.get("Q charge/mA.h", [0.0] * len(data_rows)),
            "energy_charge_wh": data_dict.get("Energy charge/W.h", [0.0] * len(data_rows)),
            "energy_discharge_wh": data_dict.get("Energy discharge/W.h", [0.0] * len(data_rows)),
        }

        # Add optional fields
        optional_fields = {}
        if "time/s" in data_dict:
            optional_fields["time_s"] = data_dict["time/s"]
        if "step_number" in data_dict:
            optional_fields["step_number"] = data_dict["step_number"]

        # Add additional columns
        additional_data = {}
        for col, values in data_dict.items():
            if col not in [
                "cycle number",
                "half cycle",
                "Ecell/V",
                "I/mA",
                "Q discharge/mA.h",
                "Q charge/mA.h",
                "Energy charge/W.h",
                "Energy discharge/W.h",
                "time/s",
                "step_number",
            ]:
                additional_data[col] = values

        return BiologicData(**required_fields, **optional_fields, additional_data=additional_data)

    def validate_data_integrity(self, data: BiologicData, metadata: BiologicMetadata) -> bool:
        """Validate data integrity and consistency.

        Args:
            data: Parsed data to validate
            metadata: Associated metadata

        Returns:
            True if data passes validation

        Raises:
            BiologicDataValidationError: If validation fails
        """
        # Check data length consistency
        lengths = [
            len(data.cycle_number),
            len(data.half_cycle),
            len(data.voltage_v),
            len(data.current_ma),
            len(data.q_discharge_mah),
            len(data.q_charge_mah),
            len(data.energy_charge_wh),
            len(data.energy_discharge_wh),
        ]

        if len(set(lengths)) > 1:
            raise BiologicDataValidationError(
                f"Inconsistent data lengths: {dict(zip(['cycle_number', 'half_cycle', 'voltage_v', 'current_ma', 'q_discharge_mah', 'q_charge_mah', 'energy_charge_wh', 'energy_discharge_wh'], lengths))}"
            )

        # Validate data ranges
        if any(v < 0 for v in data.cycle_number):
            raise BiologicDataValidationError("Negative cycle numbers found")

        # Check for reasonable voltage ranges (adjust based on your application)
        voltage_range = (min(data.voltage_v), max(data.voltage_v))
        if voltage_range[0] < -10 or voltage_range[1] > 10:
            self.logger.warning("Unusual voltage range detected", voltage_range=voltage_range)

        self.logger.info("Data validation passed", data_points=lengths[0])
        return True
