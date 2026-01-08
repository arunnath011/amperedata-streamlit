"""Neware .nda/.ndax file parser for battery cycling data.

This module provides functionality to parse Neware .nda and .ndax files from Neware BTS systems,
extracting both metadata and time-series measurement data with comprehensive error handling.

Supported file types:
- .nda (Neware Data Archive) - Binary format from Neware BTS systems
- .ndax (Neware Data Archive Extended) - Extended binary format with additional data
- .csv - Exported CSV format from Neware systems

File format specifications:
- Binary files with channel-based data structure
- Multi-channel support with individual channel metadata
- Comprehensive cycling data including capacity, energy, temperature
- Step-based protocol information
"""

import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd
import structlog
from pydantic import BaseModel, Field

try:
    import NewareNDA

    HAS_NEWARE_NDA = True
except ImportError:
    HAS_NEWARE_NDA = False
    warnings.warn(
        "NewareNDA library not available. Install with: pip install NewareNDA", stacklevel=2
    )

logger = structlog.get_logger(__name__)


class NewareParsingError(Exception):
    """Base exception for Neware file parsing errors."""


class NewareFileNotFoundError(NewareParsingError):
    """Raised when .nda/.ndax file is not found."""


class NewareFormatError(NewareParsingError):
    """Raised when file format is invalid or unsupported."""


class NewareDataValidationError(NewareParsingError):
    """Raised when parsed data fails validation."""


class NewareChannelMetadata(BaseModel):
    """Metadata for a single channel in Neware file."""

    channel_number: int = Field(description="Channel number")
    channel_name: Optional[str] = Field(None, description="Channel name/identifier")
    start_time: Optional[datetime] = Field(None, description="Channel test start time")
    end_time: Optional[datetime] = Field(None, description="Channel test end time")

    # Cell specifications
    active_mass_mg: Optional[float] = Field(None, description="Active mass in mg")
    capacity_mah: Optional[float] = Field(None, description="Nominal capacity in mAh")
    voltage_range: Optional[tuple[float, float]] = Field(
        None, description="Voltage range (min, max)"
    )

    # Test parameters
    temperature_c: Optional[float] = Field(None, description="Test temperature in Celsius")
    current_range_ma: Optional[float] = Field(None, description="Current range in mA")

    # Equipment info
    equipment_id: Optional[str] = Field(None, description="Equipment identifier")
    software_version: Optional[str] = Field(None, description="BTS software version")

    # Statistics
    total_steps: Optional[int] = Field(None, description="Total number of steps")
    total_cycles: Optional[int] = Field(None, description="Total number of cycles")
    total_records: Optional[int] = Field(None, description="Total number of data records")

    # Raw metadata
    raw_metadata: dict[str, Any] = Field(default_factory=dict, description="Raw channel metadata")


class NewareMetadata(BaseModel):
    """Metadata extracted from Neware file."""

    # File information
    file_path: Optional[str] = Field(None, description="Original file path")
    file_format: Optional[str] = Field(None, description="File format (nda, ndax, csv)")
    file_version: Optional[str] = Field(None, description="File format version")
    creation_date: Optional[datetime] = Field(None, description="File creation date")

    # Equipment information
    equipment_model: Optional[str] = Field(None, description="Neware equipment model")
    equipment_serial: Optional[str] = Field(None, description="Equipment serial number")
    software_version: Optional[str] = Field(None, description="BTS software version")

    # Test information
    test_name: Optional[str] = Field(None, description="Test name/identifier")
    operator: Optional[str] = Field(None, description="Test operator")
    comments: Optional[str] = Field(None, description="Test comments")

    # Channel information
    channel_count: int = Field(1, description="Number of channels in file")
    channels: list[NewareChannelMetadata] = Field(
        default_factory=list, description="Channel metadata"
    )

    # Raw file metadata
    raw_metadata: dict[str, Any] = Field(default_factory=dict, description="Raw file metadata")


class NewareData(BaseModel):
    """Parsed data from Neware file."""

    # Required columns based on specifications
    record_id: list[int] = Field(description="Record ID index")
    realtime: list[datetime] = Field(description="Real timestamp for each point")
    time_h: list[float] = Field(description="Test time in hours")
    step_id: list[int] = Field(description="Step ID index")
    cycle_id: list[int] = Field(description="Cycle ID index")

    # Core measurements
    current_ma: list[float] = Field(description="Cell current in mA")
    voltage_v: list[float] = Field(description="Cell voltage in V")

    # Capacity data
    capacitance_chg_mah: list[float] = Field(description="Charge capacity in mAh")
    capacitance_dchg_mah: list[float] = Field(description="Discharge capacity in mAh")

    # Energy data
    engy_chg_mwh: list[float] = Field(description="Charge energy in mWh")
    engy_dchg_mwh: list[float] = Field(description="Discharge energy in mWh")

    # Additional measurements
    dcir_ohm: Optional[list[float]] = Field(None, description="DC internal resistance in Ohms")
    capacity_mah: Optional[list[float]] = Field(None, description="Total capacity in mAh")
    capacity_density_mah_g: Optional[list[float]] = Field(
        None, description="Capacity density in mAh/g"
    )
    energy_mwh: Optional[list[float]] = Field(None, description="Total energy in mWh")
    cmp_eng_mwh_g: Optional[list[float]] = Field(None, description="Energy density in mWh/g")

    # Temperature data
    min_temp_c: Optional[list[float]] = Field(None, description="Minimum temperature in °C")
    max_temp_c: Optional[list[float]] = Field(None, description="Maximum temperature in °C")
    avg_temp_c: Optional[list[float]] = Field(None, description="Average temperature in °C")
    temperature_c: Optional[list[float]] = Field(
        None, description="Temperature (alternate sensor) in °C"
    )

    # Power and differential data
    power_mw: Optional[list[float]] = Field(None, description="Instantaneous power in mW")
    dq_dv_mah_v: Optional[list[float]] = Field(None, description="Differential capacity in mAh/V")
    dqm_dv_mah_v_g: Optional[list[float]] = Field(
        None, description="Differential capacity density in mAh/V·g"
    )

    # Channel information
    channel_number: Optional[list[int]] = Field(None, description="Channel number for each record")

    # Additional data for columns not explicitly defined
    additional_data: dict[str, list[Any]] = Field(
        default_factory=dict, description="Additional columns"
    )

    def to_dataframe(self) -> pd.DataFrame:
        """Convert parsed data to pandas DataFrame."""
        data_dict = {}

        # Add required columns
        data_dict["Record ID"] = self.record_id
        data_dict["Realtime"] = self.realtime
        data_dict["Time(h:min:s.ms)"] = self.time_h
        data_dict["Step ID"] = self.step_id
        data_dict["Cycle ID"] = self.cycle_id
        data_dict["Current(mA)"] = self.current_ma
        data_dict["Voltage(V)"] = self.voltage_v
        data_dict["Capacitance_Chg(mAh)"] = self.capacitance_chg_mah
        data_dict["Capacitance_DChg(mAh)"] = self.capacitance_dchg_mah
        data_dict["Engy_Chg(mWh)"] = self.engy_chg_mwh
        data_dict["Engy_DChg(mWh)"] = self.engy_dchg_mwh

        # Add optional columns if present
        optional_fields = {
            "DCIR(O)": self.dcir_ohm,
            "Capacity(mAh)": self.capacity_mah,
            "Capacity Density(mAh/g)": self.capacity_density_mah_g,
            "Energy(mWh)": self.energy_mwh,
            "CmpEng(mWh/g)": self.cmp_eng_mwh_g,
            "Min-T(C)": self.min_temp_c,
            "Max-T(C)": self.max_temp_c,
            "Avg-T(C)": self.avg_temp_c,
            "Power(mW)": self.power_mw,
            "dQ/dV(mAh/V)": self.dq_dv_mah_v,
            "dQm/dV(mAh/V.g)": self.dqm_dv_mah_v_g,
            "Temperature(C)": self.temperature_c,
            "Channel": self.channel_number,
        }

        for col_name, values in optional_fields.items():
            if values is not None:
                data_dict[col_name] = values

        # Add additional columns
        data_dict.update(self.additional_data)

        return pd.DataFrame(data_dict)


class NewareNDAParser:
    """Parser for Neware .nda/.ndax files with comprehensive error handling."""

    def __init__(self, strict_validation: bool = True):
        """Initialize the parser.

        Args:
            strict_validation: Whether to enforce strict data validation
        """
        if not HAS_NEWARE_NDA:
            raise NewareFormatError(
                "NewareNDA library is required. Install with: pip install NewareNDA"
            )

        self.strict_validation = strict_validation
        self.logger = logger.bind(parser="neware_nda")

        # Column name mappings for different Neware formats/versions
        self.column_mappings = {
            # Standard mappings
            "record id": "Record ID",
            "record_id": "Record ID",
            "realtime": "Realtime",
            "real time": "Realtime",
            "time(h:min:s.ms)": "Time(h:min:s.ms)",
            "time": "Time(h:min:s.ms)",
            "test_time": "Time(h:min:s.ms)",
            "step id": "Step ID",
            "step_id": "Step ID",
            "step": "Step ID",
            "cycle id": "Cycle ID",
            "cycle_id": "Cycle ID",
            "cycle": "Cycle ID",
            "current(ma)": "Current(mA)",
            "current": "Current(mA)",
            "voltage(v)": "Voltage(V)",
            "voltage": "Voltage(V)",
            "capacitance_chg(mah)": "Capacitance_Chg(mAh)",
            "charge_capacity": "Capacitance_Chg(mAh)",
            "capacitance_dchg(mah)": "Capacitance_DChg(mAh)",
            "discharge_capacity": "Capacitance_DChg(mAh)",
            "engy_chg(mwh)": "Engy_Chg(mWh)",
            "charge_energy": "Engy_Chg(mWh)",
            "engy_dchg(mwh)": "Engy_DChg(mWh)",
            "discharge_energy": "Engy_DChg(mWh)",
            "dcir(o)": "DCIR(O)",
            "resistance": "DCIR(O)",
            "capacity(mah)": "Capacity(mAh)",
            "capacity density(mah/g)": "Capacity Density(mAh/g)",
            "energy(mwh)": "Energy(mWh)",
            "cmpeng(mwh/g)": "CmpEng(mWh/g)",
            "min-t(c)": "Min-T(C)",
            "max-t(c)": "Max-T(C)",
            "avg-t(c)": "Avg-T(C)",
            "power(mw)": "Power(mW)",
            "dq/dv(mah/v)": "dQ/dV(mAh/V)",
            "dqm/dv(mah/v.g)": "dQm/dV(mAh/V.g)",
            "temperature(c)": "Temperature(C)",
            "channel": "Channel",
            "channel_number": "Channel",
        }

    def parse_file(self, file_path: Union[str, Path]) -> tuple[NewareData, NewareMetadata]:
        """Parse a Neware .nda/.ndax file.

        Args:
            file_path: Path to the .nda/.ndax file

        Returns:
            Tuple of (parsed_data, metadata)

        Raises:
            NewareFileNotFoundError: If file doesn't exist
            NewareFormatError: If file format is invalid
            NewareDataValidationError: If data validation fails
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise NewareFileNotFoundError(f"File not found: {file_path}")

        # Check file extension
        if file_path.suffix.lower() not in [".nda", ".ndax", ".csv"]:
            warnings.warn(
                f"File does not have expected extension (.nda/.ndax/.csv): {file_path}",
                stacklevel=2,
            )

        self.logger.info("Parsing Neware file", file_path=str(file_path))

        try:
            # Use NewareNDA library to read the file
            if file_path.suffix.lower() == ".csv":
                # Handle CSV files directly with pandas
                df = pd.read_csv(file_path)
                self.logger.info("Loaded CSV file directly", shape=df.shape)
            else:
                # Use NewareNDA for binary files
                # Enable logging to capture metadata
                neware_logger = logging.getLogger("NewareNDA")
                neware_logger.setLevel(logging.INFO)

                # Capture log output to extract metadata
                log_handler = logging.StreamHandler()
                neware_logger.addHandler(log_handler)

                try:
                    df = NewareNDA.read(str(file_path))
                    self.logger.info("Successfully loaded with NewareNDA", shape=df.shape)
                except Exception as e:
                    self.logger.error("NewareNDA failed to read file", error=str(e))
                    raise NewareFormatError(f"Failed to read Neware file: {e}") from e
                finally:
                    neware_logger.removeHandler(log_handler)

            # Extract metadata from DataFrame and file
            metadata = self._extract_metadata(df, file_path)

            # Convert DataFrame to structured data
            data = self._convert_to_structured_data(df, metadata)

            # Validate data if strict mode
            if self.strict_validation:
                self._validate_data_integrity(data, metadata)

            self.logger.info(
                "Successfully parsed Neware file",
                file_path=str(file_path),
                data_rows=len(data.record_id),
                channels=metadata.channel_count,
            )

            return data, metadata

        except Exception as e:
            self.logger.error("Failed to parse Neware file", file_path=str(file_path), error=str(e))
            if isinstance(
                e,
                (NewareFileNotFoundError, NewareFormatError, NewareDataValidationError),
            ):
                raise
            else:
                raise NewareFormatError(f"Failed to parse Neware file: {e}") from e

    def _extract_metadata(self, df: pd.DataFrame, file_path: Path) -> NewareMetadata:
        """Extract metadata from DataFrame and file information."""
        metadata_dict = {
            "file_path": str(file_path),
            "file_format": file_path.suffix.lower().lstrip("."),
        }

        # Safely get creation date
        try:
            if file_path.exists():
                metadata_dict["creation_date"] = datetime.fromtimestamp(file_path.stat().st_mtime)
            else:
                metadata_dict["creation_date"] = None
        except (OSError, AttributeError):
            metadata_dict["creation_date"] = None

        # Analyze DataFrame structure
        channel_count = 1
        channels = []

        # Check if multi-channel data (has Channel column)
        if "Channel" in df.columns:
            channel_numbers = df["Channel"].unique()
            channel_count = len(channel_numbers)

            for ch_num in sorted(channel_numbers):
                ch_data = df[df["Channel"] == ch_num]

                channel_metadata = NewareChannelMetadata(
                    channel_number=int(ch_num),
                    total_records=len(ch_data),
                )

                # Extract channel-specific statistics
                if "Realtime" in ch_data.columns:
                    try:
                        # Try to parse datetime
                        timestamps = pd.to_datetime(ch_data["Realtime"], errors="coerce")
                        valid_timestamps = timestamps.dropna()
                        if len(valid_timestamps) > 0:
                            channel_metadata.start_time = valid_timestamps.min()
                            channel_metadata.end_time = valid_timestamps.max()
                    except Exception:
                        pass

                if "Step ID" in ch_data.columns:
                    channel_metadata.total_steps = ch_data["Step ID"].nunique()

                if "Cycle ID" in ch_data.columns:
                    channel_metadata.total_cycles = ch_data["Cycle ID"].nunique()

                if "Voltage(V)" in ch_data.columns:
                    voltage_min = ch_data["Voltage(V)"].min()
                    voltage_max = ch_data["Voltage(V)"].max()
                    channel_metadata.voltage_range = (voltage_min, voltage_max)

                if "Temperature(C)" in ch_data.columns:
                    temp_mean = ch_data["Temperature(C)"].mean()
                    if pd.notna(temp_mean):
                        channel_metadata.temperature_c = temp_mean

                channels.append(channel_metadata)
        else:
            # Single channel data
            channel_metadata = NewareChannelMetadata(
                channel_number=1,
                total_records=len(df),
            )

            # Extract single channel statistics
            if "Realtime" in df.columns:
                try:
                    timestamps = pd.to_datetime(df["Realtime"], errors="coerce")
                    valid_timestamps = timestamps.dropna()
                    if len(valid_timestamps) > 0:
                        channel_metadata.start_time = valid_timestamps.min()
                        channel_metadata.end_time = valid_timestamps.max()
                except Exception:
                    pass

            if "Step ID" in df.columns:
                channel_metadata.total_steps = df["Step ID"].nunique()

            if "Cycle ID" in df.columns:
                channel_metadata.total_cycles = df["Cycle ID"].nunique()

            if "Voltage(V)" in df.columns:
                voltage_min = df["Voltage(V)"].min()
                voltage_max = df["Voltage(V)"].max()
                channel_metadata.voltage_range = (voltage_min, voltage_max)

            channels.append(channel_metadata)

        metadata_dict["channel_count"] = channel_count
        metadata_dict["channels"] = channels

        return NewareMetadata(**metadata_dict)

    def _convert_to_structured_data(self, df: pd.DataFrame, metadata: NewareMetadata) -> NewareData:
        """Convert DataFrame to structured NewareData."""
        # Normalize column names
        df_normalized = df.copy()
        df_normalized.columns = [self.column_mappings.get(col.lower(), col) for col in df.columns]

        # Extract required fields
        required_fields = {}

        # Record ID
        if "Record ID" in df_normalized.columns:
            required_fields["record_id"] = df_normalized["Record ID"].astype(int).tolist()
        else:
            required_fields["record_id"] = list(range(len(df_normalized)))

        # Realtime
        if "Realtime" in df_normalized.columns:
            try:
                timestamps = pd.to_datetime(df_normalized["Realtime"], errors="coerce")
                # Fill NaT values with a default timestamp
                default_time = datetime.now()
                timestamps = timestamps.fillna(default_time)
                required_fields["realtime"] = timestamps.tolist()
            except Exception:
                # Fallback to default timestamps
                default_time = datetime.now()
                required_fields["realtime"] = [default_time] * len(df_normalized)
        else:
            default_time = datetime.now()
            required_fields["realtime"] = [default_time] * len(df_normalized)

        # Time in hours
        if "Time(h:min:s.ms)" in df_normalized.columns:
            required_fields["time_h"] = df_normalized["Time(h:min:s.ms)"].astype(float).tolist()
        else:
            required_fields["time_h"] = [0.0] * len(df_normalized)

        # Step ID
        if "Step ID" in df_normalized.columns:
            required_fields["step_id"] = df_normalized["Step ID"].astype(int).tolist()
        else:
            required_fields["step_id"] = [1] * len(df_normalized)

        # Cycle ID
        if "Cycle ID" in df_normalized.columns:
            required_fields["cycle_id"] = df_normalized["Cycle ID"].astype(int).tolist()
        else:
            required_fields["cycle_id"] = [1] * len(df_normalized)

        # Current and Voltage
        if "Current(mA)" in df_normalized.columns:
            required_fields["current_ma"] = df_normalized["Current(mA)"].astype(float).tolist()
        else:
            required_fields["current_ma"] = [0.0] * len(df_normalized)

        if "Voltage(V)" in df_normalized.columns:
            required_fields["voltage_v"] = df_normalized["Voltage(V)"].astype(float).tolist()
        else:
            required_fields["voltage_v"] = [0.0] * len(df_normalized)

        # Capacity
        if "Capacitance_Chg(mAh)" in df_normalized.columns:
            required_fields["capacitance_chg_mah"] = (
                df_normalized["Capacitance_Chg(mAh)"].astype(float).tolist()
            )
        else:
            required_fields["capacitance_chg_mah"] = [0.0] * len(df_normalized)

        if "Capacitance_DChg(mAh)" in df_normalized.columns:
            required_fields["capacitance_dchg_mah"] = (
                df_normalized["Capacitance_DChg(mAh)"].astype(float).tolist()
            )
        else:
            required_fields["capacitance_dchg_mah"] = [0.0] * len(df_normalized)

        # Energy
        if "Engy_Chg(mWh)" in df_normalized.columns:
            required_fields["engy_chg_mwh"] = df_normalized["Engy_Chg(mWh)"].astype(float).tolist()
        else:
            required_fields["engy_chg_mwh"] = [0.0] * len(df_normalized)

        if "Engy_DChg(mWh)" in df_normalized.columns:
            required_fields["engy_dchg_mwh"] = (
                df_normalized["Engy_DChg(mWh)"].astype(float).tolist()
            )
        else:
            required_fields["engy_dchg_mwh"] = [0.0] * len(df_normalized)

        # Optional fields
        optional_fields = {}
        optional_columns = {
            "dcir_ohm": "DCIR(O)",
            "capacity_mah": "Capacity(mAh)",
            "capacity_density_mah_g": "Capacity Density(mAh/g)",
            "energy_mwh": "Energy(mWh)",
            "cmp_eng_mwh_g": "CmpEng(mWh/g)",
            "min_temp_c": "Min-T(C)",
            "max_temp_c": "Max-T(C)",
            "avg_temp_c": "Avg-T(C)",
            "power_mw": "Power(mW)",
            "dq_dv_mah_v": "dQ/dV(mAh/V)",
            "dqm_dv_mah_v_g": "dQm/dV(mAh/V.g)",
            "temperature_c": "Temperature(C)",
            "channel_number": "Channel",
        }

        for field_name, col_name in optional_columns.items():
            if col_name in df_normalized.columns:
                values = df_normalized[col_name].astype(float).tolist()
                optional_fields[field_name] = values

        # Additional data for unrecognized columns
        additional_data = {}
        recognized_columns = set(self.column_mappings.values()) | {
            "Record ID",
            "Realtime",
            "Time(h:min:s.ms)",
            "Step ID",
            "Cycle ID",
            "Current(mA)",
            "Voltage(V)",
            "Capacitance_Chg(mAh)",
            "Capacitance_DChg(mAh)",
            "Engy_Chg(mWh)",
            "Engy_DChg(mWh)",
        }

        for col in df_normalized.columns:
            if col not in recognized_columns:
                try:
                    additional_data[col] = df_normalized[col].tolist()
                except Exception:
                    # Skip columns that can't be converted
                    pass

        return NewareData(**required_fields, **optional_fields, additional_data=additional_data)

    def _validate_data_integrity(self, data: NewareData, metadata: NewareMetadata) -> bool:
        """Validate data integrity and consistency.

        Args:
            data: Parsed data to validate
            metadata: Associated metadata

        Returns:
            True if data passes validation

        Raises:
            NewareDataValidationError: If validation fails
        """
        # Check data length consistency
        required_lengths = [
            len(data.record_id),
            len(data.realtime),
            len(data.time_h),
            len(data.step_id),
            len(data.cycle_id),
            len(data.current_ma),
            len(data.voltage_v),
            len(data.capacitance_chg_mah),
            len(data.capacitance_dchg_mah),
            len(data.engy_chg_mwh),
            len(data.engy_dchg_mwh),
        ]

        if len(set(required_lengths)) > 1:
            raise NewareDataValidationError(
                f"Inconsistent data lengths in required fields: {dict(zip(['record_id', 'realtime', 'time_h', 'step_id', 'cycle_id', 'current_ma', 'voltage_v', 'capacitance_chg_mah', 'capacitance_dchg_mah', 'engy_chg_mwh', 'engy_dchg_mwh'], required_lengths))}"
            )

        # Validate data ranges
        if any(x < 0 for x in data.step_id):
            self.logger.warning("Negative step IDs found")

        if any(x < 0 for x in data.cycle_id):
            self.logger.warning("Negative cycle IDs found")

        # Check for reasonable voltage ranges (adjust based on application)
        if data.voltage_v:  # Only check if data exists
            voltage_range = (min(data.voltage_v), max(data.voltage_v))
            if voltage_range[0] < -10 or voltage_range[1] > 10:
                self.logger.warning("Unusual voltage range detected", voltage_range=voltage_range)

        # Check for reasonable capacity values
        if any(x < 0 for x in data.capacitance_chg_mah + data.capacitance_dchg_mah):
            self.logger.warning("Negative capacity values found")

        self.logger.info("Data validation passed", data_points=required_lengths[0])
        return True
