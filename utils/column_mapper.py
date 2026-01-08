"""
Column Mapping Utility for Battery Data
========================================
Provides intelligent column mapping from user CSV/Excel files to standardized schema.

Features:
- Standard column schema definitions
- Auto-detection of common column patterns
- Interactive mapping interface
- Mapping persistence and reuse
- Validation and preview
"""

import json
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class ColumnDefinition:
    """Defines a standard column in the battery data schema."""

    name: str
    description: str
    required: bool
    data_type: str  # 'float', 'int', 'string', 'datetime'
    unit: Optional[str] = None
    aliases: list[str] = field(default_factory=list)
    category: str = (
        "general"  # 'time', 'voltage', 'current', 'capacity', 'temperature', 'cycle', 'general'
    )


class StandardSchema:
    """
    Defines the standard battery data schema.
    All visualizations expect these column names.
    """

    # Core time-series columns
    TIME_SECONDS = ColumnDefinition(
        name="time_s",
        description="Time in seconds from start",
        required=True,
        data_type="float",
        unit="s",
        aliases=[
            "time",
            "time_seconds",
            "test_time",
            "elapsed_time",
            "time(s)",
            "t",
            "timestamp_s",
        ],
        category="time",
    )

    CYCLE_NUMBER = ColumnDefinition(
        name="cycle_number",
        description="Battery cycle number",
        required=True,
        data_type="int",
        unit=None,
        aliases=["cycle", "cycle_index", "cycle_no", "cycle_num", "cycle#", "cyc"],
        category="cycle",
    )

    # Electrical measurements
    VOLTAGE = ColumnDefinition(
        name="voltage",
        description="Cell voltage",
        required=True,
        data_type="float",
        unit="V",
        aliases=[
            "voltage_v",
            "v",
            "volt",
            "cell_voltage",
            "voltage(v)",
            "ecell",
            "vout",
        ],
        category="voltage",
    )

    CURRENT = ColumnDefinition(
        name="current",
        description="Current (positive=charge, negative=discharge)",
        required=True,
        data_type="float",
        unit="A",
        aliases=["current_a", "i", "amp", "current(a)", "i(a)", "curr"],
        category="current",
    )

    # Capacity measurements
    CHARGE_CAPACITY = ColumnDefinition(
        name="charge_capacity_ah",
        description="Charge capacity",
        required=False,
        data_type="float",
        unit="Ah",
        aliases=[
            "charge_capacity",
            "chg_capacity",
            "capacity_charge",
            "q_charge",
            "qc",
            "charge_cap",
        ],
        category="capacity",
    )

    DISCHARGE_CAPACITY = ColumnDefinition(
        name="discharge_capacity_ah",
        description="Discharge capacity",
        required=False,
        data_type="float",
        unit="Ah",
        aliases=[
            "discharge_capacity",
            "dchg_capacity",
            "capacity_discharge",
            "q_discharge",
            "qd",
            "discharge_cap",
        ],
        category="capacity",
    )

    CAPACITY = ColumnDefinition(
        name="capacity_ah",
        description="General capacity (Ah)",
        required=False,
        data_type="float",
        unit="Ah",
        aliases=["capacity", "cap", "q", "ah", "c", "capacity(ah)"],
        category="capacity",
    )

    # Temperature
    TEMPERATURE = ColumnDefinition(
        name="temperature_c",
        description="Cell temperature",
        required=False,
        data_type="float",
        unit="°C",
        aliases=[
            "temperature",
            "temp",
            "temp_c",
            "t_cell",
            "cell_temp",
            "temperature(c)",
            "t(c)",
        ],
        category="temperature",
    )

    # Energy
    ENERGY = ColumnDefinition(
        name="energy_wh",
        description="Energy (Wh)",
        required=False,
        data_type="float",
        unit="Wh",
        aliases=["energy", "e", "wh", "energy(wh)", "e(wh)"],
        category="energy",
    )

    # State of Charge
    SOC = ColumnDefinition(
        name="soc_percent",
        description="State of Charge (%)",
        required=False,
        data_type="float",
        unit="%",
        aliases=["soc", "state_of_charge", "soc(%)", "soc_pct"],
        category="state",
    )

    # State of Health
    SOH = ColumnDefinition(
        name="soh_percent",
        description="State of Health (%)",
        required=False,
        data_type="float",
        unit="%",
        aliases=["soh", "state_of_health", "soh(%)", "soh_pct"],
        category="state",
    )

    # Impedance/EIS data
    FREQUENCY = ColumnDefinition(
        name="frequency_hz",
        description="Frequency for EIS measurements",
        required=False,
        data_type="float",
        unit="Hz",
        aliases=["frequency", "freq", "f", "frequency(hz)", "freq_hz"],
        category="impedance",
    )

    IMPEDANCE_REAL = ColumnDefinition(
        name="z_real_ohm",
        description="Real part of impedance",
        required=False,
        data_type="float",
        unit="Ω",
        aliases=["z_real", "re_z", "z'", "zre", "real_z", "impedance_real"],
        category="impedance",
    )

    IMPEDANCE_IMAG = ColumnDefinition(
        name="z_imag_ohm",
        description="Imaginary part of impedance",
        required=False,
        data_type="float",
        unit="Ω",
        aliases=["z_imag", "im_z", "z''", "zim", "imag_z", "impedance_imag"],
        category="impedance",
    )

    # Battery/Cell identifiers
    BATTERY_ID = ColumnDefinition(
        name="battery_id",
        description="Battery/cell identifier (auto-generated if not provided)",
        required=False,
        data_type="string",
        unit=None,
        aliases=[
            "battery",
            "cell_id",
            "cell",
            "id",
            "battery_name",
            "cell_name",
            "sample_id",
            "device_id",
        ],
        category="identifier",
    )

    @classmethod
    def get_all_columns(cls) -> list[ColumnDefinition]:
        """Returns all defined standard columns."""
        return [
            cls.TIME_SECONDS,
            cls.CYCLE_NUMBER,
            cls.VOLTAGE,
            cls.CURRENT,
            cls.CHARGE_CAPACITY,
            cls.DISCHARGE_CAPACITY,
            cls.CAPACITY,
            cls.TEMPERATURE,
            cls.ENERGY,
            cls.SOC,
            cls.SOH,
            cls.FREQUENCY,
            cls.IMPEDANCE_REAL,
            cls.IMPEDANCE_IMAG,
            cls.BATTERY_ID,
        ]

    @classmethod
    def get_required_columns(cls) -> list[ColumnDefinition]:
        """Returns only required columns."""
        return [col for col in cls.get_all_columns() if col.required]

    @classmethod
    def get_by_category(cls, category: str) -> list[ColumnDefinition]:
        """Returns columns in a specific category."""
        return [col for col in cls.get_all_columns() if col.category == category]


class ColumnMapper:
    """
    Intelligent column mapping utility.
    Maps user CSV/Excel columns to standard schema.
    """

    def __init__(self):
        self.schema = StandardSchema
        self.mapping: dict[str, str] = {}  # user_column -> standard_column
        self.confidence_scores: dict[str, float] = {}  # mapping confidence (0-1)

    def auto_detect_mapping(self, df: pd.DataFrame) -> dict[str, tuple[Optional[str], float]]:
        """
        Auto-detects column mapping using aliases and pattern matching.

        Args:
            df: Input DataFrame with user columns

        Returns:
            Dict mapping standard_column_name -> (detected_user_column, confidence_score)
        """
        detected_mapping = {}
        user_columns_lower = {col: col.lower().strip() for col in df.columns}

        for std_col in self.schema.get_all_columns():
            best_match = None
            best_score = 0.0

            for user_col, user_col_lower in user_columns_lower.items():
                # Exact match with standard name
                if user_col_lower == std_col.name.lower():
                    best_match = user_col
                    best_score = 1.0
                    break

                # Check aliases
                for alias in std_col.aliases:
                    alias_lower = alias.lower()
                    if user_col_lower == alias_lower:
                        if 0.95 > best_score:
                            best_match = user_col
                            best_score = 0.95
                    elif alias_lower in user_col_lower or user_col_lower in alias_lower:
                        # Partial match
                        score = self._calculate_similarity(user_col_lower, alias_lower)
                        if score > best_score and score > 0.6:
                            best_match = user_col
                            best_score = score

            detected_mapping[std_col.name] = (best_match, best_score)

        return detected_mapping

    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity score between two strings (0-1)."""

        # Simple Jaccard similarity on character n-grams
        def get_ngrams(s: str, n: int = 2) -> set[str]:
            return {s[i : i + n] for i in range(len(s) - n + 1)}

        ngrams1 = get_ngrams(str1)
        ngrams2 = get_ngrams(str2)

        if not ngrams1 and not ngrams2:
            return 1.0 if str1 == str2 else 0.0
        if not ngrams1 or not ngrams2:
            return 0.0

        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)

        return intersection / union if union > 0 else 0.0

    def set_mapping(self, user_column: str, standard_column: str):
        """Manually set a column mapping."""
        self.mapping[user_column] = standard_column

    def apply_mapping(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the column mapping to a DataFrame.

        Args:
            df: Input DataFrame with user columns

        Returns:
            DataFrame with columns renamed to standard names
        """
        if not self.mapping:
            return df

        # Create reverse mapping for renaming
        rename_dict = {
            user_col: std_col
            for user_col, std_col in self.mapping.items()
            if user_col in df.columns
        }

        # Rename columns
        df_mapped = df.rename(columns=rename_dict)

        return df_mapped

    def validate_mapping(self, df: pd.DataFrame) -> dict[str, list[str]]:
        """
        Validate that required columns are mapped.

        Args:
            df: DataFrame after mapping

        Returns:
            Dict with 'errors' and 'warnings' lists
        """
        errors = []
        warnings = []

        required_cols = self.schema.get_required_columns()
        for req_col in required_cols:
            if req_col.name not in df.columns:
                errors.append(
                    f"Required column '{req_col.name}' ({req_col.description}) is not mapped"
                )

        # Check data types
        for std_col in self.schema.get_all_columns():
            if std_col.name in df.columns:
                col_data = df[std_col.name]
                if std_col.data_type == "float" and not pd.api.types.is_numeric_dtype(col_data):
                    warnings.append(
                        f"Column '{std_col.name}' should be numeric but appears to be {col_data.dtype}"
                    )
                elif std_col.data_type == "int" and not pd.api.types.is_integer_dtype(col_data):
                    try:
                        # Try to convert
                        df[std_col.name] = pd.to_numeric(df[std_col.name], errors="coerce").astype(
                            "Int64"
                        )
                    except:
                        warnings.append(
                            f"Column '{std_col.name}' should be integer but appears to be {col_data.dtype}"
                        )

        return {"errors": errors, "warnings": warnings}

    def get_mapping_summary(self) -> pd.DataFrame:
        """Get a summary of current mapping as a DataFrame."""
        if not self.mapping:
            return pd.DataFrame(columns=["User Column", "Standard Column", "Description", "Unit"])

        summary_data = []
        std_cols = {col.name: col for col in self.schema.get_all_columns()}

        for user_col, std_col_name in self.mapping.items():
            std_col = std_cols.get(std_col_name)
            if std_col:
                summary_data.append(
                    {
                        "User Column": user_col,
                        "Standard Column": std_col.name,
                        "Description": std_col.description,
                        "Unit": std_col.unit or "N/A",
                        "Required": "Yes" if std_col.required else "No",
                    }
                )

        return pd.DataFrame(summary_data)

    def save_mapping(self, filepath: str):
        """Save mapping to JSON file."""
        mapping_data = {
            "mapping": self.mapping,
            "confidence_scores": self.confidence_scores,
        }
        with open(filepath, "w") as f:
            json.dump(mapping_data, f, indent=2)

    def load_mapping(self, filepath: str):
        """Load mapping from JSON file."""
        with open(filepath) as f:
            mapping_data = json.load(f)
        self.mapping = mapping_data.get("mapping", {})
        self.confidence_scores = mapping_data.get("confidence_scores", {})

    def suggest_unmapped_columns(self, df: pd.DataFrame) -> list[str]:
        """Returns list of columns that haven't been mapped yet."""
        mapped_user_cols = set(self.mapping.keys())
        all_user_cols = set(df.columns)
        return list(all_user_cols - mapped_user_cols)

    def save_to_database(
        self,
        mapping_name: str,
        description: str,
        db_path: str = "nasa_amperedata_full.db",
    ):
        """
        Save mapping to database for reuse.

        Args:
            mapping_name: Unique name for this mapping
            description: Description of the mapping
            db_path: Path to SQLite database
        """
        import json
        import sqlite3
        from datetime import datetime

        mapping_data = {
            "mapping": self.mapping,
            "confidence_scores": self.confidence_scores,
        }
        mapping_json = json.dumps(mapping_data)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                INSERT OR REPLACE INTO column_mappings (mapping_name, description, mapping_json, created_at, use_count)
                VALUES (?, ?, ?, ?, COALESCE((SELECT use_count FROM column_mappings WHERE mapping_name = ?), 0))
            """,
                (
                    mapping_name,
                    description,
                    mapping_json,
                    datetime.now().isoformat(),
                    mapping_name,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    @classmethod
    def load_from_database(
        cls, mapping_name: str, db_path: str = "nasa_amperedata_full.db"
    ) -> "ColumnMapper":
        """
        Load mapping from database.

        Args:
            mapping_name: Name of the mapping to load
            db_path: Path to SQLite database

        Returns:
            ColumnMapper instance with loaded mapping
        """
        import json
        import sqlite3
        from datetime import datetime

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                SELECT mapping_json FROM column_mappings WHERE mapping_name = ?
            """,
                (mapping_name,),
            )
            result = cursor.fetchone()

            if not result:
                raise ValueError(f"Mapping '{mapping_name}' not found in database")

            mapping_data = json.loads(result[0])

            # Update last_used and use_count
            cursor.execute(
                """
                UPDATE column_mappings
                SET last_used = ?, use_count = use_count + 1
                WHERE mapping_name = ?
            """,
                (datetime.now().isoformat(), mapping_name),
            )
            conn.commit()

            # Create mapper and load data
            mapper = cls()
            mapper.mapping = mapping_data.get("mapping", {})
            mapper.confidence_scores = mapping_data.get("confidence_scores", {})

            return mapper
        finally:
            conn.close()

    @staticmethod
    def list_saved_mappings(db_path: str = "nasa_amperedata_full.db") -> pd.DataFrame:
        """
        List all saved mappings in the database.

        Args:
            db_path: Path to SQLite database

        Returns:
            DataFrame with mapping information
        """
        import sqlite3

        conn = sqlite3.connect(db_path)
        try:
            query = """
                SELECT mapping_name, description, created_at, last_used, use_count
                FROM column_mappings
                ORDER BY use_count DESC, created_at DESC
            """
            df = pd.read_sql_query(query, conn)
            return df
        finally:
            conn.close()

    @staticmethod
    def delete_mapping(mapping_name: str, db_path: str = "nasa_amperedata_full.db"):
        """Delete a saved mapping from the database."""
        import sqlite3

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM column_mappings WHERE mapping_name = ?", (mapping_name,))
            conn.commit()
        finally:
            conn.close()


def get_column_preview(df: pd.DataFrame, column: str, num_samples: int = 5) -> dict:
    """
    Get a preview of column data for mapping interface.

    Args:
        df: DataFrame
        column: Column name
        num_samples: Number of sample values to show

    Returns:
        Dict with column statistics and samples
    """
    if column not in df.columns:
        return {}

    col_data = df[column]

    preview = {
        "name": column,
        "dtype": str(col_data.dtype),
        "count": int(col_data.count()),
        "null_count": int(col_data.isnull().sum()),
        "samples": col_data.head(num_samples).tolist(),
    }

    # Add numeric statistics if applicable
    if pd.api.types.is_numeric_dtype(col_data):
        preview.update(
            {
                "min": float(col_data.min()) if not col_data.isna().all() else None,
                "max": float(col_data.max()) if not col_data.isna().all() else None,
                "mean": float(col_data.mean()) if not col_data.isna().all() else None,
            }
        )

    return preview
