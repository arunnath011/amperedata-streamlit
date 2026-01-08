"""
NASA Battery Data Loader
========================
Loads NASA battery testing data from CSV files into SQLite database.
"""

import sqlite3
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def create_tables(conn):
    """Create database tables if they don't exist."""
    cursor = conn.cursor()

    # Batteries table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS batteries (
            battery_id TEXT PRIMARY KEY,
            total_cycles INTEGER,
            initial_capacity REAL,
            final_capacity REAL
        )
    """
    )

    # Column mappings table (for upload feature)
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS column_mappings (
            mapping_name TEXT PRIMARY KEY,
            description TEXT,
            mapping_json TEXT,
            created_at TEXT,
            last_used TEXT,
            use_count INTEGER DEFAULT 0
        )
    """
    )

    # Battery metadata table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS battery_metadata (
            battery_id TEXT PRIMARY KEY,
            serial_number TEXT,
            status TEXT,
            position_in_build TEXT,
            assembly_date TEXT,
            build_id TEXT,
            FOREIGN KEY (battery_id) REFERENCES batteries(battery_id)
        )
    """
    )

    # Build sheets table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS build_sheets (
            build_id TEXT PRIMARY KEY,
            build_name TEXT,
            build_type TEXT,
            cathode_material TEXT,
            anode_material TEXT,
            electrolyte TEXT,
            nominal_capacity_ah REAL,
            nominal_voltage_v REAL,
            form_factor TEXT,
            manufacturer TEXT,
            created_at TEXT
        )
    """
    )

    # Capacity fade table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS capacity_fade (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            battery_id TEXT NOT NULL,
            cycle_number INTEGER,
            capacity_ah REAL,
            retention_percent REAL,
            timestamp TEXT,
            FOREIGN KEY (battery_id) REFERENCES batteries(battery_id)
        )
    """
    )

    # Resistance data table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS resistance_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            battery_id TEXT NOT NULL,
            re_ohms REAL,
            rct_ohms REAL,
            timestamp TEXT,
            cycle_number INTEGER,
            FOREIGN KEY (battery_id) REFERENCES batteries(battery_id)
        )
    """
    )

    # Cycles table (raw data)
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS cycles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            battery_id TEXT NOT NULL,
            test_id TEXT,
            time_seconds REAL,
            voltage REAL,
            current REAL,
            capacity_ah REAL,
            temperature REAL,
            FOREIGN KEY (battery_id) REFERENCES batteries(battery_id)
        )
    """
    )

    # Create indexes for faster queries
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_capacity_battery ON capacity_fade(battery_id)")
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_resistance_battery ON resistance_data(battery_id)"
    )
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_cycles_battery ON cycles(battery_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_cycles_test ON cycles(battery_id, test_id)")

    conn.commit()


def load_nasa_data(data_dir: Path, db_path: Path):
    """Load NASA battery data from CSV files into SQLite database."""

    metadata_path = data_dir / "metadata.csv"
    data_folder = data_dir / "data"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    if not data_folder.exists():
        raise FileNotFoundError(f"Data folder not found: {data_folder}")

    # Read metadata
    print("Reading metadata...")
    metadata = pd.read_csv(metadata_path)

    # Convert numeric columns
    metadata["Capacity"] = pd.to_numeric(metadata["Capacity"], errors="coerce")
    metadata["Re"] = pd.to_numeric(metadata["Re"], errors="coerce")
    metadata["Rct"] = pd.to_numeric(metadata["Rct"], errors="coerce")
    metadata["test_id"] = pd.to_numeric(metadata["test_id"], errors="coerce")

    print(f"Found {len(metadata)} records in metadata")

    # Connect to database
    conn = sqlite3.connect(str(db_path))
    create_tables(conn)
    cursor = conn.cursor()

    # Get unique batteries
    batteries = metadata["battery_id"].unique()
    print(f"Found {len(batteries)} unique batteries: {list(batteries)}")

    # Process each battery
    for battery_id in batteries:
        print(f"\nProcessing battery: {battery_id}")

        battery_meta = metadata[metadata["battery_id"] == battery_id]

        # Get discharge records for capacity fade
        discharge_records = battery_meta[battery_meta["type"] == "discharge"].copy()

        if not discharge_records.empty:
            # Extract capacity data
            initial_capacity = (
                discharge_records["Capacity"].iloc[0]
                if pd.notna(discharge_records["Capacity"].iloc[0])
                else None
            )
            final_capacity = (
                discharge_records["Capacity"].iloc[-1]
                if pd.notna(discharge_records["Capacity"].iloc[-1])
                else None
            )

            # Insert battery record
            cursor.execute(
                """
                INSERT OR REPLACE INTO batteries (battery_id, total_cycles, initial_capacity, final_capacity)
                VALUES (?, ?, ?, ?)
            """,
                (battery_id, len(discharge_records), initial_capacity, final_capacity),
            )

            # Insert capacity fade data
            cycle_num = 0
            for idx, row in discharge_records.iterrows():
                if pd.notna(row.get("Capacity")):
                    retention = (
                        (row["Capacity"] / initial_capacity * 100) if initial_capacity else None
                    )
                    cursor.execute(
                        """
                        INSERT INTO capacity_fade (battery_id, cycle_number, capacity_ah, retention_percent, timestamp)
                        VALUES (?, ?, ?, ?, ?)
                    """,
                        (
                            battery_id,
                            cycle_num,
                            row["Capacity"],
                            retention,
                            str(row.get("start_time", "")),
                        ),
                    )
                    cycle_num += 1

        # Get impedance records for resistance data
        impedance_records = battery_meta[battery_meta["type"] == "impedance"]

        if not impedance_records.empty:
            for idx, row in impedance_records.iterrows():
                if pd.notna(row.get("Re")) or pd.notna(row.get("Rct")):
                    cursor.execute(
                        """
                        INSERT INTO resistance_data (battery_id, re_ohms, rct_ohms, timestamp, cycle_number)
                        VALUES (?, ?, ?, ?, ?)
                    """,
                        (
                            battery_id,
                            row.get("Re"),
                            row.get("Rct"),
                            str(row.get("start_time", "")),
                            row.get("test_id"),
                        ),
                    )

        # Load cycle data from CSV files (sample - first 100 cycles per battery to avoid huge database)
        print(f"  Loading cycle data for {battery_id}...")
        cycle_count = 0
        max_cycles = 100  # Limit cycles to keep database manageable

        for idx, row in tqdm(
            battery_meta.iterrows(), total=len(battery_meta), desc=f"  {battery_id}"
        ):
            if cycle_count >= max_cycles:
                break

            filename = row.get("filename")
            if pd.isna(filename):
                continue

            csv_path = data_folder / filename
            if csv_path.exists():
                try:
                    cycle_df = pd.read_csv(csv_path)

                    # Check for expected columns
                    voltage_col = None
                    current_col = None
                    time_col = None

                    for col in cycle_df.columns:
                        col_lower = col.lower()
                        if "voltage" in col_lower:
                            voltage_col = col
                        elif "current" in col_lower:
                            current_col = col
                        elif "time" in col_lower:
                            time_col = col

                    # Insert cycle data (sample every 10th point for space efficiency)
                    sample_rate = max(1, len(cycle_df) // 100)
                    sampled_df = cycle_df.iloc[::sample_rate]

                    for _, cycle_row in sampled_df.iterrows():
                        cursor.execute(
                            """
                            INSERT INTO cycles (battery_id, test_id, time_seconds, voltage, current, capacity_ah, temperature)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                            (
                                battery_id,
                                str(row.get("test_id", "")),
                                cycle_row.get(time_col) if time_col else None,
                                cycle_row.get(voltage_col) if voltage_col else None,
                                cycle_row.get(current_col) if current_col else None,
                                (
                                    cycle_row.get("Capacity_Ah")
                                    if "Capacity_Ah" in cycle_df.columns
                                    else None
                                ),
                                (
                                    cycle_row.get("Temperature_measured")
                                    if "Temperature_measured" in cycle_df.columns
                                    else None
                                ),
                            ),
                        )

                    cycle_count += 1
                except Exception as e:
                    print(f"    Error loading {filename}: {e}")
                    continue

        conn.commit()

    # Print summary
    cursor.execute("SELECT COUNT(DISTINCT battery_id) FROM batteries")
    battery_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM capacity_fade")
    capacity_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM resistance_data")
    resistance_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM cycles")
    cycles_count = cursor.fetchone()[0]

    print(f"\n{'='*50}")
    print("Data loading complete!")
    print(f"{'='*50}")
    print(f"Batteries: {battery_count}")
    print(f"Capacity fade records: {capacity_count}")
    print(f"Resistance records: {resistance_count}")
    print(f"Cycle data points: {cycles_count}")

    conn.close()


if __name__ == "__main__":
    # Paths
    project_root = Path(__file__).parent
    data_dir = project_root / "data" / "cleaned_dataset"
    db_path = project_root / "nasa_amperedata_full.db"

    print("NASA Battery Data Loader")
    print("=" * 50)
    print(f"Data directory: {data_dir}")
    print(f"Database path: {db_path}")
    print("=" * 50)

    load_nasa_data(data_dir, db_path)
