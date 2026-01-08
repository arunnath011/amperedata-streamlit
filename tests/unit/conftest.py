"""
Shared test fixtures for unit tests.

Provides common fixtures for testing the AmpereData Streamlit application,
including mock data, sample files, and database sessions.
"""

import sqlite3
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Generator
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def temp_db() -> Generator[Path, None, None]:
    """
    Create a temporary SQLite database for testing.

    Yields:
        Path to temporary database file
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    # Create basic tables
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE batteries (
            battery_id TEXT PRIMARY KEY,
            total_cycles INTEGER,
            initial_capacity REAL,
            final_capacity REAL
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE capacity_fade (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            battery_id TEXT,
            cycle_number INTEGER,
            capacity_ah REAL,
            retention_percent REAL,
            timestamp TEXT
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE resistance_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            battery_id TEXT,
            re_ohms REAL,
            rct_ohms REAL,
            timestamp TEXT,
            cycle_number INTEGER
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE cycles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            battery_id TEXT,
            test_id TEXT,
            time_seconds REAL,
            voltage REAL,
            current REAL,
            capacity_ah REAL,
            temperature REAL
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE column_mappings (
            mapping_name TEXT PRIMARY KEY,
            description TEXT,
            mapping_json TEXT,
            created_at TEXT,
            last_used TEXT,
            use_count INTEGER DEFAULT 0
        )
    """
    )

    conn.commit()
    conn.close()

    yield db_path

    # Cleanup
    db_path.unlink(missing_ok=True)


@pytest.fixture
def sample_battery_data() -> pd.DataFrame:
    """
    Create sample battery cycling data.

    Returns:
        DataFrame with sample battery data
    """
    np.random.seed(42)
    n_cycles = 100

    # Simulate capacity fade
    initial_capacity = 2.0
    fade_rate = 0.002
    capacity = initial_capacity * (1 - fade_rate * np.arange(n_cycles))
    capacity += np.random.normal(0, 0.01, n_cycles)

    return pd.DataFrame(
        {
            "cycle_number": np.arange(n_cycles),
            "discharge_capacity": capacity,
            "charge_capacity": capacity * 1.01,
            "voltage_max": 4.2 + np.random.normal(0, 0.01, n_cycles),
            "voltage_min": 2.5 + np.random.normal(0, 0.01, n_cycles),
            "temperature": 25 + np.random.normal(0, 2, n_cycles),
        }
    )


@pytest.fixture
def sample_cycle_data() -> pd.DataFrame:
    """
    Create sample time-series cycle data.

    Returns:
        DataFrame with voltage, current, time data for one cycle
    """
    n_points = 1000
    time = np.linspace(0, 3600, n_points)  # 1 hour

    # Charge then discharge
    current = np.concatenate(
        [
            np.ones(n_points // 2) * 2.0,  # Charge at 2A
            np.ones(n_points // 2) * -2.0,  # Discharge at 2A
        ]
    )

    # Voltage profile
    voltage = np.concatenate(
        [
            np.linspace(3.0, 4.2, n_points // 2),  # Charge
            np.linspace(4.2, 2.5, n_points // 2),  # Discharge
        ]
    )

    return pd.DataFrame(
        {
            "time_seconds": time,
            "voltage": voltage,
            "current": current,
            "capacity_ah": np.cumsum(np.abs(current)) / 3600,
        }
    )


@pytest.fixture
def sample_eis_data() -> pd.DataFrame:
    """
    Create sample EIS (Electrochemical Impedance Spectroscopy) data.

    Returns:
        DataFrame with impedance data
    """
    frequencies = np.logspace(-2, 5, 50)

    # Simple Randles circuit model
    Rs = 0.05  # Solution resistance
    Rct = 0.2  # Charge transfer resistance
    Cdl = 1e-3  # Double layer capacitance

    omega = 2 * np.pi * frequencies
    Z_cdl = 1 / (1j * omega * Cdl)
    Z_total = Rs + (Rct * Z_cdl) / (Rct + Z_cdl)

    return pd.DataFrame(
        {
            "frequency": frequencies,
            "z_real": np.real(Z_total),
            "z_imag": -np.imag(Z_total),  # Convention: positive imaginary
        }
    )


@pytest.fixture
def sample_csv_content() -> bytes:
    """
    Create sample CSV file content for upload testing.

    Returns:
        Bytes content representing a CSV file
    """
    return b"""cycle,voltage,current,capacity
1,3.7,2.5,1.95
2,3.65,2.4,1.92
3,3.62,2.35,1.89
4,3.58,2.3,1.86
5,3.55,2.25,1.83
"""


@pytest.fixture
def sample_mpt_content() -> bytes:
    """
    Create sample BioLogic .mpt file content.

    Returns:
        Bytes representing an MPT file header and data
    """
    return b"""EC-Lab ASCII FILE
Nb header lines : 10

Technique : CV

mode\tox/red\terror\ttime/s\tEwe/V\t<I>/mA
1\t0\t0\t0.000\t0.500\t0.001
1\t0\t0\t0.100\t0.510\t0.002
1\t0\t0\t0.200\t0.520\t0.003
"""


@pytest.fixture
def mock_streamlit_session():
    """
    Create a mock Streamlit session state.

    Returns:
        Mock session state dictionary
    """
    return {
        "authenticated": True,
        "username": "test_user",
        "uploaded_data": None,
        "selected_batteries": [],
    }


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "unit: mark test as unit test")
