"""
Live Dashboard Page
===================
Real-time monitoring of battery data from connected cyclers.

Features:
- Live voltage/current charts with auto-refresh
- Real-time capacity tracking
- Multi-battery comparison
- Anomaly detection alerts
- Data streaming status

This page monitors for new data from:
- File Watcher (scripts/file_watcher.py)
- MQTT Connector (backend/connectors/mqtt_connector.py)
- Manual uploads
"""

import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import authentication
from utils.auth import require_auth, show_logout_button  # noqa: E402

# Page config
st.set_page_config(
    page_title="Live Dashboard - AmpereData",
    page_icon=None,
    layout="wide",
)

# Authentication
require_auth()

# Title
st.title("Live Dashboard")
st.markdown("Real-time monitoring of battery data from connected cyclers and data sources.")

# Show logout button
show_logout_button()

# Database path
DB_PATH = "nasa_amperedata_full.db"


@st.cache_data(ttl=5)  # Cache for 5 seconds for live updates
def get_recent_data(minutes: int = 60) -> pd.DataFrame:
    """Get recent data from the last N minutes."""
    conn = sqlite3.connect(DB_PATH)

    cutoff_time = (datetime.now() - timedelta(minutes=minutes)).isoformat()

    query = """
        SELECT
            c.battery_id,
            c.timestamp,
            c.voltage,
            c.current,
            c.temperature,
            c.capacity_ah,
            c.test_id as cycle_number
        FROM cycles c
        WHERE c.timestamp > ?
        ORDER BY c.timestamp DESC
        LIMIT 10000
    """

    try:
        df = pd.read_sql_query(query, conn, params=[cutoff_time])
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()
    finally:
        conn.close()


@st.cache_data(ttl=10)
def get_active_batteries() -> list:
    """Get list of batteries with recent activity."""
    conn = sqlite3.connect(DB_PATH)

    cutoff_time = (datetime.now() - timedelta(hours=24)).isoformat()

    query = """
        SELECT DISTINCT battery_id,
               MAX(timestamp) as last_seen,
               COUNT(*) as data_points
        FROM cycles
        WHERE timestamp > ?
        GROUP BY battery_id
        ORDER BY last_seen DESC
        LIMIT 50
    """

    try:
        df = pd.read_sql_query(query, conn, params=[cutoff_time])
        return df.to_dict("records")
    except Exception:
        return []
    finally:
        conn.close()


@st.cache_data(ttl=5)
def get_latest_values(battery_id: str) -> dict:
    """Get latest values for a battery."""
    conn = sqlite3.connect(DB_PATH)

    query = """
        SELECT
            voltage,
            current,
            temperature,
            capacity_ah,
            test_id as cycle_number,
            timestamp
        FROM cycles
        WHERE battery_id = ?
        ORDER BY timestamp DESC
        LIMIT 1
    """

    try:
        df = pd.read_sql_query(query, conn, params=[battery_id])
        if not df.empty:
            return df.iloc[0].to_dict()
        return {}
    except Exception:
        return {}
    finally:
        conn.close()


@st.cache_data(ttl=5)
def get_data_rate() -> dict:
    """Calculate data ingestion rate."""
    conn = sqlite3.connect(DB_PATH)

    # Data in last minute
    one_min_ago = (datetime.now() - timedelta(minutes=1)).isoformat()
    five_min_ago = (datetime.now() - timedelta(minutes=5)).isoformat()
    one_hour_ago = (datetime.now() - timedelta(hours=1)).isoformat()

    try:
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM cycles WHERE timestamp > ?", [one_min_ago])
        last_minute = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM cycles WHERE timestamp > ?", [five_min_ago])
        last_5_minutes = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM cycles WHERE timestamp > ?", [one_hour_ago])
        last_hour = cursor.fetchone()[0]

        return {
            "last_minute": last_minute,
            "last_5_minutes": last_5_minutes,
            "last_hour": last_hour,
            "rate_per_sec": last_minute / 60 if last_minute > 0 else 0,
        }
    except Exception:
        return {"last_minute": 0, "last_5_minutes": 0, "last_hour": 0, "rate_per_sec": 0}
    finally:
        conn.close()


def create_live_chart(df: pd.DataFrame, battery_id: str, metric: str = "voltage") -> go.Figure:
    """Create a live updating chart for a battery."""
    battery_data = df[df["battery_id"] == battery_id].sort_values("timestamp")

    if battery_data.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No recent data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    fig = go.Figure()

    if metric == "voltage" and "voltage" in battery_data.columns:
        fig.add_trace(
            go.Scatter(
                x=battery_data["timestamp"],
                y=battery_data["voltage"],
                mode="lines",
                name="Voltage (V)",
                line=dict(color="#2196F3", width=2),
            )
        )
        fig.update_layout(yaxis_title="Voltage (V)")

    elif metric == "current" and "current" in battery_data.columns:
        fig.add_trace(
            go.Scatter(
                x=battery_data["timestamp"],
                y=battery_data["current"],
                mode="lines",
                name="Current (A)",
                line=dict(color="#4CAF50", width=2),
            )
        )
        fig.update_layout(yaxis_title="Current (A)")

    elif metric == "temperature" and "temperature" in battery_data.columns:
        fig.add_trace(
            go.Scatter(
                x=battery_data["timestamp"],
                y=battery_data["temperature"],
                mode="lines",
                name="Temperature (C)",
                line=dict(color="#FF5722", width=2),
            )
        )
        fig.update_layout(yaxis_title="Temperature (C)")

    fig.update_layout(
        height=300,
        margin=dict(l=50, r=20, t=30, b=50),
        xaxis_title="Time",
        showlegend=False,
        template="plotly_white",
    )

    return fig


def create_multi_metric_chart(df: pd.DataFrame, battery_id: str) -> go.Figure:
    """Create a multi-metric chart with subplots."""
    battery_data = df[df["battery_id"] == battery_id].sort_values("timestamp")

    if battery_data.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No recent data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Voltage (V)", "Current (A)", "Temperature (C)"),
    )

    # Voltage
    if "voltage" in battery_data.columns:
        fig.add_trace(
            go.Scatter(
                x=battery_data["timestamp"],
                y=battery_data["voltage"],
                mode="lines",
                name="Voltage",
                line=dict(color="#2196F3", width=2),
            ),
            row=1,
            col=1,
        )

    # Current
    if "current" in battery_data.columns:
        fig.add_trace(
            go.Scatter(
                x=battery_data["timestamp"],
                y=battery_data["current"],
                mode="lines",
                name="Current",
                line=dict(color="#4CAF50", width=2),
            ),
            row=2,
            col=1,
        )

    # Temperature
    if "temperature" in battery_data.columns:
        fig.add_trace(
            go.Scatter(
                x=battery_data["timestamp"],
                y=battery_data["temperature"],
                mode="lines",
                name="Temperature",
                line=dict(color="#FF5722", width=2),
            ),
            row=3,
            col=1,
        )

    fig.update_layout(
        height=500,
        margin=dict(l=50, r=20, t=40, b=50),
        showlegend=False,
        template="plotly_white",
    )

    return fig


# Sidebar controls
st.sidebar.header("Live Dashboard Settings")

# Auto-refresh toggle
auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True, help="Automatically refresh data")
refresh_interval = st.sidebar.slider(
    "Refresh interval (seconds)",
    min_value=5,
    max_value=60,
    value=10,
    disabled=not auto_refresh,
)

# Time window
time_window = st.sidebar.selectbox(
    "Time window",
    options=[15, 30, 60, 120, 360, 1440],
    format_func=lambda x: f"{x} minutes" if x < 60 else f"{x // 60} hour(s)",
    index=2,
)

# Manual refresh button
if st.sidebar.button("Refresh Now"):
    st.cache_data.clear()
    st.rerun()

# Auto-refresh logic
if auto_refresh:
    # Add auto-refresh using streamlit-autorefresh or native rerun
    import time

    # Display countdown
    placeholder = st.sidebar.empty()

    # Use session state for countdown
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = time.time()

    elapsed = time.time() - st.session_state.last_refresh
    remaining = max(0, refresh_interval - int(elapsed))

    placeholder.info(f"Next refresh in {remaining}s")

    if elapsed >= refresh_interval:
        st.session_state.last_refresh = time.time()
        st.cache_data.clear()
        st.rerun()

# Main content
# Status metrics row
st.subheader("Data Ingestion Status")

rate_data = get_data_rate()
active_batteries = get_active_batteries()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Active Batteries",
        len(active_batteries),
        help="Batteries with data in the last 24 hours",
    )

with col2:
    st.metric(
        "Data Points (Last Hour)",
        f"{rate_data['last_hour']:,}",
        delta=f"{rate_data['last_5_minutes']} in 5 min",
    )

with col3:
    st.metric(
        "Ingestion Rate",
        f"{rate_data['rate_per_sec']:.1f}/sec",
        help="Average data points per second (last minute)",
    )

with col4:
    # Connection status
    is_active = rate_data["last_minute"] > 0
    status_color = "green" if is_active else "gray"
    status_text = "Receiving Data" if is_active else "No Recent Data"
    st.metric("Status", status_text)

st.markdown("---")

# Battery selection
if active_batteries:
    st.subheader("Live Battery Monitoring")

    # Battery selector
    battery_options = [b["battery_id"] for b in active_batteries]
    selected_batteries = st.multiselect(
        "Select batteries to monitor",
        options=battery_options,
        default=battery_options[: min(3, len(battery_options))],
        max_selections=6,
    )

    if selected_batteries:
        # Load recent data
        recent_data = get_recent_data(minutes=time_window)

        # Display charts for each selected battery
        for battery_id in selected_batteries:
            with st.expander(f"Battery: {battery_id}", expanded=True):
                # Latest values
                latest = get_latest_values(battery_id)

                if latest:
                    cols = st.columns(5)

                    with cols[0]:
                        voltage = latest.get("voltage")
                        st.metric(
                            "Voltage",
                            f"{voltage:.3f} V" if voltage else "N/A",
                        )

                    with cols[1]:
                        current = latest.get("current")
                        st.metric(
                            "Current",
                            f"{current:.3f} A" if current else "N/A",
                        )

                    with cols[2]:
                        temp = latest.get("temperature")
                        st.metric(
                            "Temperature",
                            f"{temp:.1f} C" if temp else "N/A",
                        )

                    with cols[3]:
                        capacity = latest.get("capacity_ah")
                        st.metric(
                            "Capacity",
                            f"{capacity:.3f} Ah" if capacity else "N/A",
                        )

                    with cols[4]:
                        cycle = latest.get("cycle_number")
                        st.metric(
                            "Cycle",
                            f"{int(cycle)}" if cycle else "N/A",
                        )

                    # Last updated time
                    timestamp = latest.get("timestamp")
                    if timestamp:
                        try:
                            ts = pd.to_datetime(timestamp)
                            st.caption(f"Last updated: {ts.strftime('%Y-%m-%d %H:%M:%S')}")
                        except Exception:
                            pass

                # Chart tabs
                chart_tab1, chart_tab2 = st.tabs(["Individual Metrics", "Combined View"])

                with chart_tab1:
                    metric_cols = st.columns(3)

                    with metric_cols[0]:
                        st.plotly_chart(
                            create_live_chart(recent_data, battery_id, "voltage"),
                            use_container_width=True,
                            key=f"voltage_{battery_id}",
                        )

                    with metric_cols[1]:
                        st.plotly_chart(
                            create_live_chart(recent_data, battery_id, "current"),
                            use_container_width=True,
                            key=f"current_{battery_id}",
                        )

                    with metric_cols[2]:
                        st.plotly_chart(
                            create_live_chart(recent_data, battery_id, "temperature"),
                            use_container_width=True,
                            key=f"temp_{battery_id}",
                        )

                with chart_tab2:
                    st.plotly_chart(
                        create_multi_metric_chart(recent_data, battery_id),
                        use_container_width=True,
                        key=f"combined_{battery_id}",
                    )
    else:
        st.info("Select one or more batteries to monitor.")

else:
    st.warning("No batteries with recent activity found.")
    st.info(
        """
        To start receiving live data, use one of these methods:

        **1. File Watcher** (for cyclers that export to folders):
        ```bash
        python scripts/file_watcher.py /path/to/cycler/output
        ```

        **2. MQTT Connector** (for IoT-enabled cyclers):
        ```bash
        python -m backend.connectors.mqtt_connector --broker mqtt.example.com
        ```

        **3. Manual Upload**: Go to the Upload Data page to manually upload files.
        """
    )

st.markdown("---")

# Data Source Configuration
with st.expander("Data Source Configuration", expanded=False):
    st.markdown(
        """
        ### Available Data Connectors

        | Connector | Status | Description |
        |-----------|--------|-------------|
        | **File Watcher** | Ready | Monitors folders for new files |
        | **MQTT** | Ready | Connects to IoT message brokers |
        | **REST API** | Available | Cyclers can push data via HTTP |

        ### Starting the File Watcher

        ```bash
        # Basic usage
        python scripts/file_watcher.py /path/to/cycler/output

        # With custom settings
        python scripts/file_watcher.py /data --db nasa_amperedata_full.db --prefix CYCLER1
        ```

        ### Starting the MQTT Connector

        ```bash
        # Connect to MQTT broker
        python -m backend.connectors.mqtt_connector --broker localhost --topic "battery/+/data"

        # With authentication
        python -m backend.connectors.mqtt_connector --broker mqtt.example.com --user admin --password secret
        ```

        ### Expected MQTT Message Format

        ```json
        {
            "voltage": 3.7,
            "current": 1.0,
            "temperature": 25.0,
            "capacity": 2.1,
            "cycle": 10,
            "timestamp": "2024-01-15T10:30:00Z"
        }
        ```

        Topic structure: `battery/{cell_id}/data`
        """
    )

# Alert Configuration (placeholder for future)
with st.expander("Alert Configuration (Coming Soon)", expanded=False):
    st.markdown(
        """
        Future features:
        - Voltage threshold alerts
        - Temperature warnings
        - Capacity fade detection
        - Anomaly detection
        - Email/SMS notifications
        """
    )

# Footer
st.markdown("---")
st.caption(
    f"Last page load: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | " f"Database: {DB_PATH}"
)
