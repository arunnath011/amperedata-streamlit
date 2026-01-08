"""
Visualizations Page
===================
Interactive battery testing data visualizations.

Features:
- Capacity fade analysis
- EIS/Impedance spectroscopy
- Voltage profiles
- Cycle life analysis
- dQ/dV analysis
- Comparative analysis
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Page config
st.set_page_config(page_title="Visualizations - AmpereData", page_icon=None, layout="wide")

# Title
st.title("Battery Data Visualizations")
st.markdown("Interactive visualizations for battery testing data analysis.")

# Check if database exists
db_path = Path("nasa_amperedata_full.db")
if not db_path.exists():
    st.error("Database not found. Please upload some data first.")
    if st.button("Go to Upload Page"):
        st.switch_page("pages/1_Upload_Data.py")
    st.stop()


# Connect to database
@st.cache_resource
def get_db_connection():
    return sqlite3.connect(str(db_path), check_same_thread=False)


conn = get_db_connection()


# === Database initialization for saved groups ===
def init_saved_groups_table():
    """Initialize database table for saved battery groups."""
    import sqlite3

    conn_init = sqlite3.connect("nasa_amperedata_full.db")
    cursor = conn_init.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS saved_battery_groups (
            group_id TEXT PRIMARY KEY,
            group_name TEXT UNIQUE NOT NULL,
            original_batteries TEXT NOT NULL,
            aggregation_method TEXT NOT NULL,
            confidence_level INTEGER,
            data_type TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            notes TEXT
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS saved_group_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            group_id TEXT NOT NULL,
            data_table TEXT NOT NULL,
            cycle_number INTEGER,
            value REAL,
            lower_ci REAL,
            upper_ci REAL,
            std_dev REAL,
            count INTEGER,
            FOREIGN KEY (group_id) REFERENCES saved_battery_groups(group_id)
        )
    """
    )

    conn_init.commit()
    conn_init.close()


# Initialize the tables
init_saved_groups_table()


# === Functions for saving and managing grouped data ===
def save_grouped_battery_data(
    group_name, battery_ids, group_stats, method, confidence, data_type="capacity_fade"
):
    """Save grouped battery data to database."""
    import sqlite3
    import uuid
    from datetime import datetime

    conn_save = sqlite3.connect("nasa_amperedata_full.db")
    cursor = conn_save.cursor()

    try:
        group_id = f"GROUP_{uuid.uuid4().hex[:8]}"
        now = datetime.now().isoformat()

        # Save group metadata
        cursor.execute(
            """
            INSERT INTO saved_battery_groups
            (group_id, group_name, original_batteries, aggregation_method,
             confidence_level, data_type, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (group_id, group_name, ",".join(battery_ids), method, confidence, data_type, now, now),
        )

        # Save group data
        for _, row in group_stats.iterrows():
            cursor.execute(
                """
                INSERT INTO saved_group_data
                (group_id, data_table, cycle_number, value, lower_ci, upper_ci, std_dev, count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    group_id,
                    data_type,
                    int(row.get("x", row.get("cycle_number", 0))),
                    float(row["y_avg"]),
                    float(row["y_lower"]),
                    float(row["y_upper"]),
                    float(row.get("std", 0)),
                    int(row.get("count", 0)),
                ),
            )

        conn_save.commit()
        return True, group_id
    except sqlite3.IntegrityError:
        return False, "Group name already exists"
    except Exception as e:
        return False, str(e)
    finally:
        conn_save.close()


def load_saved_groups():
    """Load all saved battery groups."""
    import sqlite3

    conn_load = sqlite3.connect("nasa_amperedata_full.db")
    try:
        df = pd.read_sql_query(
            "SELECT * FROM saved_battery_groups ORDER BY created_at DESC", conn_load
        )
        return df
    except Exception:
        return pd.DataFrame()
    finally:
        conn_load.close()


def delete_saved_group(group_id):
    """Delete a saved battery group."""
    import sqlite3

    conn_del = sqlite3.connect("nasa_amperedata_full.db")
    cursor = conn_del.cursor()
    try:
        cursor.execute("DELETE FROM saved_group_data WHERE group_id = ?", (group_id,))
        cursor.execute("DELETE FROM saved_battery_groups WHERE group_id = ?", (group_id,))
        conn_del.commit()
        return True
    except Exception:
        return False
    finally:
        conn_del.close()


def update_saved_group_metadata(group_id, group_name=None, notes=None):
    """Update metadata for a saved group."""
    import sqlite3
    from datetime import datetime

    conn_update = sqlite3.connect("nasa_amperedata_full.db")
    cursor = conn_update.cursor()
    try:
        updates = []
        params = []

        if group_name:
            updates.append("group_name = ?")
            params.append(group_name)

        if notes is not None:
            updates.append("notes = ?")
            params.append(notes)

        if updates:
            updates.append("updated_at = ?")
            params.append(datetime.now().isoformat())
            params.append(group_id)

            query = f"UPDATE saved_battery_groups SET {', '.join(updates)} WHERE group_id = ?"
            cursor.execute(query, params)
            conn_update.commit()
            return True
        return False
    except Exception:
        return False
    finally:
        conn_update.close()


def load_group_data(group_id):
    """Load data for a specific saved group."""
    import sqlite3

    conn_grp = sqlite3.connect("nasa_amperedata_full.db")
    try:
        df = pd.read_sql_query(
            "SELECT * FROM saved_group_data WHERE group_id = ? ORDER BY cycle_number",
            conn_grp,
            params=(group_id,),
        )
        return df
    finally:
        conn_grp.close()


def create_saved_group_visualization(group_data, group_name, confidence_level):
    """Create a plotly figure for saved group data."""
    fig = go.Figure()

    # Add confidence interval
    fig.add_trace(
        go.Scatter(
            x=group_data["cycle_number"],
            y=group_data["upper_ci"],
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=group_data["cycle_number"],
            y=group_data["lower_ci"],
            mode="lines",
            line={"width": 0},
            fillcolor="rgba(255, 0, 0, 0.2)",
            fill="tonexty",
            name=f"{confidence_level}% CI",
            hovertemplate="Cycle: %{x}<br>Lower CI: %{y:.3f} Ah<extra></extra>",
        )
    )

    # Add average line
    fig.add_trace(
        go.Scatter(
            x=group_data["cycle_number"],
            y=group_data["value"],
            mode="lines+markers",
            name="Group Average",
            line={"color": "red", "width": 3},
            marker={"size": 6, "color": "red"},
            hovertemplate="Cycle: %{x}<br>Average: %{y:.3f} Ah<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"{group_name} - Capacity Fade",
        xaxis_title="Cycle Number",
        yaxis_title="Capacity (Ah)",
        height=500,
        hovermode="x unified",
        showlegend=True,
        template="plotly_white",
    )

    return fig


# Load available batteries
@st.cache_data(ttl=60)
def load_batteries():
    try:
        query = "SELECT DISTINCT battery_id FROM batteries ORDER BY battery_id"
        df = pd.read_sql_query(query, conn)
        return df["battery_id"].tolist()
    except Exception as e:
        st.error(f"Error loading batteries: {e}")
        return []


batteries = load_batteries()

if not batteries:
    st.warning("No battery data found in database. Upload data to get started.")
    if st.button("Go to Upload Page"):
        st.switch_page("pages/1_Upload_Data.py")
    st.stop()

# Sidebar controls
with st.sidebar:
    st.subheader("Visualization Controls")

    # Load build types for filtering
    @st.cache_data(ttl=60)
    def load_build_types():
        try:
            query = """
                SELECT DISTINCT bs.build_type
                FROM build_sheets bs
                JOIN battery_metadata bm ON bs.build_id = bm.build_id
                WHERE bs.build_type IS NOT NULL
                ORDER BY bs.build_type
            """
            df = pd.read_sql_query(query, conn)
            return df["build_type"].tolist()
        except Exception:
            return []

    build_types = load_build_types()

    # Filter by build type
    if build_types:
        selected_build_type = st.selectbox(
            "Filter by Build Type", ["All"] + build_types, help="Filter batteries by build type"
        )

        # Filter batteries based on build type
        if selected_build_type != "All":
            try:
                filtered_query = f"""
                    SELECT DISTINCT bm.battery_id
                    FROM battery_metadata bm
                    JOIN build_sheets bs ON bm.build_id = bs.build_id
                    WHERE bs.build_type = '{selected_build_type}'
                    ORDER BY bm.battery_id
                """
                filtered_df = pd.read_sql_query(filtered_query, conn)
                available_batteries = filtered_df["battery_id"].tolist()

                # Filter to only batteries that exist in main list
                batteries = [b for b in batteries if b in available_batteries]
            except Exception:
                pass

    # Battery selection
    selected_batteries = st.multiselect(
        "Select Batteries",
        batteries,
        default=[batteries[0]] if batteries else [],
        help="Choose one or more batteries to visualize",
    )

    st.markdown("---")

    # Visualization type
    viz_type = st.selectbox(
        "Visualization Type",
        [
            "Capacity Fade",
            "Voltage Profiles",
            "EIS/Impedance",
            "Cycle Statistics",
            "dQ/dV Analysis",
            "Calendar Life",
            "Comparative Analysis",
        ],
    )

    st.markdown("---")

    # Additional filters
    with st.expander("Advanced Options"):
        show_grid = st.checkbox("Show grid", value=True)
        show_legend = st.checkbox("Show legend", value=True)
        normalize_data = st.checkbox("Normalize data", value=False)

    st.markdown("---")

    # Battery Grouping Options
    if len(selected_batteries) > 1:
        with st.expander("Battery Grouping", expanded=False):
            enable_grouping = st.checkbox(
                "Enable Group Statistics",
                value=False,
                help="Show average line with confidence intervals for selected batteries",
            )

            if enable_grouping:
                confidence_level = st.slider(
                    "Confidence Interval",
                    min_value=90,
                    max_value=99,
                    value=95,
                    step=1,
                    help="95% means 95% of data falls within the shaded region",
                )

                show_individual = st.checkbox(
                    "Show Individual Batteries",
                    value=True,
                    help="Show individual battery traces along with group average",
                )

                group_method = st.radio(
                    "Aggregation Method",
                    ["Mean", "Median"],
                    help="Method to calculate group average",
                )

                # === NEW: Save Grouped Data Feature ===
                st.markdown("---")
                st.markdown("**Save Grouped Data**")

                col1, col2 = st.columns([3, 1])
                with col1:
                    group_name = st.text_input(
                        "Group Name",
                        placeholder="e.g., NMC811_Batch_A",
                        help="Name for this grouped battery dataset",
                        key="group_name_input",
                    )
                with col2:
                    st.write("")  # Spacing
                    st.write("")  # Spacing
                    save_button = st.button(
                        "Save",
                        help="Save grouped data as new battery entry",
                        type="primary",
                        use_container_width=True,
                    )

                if save_button and group_name:
                    # Will implement save functionality below
                    st.session_state["pending_save"] = {
                        "name": group_name,
                        "batteries": selected_batteries,
                        "method": group_method,
                        "confidence": confidence_level,
                    }
                    st.success(f"'{group_name}' will be saved after data is loaded")
                elif save_button:
                    st.error("Please enter a group name")
            else:
                confidence_level = 95
                show_individual = True
                group_method = "Mean"
    else:
        enable_grouping = False
        confidence_level = 95
        show_individual = True
        group_method = "Mean"

# === NEW: Manage Saved Groups UI ===
with st.sidebar.expander("Manage Saved Groups", expanded=False):
    saved_groups_df = load_saved_groups()

    if not saved_groups_df.empty:
        st.markdown(f"**{len(saved_groups_df)} saved groups**")

        for _idx, group in saved_groups_df.iterrows():
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

                with col1:
                    st.markdown(f"**{group['group_name']}**")
                    batteries = group["original_batteries"].split(",")
                    st.caption(f"{len(batteries)} batteries • {group['aggregation_method']}")

                with col2:
                    view_button = st.button(
                        "View", key=f"view_{group['group_id']}", help="View details"
                    )

                with col3:
                    edit_button = st.button(
                        "Edit", key=f"edit_{group['group_id']}", help="Edit group"
                    )

                with col4:
                    delete_button = st.button(
                        "Del", key=f"del_{group['group_id']}", help="Delete group"
                    )

                if delete_button:
                    if delete_saved_group(group["group_id"]):
                        st.success(f"Deleted '{group['group_name']}'")
                        st.rerun()
                    else:
                        st.error("Failed to delete")

                if edit_button:
                    st.session_state[f'editing_{group["group_id"]}'] = True

                if view_button:
                    st.session_state["viewing_group"] = group["group_id"]

                # Edit form
                if st.session_state.get(f'editing_{group["group_id"]}', False):
                    with st.form(key=f"edit_form_{group['group_id']}"):
                        st.markdown("**Edit Group**")
                        new_name = st.text_input("Group Name", value=group["group_name"])
                        new_notes = st.text_area(
                            "Notes", value=group.get("notes", "") or "", height=80
                        )

                        col_save, col_cancel = st.columns(2)
                        with col_save:
                            save_edit = st.form_submit_button("Save", use_container_width=True)
                        with col_cancel:
                            cancel_edit = st.form_submit_button("Cancel", use_container_width=True)

                        if save_edit:
                            if update_saved_group_metadata(group["group_id"], new_name, new_notes):
                                st.success(f"Updated '{new_name}'")
                                del st.session_state[f'editing_{group["group_id"]}']
                                st.rerun()
                            else:
                                st.error("Failed to update")

                        if cancel_edit:
                            del st.session_state[f'editing_{group["group_id"]}']
                            st.rerun()

                # View details
                if st.session_state.get("viewing_group") == group["group_id"]:
                    with st.expander(f"{group['group_name']} Details", expanded=True):
                        col_info, col_viz = st.columns([1, 2])

                        with col_info:
                            st.markdown("**Group Information**")
                            st.write(f"**Created:** {group['created_at'][:10]}")
                            st.write(f"**Updated:** {group['updated_at'][:10]}")
                            st.write(f"**Method:** {group['aggregation_method']}")
                            st.write(f"**Confidence:** {group['confidence_level']}%")
                            st.write(f"**Data Type:** {group['data_type']}")

                            batteries = group["original_batteries"].split(",")
                            st.write(f"**Batteries ({len(batteries)}):**")
                            for bat in batteries[:5]:
                                st.caption(f"  • {bat}")
                            if len(batteries) > 5:
                                st.caption(f"  ... and {len(batteries) - 5} more")

                            if group.get("notes"):
                                st.write(f"**Notes:** {group['notes']}")

                        with col_viz:
                            group_data = load_group_data(group["group_id"])
                            if not group_data.empty:
                                st.markdown("**Visualization**")
                                fig = create_saved_group_visualization(
                                    group_data, group["group_name"], group["confidence_level"]
                                )
                                st.plotly_chart(fig, use_container_width=True)

                        # Data table below
                        group_data = load_group_data(group["group_id"])
                        if not group_data.empty:
                            st.markdown("---")
                            st.markdown(f"**Data Summary** ({len(group_data)} data points)")
                            display_cols = [
                                "cycle_number",
                                "value",
                                "lower_ci",
                                "upper_ci",
                                "std_dev",
                                "count",
                            ]
                            display_df = group_data[display_cols].copy()
                            display_df.columns = [
                                "Cycle",
                                "Average (Ah)",
                                "Lower CI",
                                "Upper CI",
                                "Std Dev",
                                "N",
                            ]
                            st.dataframe(display_df.head(20), use_container_width=True)

                st.markdown("---")
    else:
        st.info("No saved groups yet. Create one by enabling grouping and clicking Save!")


# Helper functions for battery grouping
def calculate_group_statistics(data_dict, x_col, y_col, method="Mean", confidence=95):
    """
    Calculate group statistics with confidence intervals.

    Args:
        data_dict: Dict of {battery_id: DataFrame}
        x_col: Column name for x-axis (e.g., 'cycle_number')
        y_col: Column name for y-axis (e.g., 'discharge_capacity')
        method: 'Mean' or 'Median'
        confidence: Confidence level (90-99)

    Returns:
        DataFrame with columns: x, y_avg, y_lower, y_upper, count
    """
    from scipy import stats

    # Combine all data
    all_data = []
    for battery_id, df in data_dict.items():
        if not df.empty and x_col in df.columns and y_col in df.columns:
            temp_df = df[[x_col, y_col]].copy()
            temp_df["battery_id"] = battery_id
            all_data.append(temp_df)

    if not all_data:
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)

    # Group by x_col and calculate statistics
    grouped = combined.groupby(x_col)[y_col]

    if method == "Mean":
        y_center = grouped.mean()
    else:  # Median
        y_center = grouped.median()

    y_std = grouped.std()
    y_count = grouped.count()

    # Calculate confidence intervals
    alpha = 1 - (confidence / 100)
    z_score = stats.norm.ppf(1 - alpha / 2)

    # Standard error of the mean
    y_sem = y_std / np.sqrt(y_count)

    # Confidence interval
    y_lower = y_center - z_score * y_sem
    y_upper = y_center + z_score * y_sem

    # Create result DataFrame
    result = pd.DataFrame(
        {
            "x": y_center.index,
            "y_avg": y_center.values,
            "y_lower": y_lower.values,
            "y_upper": y_upper.values,
            "count": y_count.values,
            "std": y_std.values,
        }
    )

    return result


def add_group_trace_to_figure(fig, group_stats, name="Group Average", color="red"):
    """Add group average line and confidence interval to plotly figure."""
    if group_stats.empty:
        return fig

    # Add confidence interval (shaded region)
    fig.add_trace(
        go.Scatter(
            x=group_stats["x"],
            y=group_stats["y_upper"],
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=group_stats["x"],
            y=group_stats["y_lower"],
            mode="lines",
            line={"width": 0},
            fillcolor="rgba(255, 0, 0, 0.2)",
            fill="tonexty",
            name=f"{name} (CI)",
            hovertemplate="CI: %{y:.3f}<extra></extra>",
        )
    )

    # Add average line
    fig.add_trace(
        go.Scatter(
            x=group_stats["x"],
            y=group_stats["y_avg"],
            mode="lines+markers",
            name=name,
            line={"color": color, "width": 3, "dash": "solid"},
            marker={"size": 6, "color": color},
            hovertemplate=(f"{name}<br>" + "X: %{x}<br>" + "Avg: %{y:.3f}<br>" + "<extra></extra>"),
        )
    )

    return fig


# Load battery metadata function
@st.cache_data(ttl=60)
def load_battery_metadata_info(battery_ids):
    """Load metadata for selected batteries."""
    try:
        query = f"""
            SELECT
                bm.battery_id,
                bm.serial_number,
                bm.status,
                bm.position_in_build,
                bm.assembly_date,
                bs.build_id,
                bs.build_name,
                bs.build_type,
                bs.cathode_material,
                bs.anode_material,
                bs.electrolyte,
                bs.nominal_capacity_ah,
                bs.nominal_voltage_v,
                bs.form_factor,
                bs.manufacturer
            FROM battery_metadata bm
            LEFT JOIN build_sheets bs ON bm.build_id = bs.build_id
            WHERE bm.battery_id IN ({','.join([f"'{b}'" for b in battery_ids])})
        """
        df = pd.read_sql_query(query, conn)
        return df
    except Exception:
        return pd.DataFrame()


# Main visualization area
if not selected_batteries:
    st.info("Select one or more batteries from the sidebar to begin visualization")
else:
    # Show metadata panel if available
    metadata_df = load_battery_metadata_info(selected_batteries)

    if not metadata_df.empty:
        with st.expander("Battery Build Sheet Information", expanded=False):
            for _, battery_meta in metadata_df.iterrows():
                st.markdown(f"### {battery_meta['battery_id']}")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("**Build Information**")
                    if pd.notna(battery_meta.get("build_name")):
                        st.markdown(f"**Build:** {battery_meta['build_name']}")
                    if pd.notna(battery_meta.get("build_type")):
                        st.markdown(f"**Type:** {battery_meta['build_type']}")
                    if pd.notna(battery_meta.get("serial_number")):
                        st.markdown(f"**Serial:** {battery_meta['serial_number']}")
                    if pd.notna(battery_meta.get("status")):
                        st.markdown(f"**Status:** {battery_meta['status']}")

                with col2:
                    st.markdown("**Materials**")
                    if pd.notna(battery_meta.get("cathode_material")):
                        st.markdown(f"**Cathode:** {battery_meta['cathode_material']}")
                    if pd.notna(battery_meta.get("anode_material")):
                        st.markdown(f"**Anode:** {battery_meta['anode_material']}")
                    if pd.notna(battery_meta.get("electrolyte")):
                        st.markdown(f"**Electrolyte:** {battery_meta['electrolyte']}")
                    if pd.notna(battery_meta.get("form_factor")):
                        st.markdown(f"**Form:** {battery_meta['form_factor']}")

                with col3:
                    st.markdown("**Specifications**")
                    if pd.notna(battery_meta.get("nominal_capacity_ah")):
                        st.markdown(f"**Capacity:** {battery_meta['nominal_capacity_ah']} Ah")
                    if pd.notna(battery_meta.get("nominal_voltage_v")):
                        st.markdown(f"**Voltage:** {battery_meta['nominal_voltage_v']} V")
                    if pd.notna(battery_meta.get("manufacturer")):
                        st.markdown(f"**Mfg:** {battery_meta['manufacturer']}")
                    if pd.notna(battery_meta.get("assembly_date")):
                        st.markdown(f"**Assembled:** {battery_meta['assembly_date']}")

                st.markdown("---")

            # Link to build sheet page
            if st.button("Manage Build Sheets"):
                st.switch_page("pages/6_Build_Sheet.py")

    # Load data for selected batteries
    @st.cache_data(ttl=60)
    def load_battery_data(battery_id):
        try:
            # Load capacity fade data
            query_capacity = f"""
                SELECT cycle_number, capacity_ah as discharge_capacity,
                       retention_percent, timestamp
                FROM capacity_fade
                WHERE battery_id = '{battery_id}'
                ORDER BY cycle_number
            """
            capacity_df = pd.read_sql_query(query_capacity, conn)

            # Add charge_capacity (same as discharge for now)
            if not capacity_df.empty:
                capacity_df["charge_capacity"] = capacity_df["discharge_capacity"]

            # Load resistance data
            query_resistance = f"""
                SELECT re_ohms as Re, rct_ohms as Rct, timestamp
                FROM resistance_data
                WHERE battery_id = '{battery_id}'
                ORDER BY timestamp
            """
            resistance_df = pd.read_sql_query(query_resistance, conn)

            return capacity_df, resistance_df
        except Exception as e:
            st.error(f"Error loading data for {battery_id}: {e}")
            return pd.DataFrame(), pd.DataFrame()

    # === CAPACITY FADE VISUALIZATION ===
    if viz_type == "Capacity Fade":
        st.subheader("Capacity Fade Analysis")

        fig = go.Figure()

        # Collect all battery data for grouping
        batteries_data_dict = {}

        for battery_id in selected_batteries:
            capacity_df, _ = load_battery_data(battery_id)

            if not capacity_df.empty:
                batteries_data_dict[battery_id] = capacity_df

                # Add individual traces
                if not enable_grouping or show_individual:
                    opacity = 0.4 if enable_grouping and show_individual else 1.0
                    fig.add_trace(
                        go.Scatter(
                            x=capacity_df["cycle_number"],
                            y=capacity_df["discharge_capacity"],
                            mode="lines+markers",
                            name=f"{battery_id} - Discharge",
                            line={"width": 2 if not enable_grouping else 1},
                            marker={"size": 6 if not enable_grouping else 4},
                            opacity=opacity,
                        )
                    )

        # Add group statistics if enabled
        if enable_grouping and len(batteries_data_dict) > 1:
            group_stats = calculate_group_statistics(
                batteries_data_dict,
                x_col="cycle_number",
                y_col="discharge_capacity",
                method=group_method,
                confidence=confidence_level,
            )

            if not group_stats.empty:
                # === NEW: Save grouped data if requested ===
                if "pending_save" in st.session_state:
                    save_info = st.session_state["pending_save"]
                    success, result = save_grouped_battery_data(
                        group_name=save_info["name"],
                        battery_ids=save_info["batteries"],
                        group_stats=group_stats,
                        method=save_info["method"],
                        confidence=save_info["confidence"],
                        data_type="capacity_fade",
                    )

                    if success:
                        st.success(f"Saved '{save_info['name']}' successfully! (ID: {result})")
                        del st.session_state["pending_save"]
                        # Clear the input
                        if "group_name_input" in st.session_state:
                            del st.session_state["group_name_input"]
                    else:
                        st.error(f"Failed to save: {result}")
                        if "pending_save" in st.session_state:
                            del st.session_state["pending_save"]

                fig = add_group_trace_to_figure(
                    fig,
                    group_stats,
                    name=f"Group {group_method} ({len(batteries_data_dict)} batteries)",
                    color="red",
                )

                # Display group statistics
                st.info(
                    f"**Group Statistics:** Showing {group_method.lower()} with {confidence_level}% confidence interval across {len(batteries_data_dict)} batteries"
                )

        fig.update_layout(
            title="Discharge Capacity vs Cycle Number",
            xaxis_title="Cycle Number",
            yaxis_title="Capacity (Ah)",
            height=600,
            hovermode="x unified",
            showlegend=show_legend,
            xaxis={"showgrid": show_grid},
            yaxis={"showgrid": show_grid},
            template="plotly_white",
        )

        st.plotly_chart(fig, use_container_width=True)

        # Show detailed group statistics if enabled
        if enable_grouping and len(batteries_data_dict) > 1 and not group_stats.empty:
            with st.expander("Detailed Group Statistics"):
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Group Size", f"{len(batteries_data_dict)} batteries")
                with col2:
                    avg_capacity = group_stats["y_avg"].mean()
                    st.metric("Average Capacity", f"{avg_capacity:.3f} Ah")
                with col3:
                    avg_std = group_stats["std"].mean()
                    st.metric("Average Std Dev", f"{avg_std:.3f} Ah")
                with col4:
                    st.metric("Confidence Level", f"{confidence_level}%")

                # Show statistics table
                st.markdown("**Statistics by Cycle:**")
                stats_display = group_stats[
                    ["x", "y_avg", "y_lower", "y_upper", "count", "std"]
                ].copy()
                stats_display.columns = [
                    "Cycle",
                    "Average (Ah)",
                    "Lower CI",
                    "Upper CI",
                    "N Batteries",
                    "Std Dev",
                ]
                st.dataframe(stats_display.head(20), use_container_width=True)

        # Statistics
        st.markdown("---")
        st.subheader("Statistics")
        cols = st.columns(len(selected_batteries))
        for idx, battery_id in enumerate(selected_batteries):
            capacity_df, _ = load_battery_data(battery_id)
            if not capacity_df.empty:
                with cols[idx]:
                    st.metric(
                        f"{battery_id}",
                        f"{capacity_df['discharge_capacity'].iloc[-1]:.3f} Ah",
                        f"{(capacity_df['discharge_capacity'].iloc[-1] - capacity_df['discharge_capacity'].iloc[0]):.3f} Ah",
                    )
                    st.caption(f"Total Cycles: {len(capacity_df)}")

    # === VOLTAGE PROFILES ===
    elif viz_type == "Voltage Profiles":
        st.subheader("Voltage Profiles")

        # Get available cycles for selected batteries
        try:
            max_cycle_query = f"""
                SELECT MAX(CAST(test_id AS INTEGER)) as max_cycle
                FROM cycles
                WHERE battery_id IN ({','.join([f"'{b}'" for b in selected_batteries])})
            """
            max_cycle_result = pd.read_sql_query(max_cycle_query, conn)
            max_cycle = (
                int(max_cycle_result["max_cycle"].iloc[0]) if not max_cycle_result.empty else 100
            )
            max_cycle = max(max_cycle, 1)  # Ensure at least 1
        except Exception:
            max_cycle = 100

        # Cycle selection
        cycle_numbers = st.slider(
            "Select Cycles to Display",
            min_value=0,
            max_value=max_cycle,
            value=(0, min(10, max_cycle)),
            step=1,
            help=f"Select cycle range (0 to {max_cycle})",
        )

        col1, col2 = st.columns(2)
        with col1:
            show_individual_cycles = st.checkbox(
                "Show Individual Cycles", value=True, help="Show each cycle separately"
            )
        with col2:
            overlay_cycles = st.checkbox(
                "Overlay Cycles", value=False, help="Overlay all cycles on same plot"
            )

        # Try to load voltage data from cycles table
        try:
            fig = go.Figure()
            colors = px.colors.qualitative.Plotly

            for idx, battery_id in enumerate(selected_batteries):
                # Query cycles within selected range
                query = f"""
                    SELECT time_seconds, voltage, current, test_id
                    FROM cycles
                    WHERE battery_id = '{battery_id}'
                      AND CAST(test_id AS INTEGER) BETWEEN {cycle_numbers[0]} AND {cycle_numbers[1]}
                    ORDER BY test_id, time_seconds
                """
                voltage_df = pd.read_sql_query(query, conn)

                if not voltage_df.empty:
                    if show_individual_cycles and not overlay_cycles:
                        # Show each cycle as separate trace
                        for cycle in voltage_df["test_id"].unique():
                            cycle_data = voltage_df[voltage_df["test_id"] == cycle]
                            fig.add_trace(
                                go.Scatter(
                                    x=cycle_data["time_seconds"],
                                    y=cycle_data["voltage"],
                                    mode="lines",
                                    name=f"{battery_id} - Cycle {cycle}",
                                    line={"width": 2, "color": colors[idx % len(colors)]},
                                    legendgroup=battery_id,
                                )
                            )
                    elif overlay_cycles:
                        # Overlay cycles by normalizing time to start from 0 for each cycle
                        for cycle in voltage_df["test_id"].unique():
                            cycle_data = voltage_df[voltage_df["test_id"] == cycle].copy()
                            # Normalize time to start from 0
                            cycle_data["time_normalized"] = (
                                cycle_data["time_seconds"] - cycle_data["time_seconds"].min()
                            )
                            fig.add_trace(
                                go.Scatter(
                                    x=cycle_data["time_normalized"],
                                    y=cycle_data["voltage"],
                                    mode="lines",
                                    name=f"{battery_id} - Cycle {cycle}",
                                    line={"width": 2},
                                    opacity=0.7,
                                    legendgroup=battery_id,
                                )
                            )
                    else:
                        # Show all selected cycles as one continuous trace
                        fig.add_trace(
                            go.Scatter(
                                x=voltage_df["time_seconds"],
                                y=voltage_df["voltage"],
                                mode="lines",
                                name=f"{battery_id} (Cycles {cycle_numbers[0]}-{cycle_numbers[1]})",
                                line={"width": 2, "color": colors[idx % len(colors)]},
                            )
                        )

            if fig.data:
                x_title = (
                    "Time (seconds)" if not overlay_cycles else "Time from Cycle Start (seconds)"
                )
                fig.update_layout(
                    title=f"Voltage Profiles - Cycles {cycle_numbers[0]} to {cycle_numbers[1]}",
                    xaxis_title=x_title,
                    yaxis_title="Voltage (V)",
                    height=600,
                    showlegend=show_legend,
                    template="plotly_white",
                    hovermode="x unified",
                )

                st.plotly_chart(fig, use_container_width=True)

                # Show cycle statistics
                if not voltage_df.empty:
                    st.markdown("### Cycle Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric(
                            "Cycles Displayed",
                            f"{cycle_numbers[1] - cycle_numbers[0] + 1}",
                        )
                    with col2:
                        st.metric(
                            "Voltage Range",
                            f"{voltage_df['voltage'].min():.2f} - {voltage_df['voltage'].max():.2f} V",
                        )
                    with col3:
                        st.metric(
                            "Current Range",
                            f"{voltage_df['current'].min():.3f} - {voltage_df['current'].max():.3f} A",
                        )
                    with col4:
                        st.metric("Data Points", len(voltage_df))
            else:
                st.warning(
                    f"No voltage data found for cycles {cycle_numbers[0]} to {cycle_numbers[1]}"
                )

        except Exception as e:
            st.error(f"Error loading voltage profile data: {e}")
            import traceback

            with st.expander("Error Details"):
                st.code(traceback.format_exc())

    # === EIS/IMPEDANCE ===
    elif viz_type == "EIS/Impedance":
        st.subheader("EIS/Impedance Spectroscopy")

        fig_re = go.Figure()
        fig_rct = go.Figure()

        for battery_id in selected_batteries:
            _, resistance_df = load_battery_data(battery_id)

            if not resistance_df.empty and "Re" in resistance_df.columns:
                # Re evolution
                fig_re.add_trace(
                    go.Scatter(
                        x=resistance_df.index,
                        y=resistance_df["Re"],
                        mode="lines+markers",
                        name=battery_id,
                        line={"width": 2},
                        marker={"size": 6},
                    )
                )

                # Rct evolution
                if "Rct" in resistance_df.columns:
                    fig_rct.add_trace(
                        go.Scatter(
                            x=resistance_df.index,
                            y=resistance_df["Rct"],
                            mode="lines+markers",
                            name=battery_id,
                            line={"width": 2},
                            marker={"size": 6},
                        )
                    )

        # Plot Re
        fig_re.update_layout(
            title="Electrolyte Resistance (Re) Evolution",
            xaxis_title="Test Index",
            yaxis_title="Re (Ω)",
            height=400,
            showlegend=show_legend,
            template="plotly_white",
        )
        st.plotly_chart(fig_re, use_container_width=True)

        # Plot Rct
        fig_rct.update_layout(
            title="Charge Transfer Resistance (Rct) Evolution",
            xaxis_title="Test Index",
            yaxis_title="Rct (Ω)",
            height=400,
            showlegend=show_legend,
            template="plotly_white",
        )
        st.plotly_chart(fig_rct, use_container_width=True)

    # === CYCLE STATISTICS ===
    elif viz_type == "Cycle Statistics":
        st.subheader("Cycle Statistics")

        for battery_id in selected_batteries:
            capacity_df, _ = load_battery_data(battery_id)

            if not capacity_df.empty:
                st.markdown(f"### {battery_id}")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Cycles", len(capacity_df))
                with col2:
                    st.metric(
                        "Initial Capacity",
                        f"{capacity_df['discharge_capacity'].iloc[0]:.3f} Ah",
                    )
                with col3:
                    st.metric(
                        "Final Capacity",
                        f"{capacity_df['discharge_capacity'].iloc[-1]:.3f} Ah",
                    )
                with col4:
                    fade = (
                        (
                            capacity_df["discharge_capacity"].iloc[0]
                            - capacity_df["discharge_capacity"].iloc[-1]
                        )
                        / capacity_df["discharge_capacity"].iloc[0]
                        * 100
                    )
                    st.metric("Capacity Fade", f"{fade:.1f}%")

                # Distribution plots
                col1, col2 = st.columns(2)

                with col1:
                    fig_hist = px.histogram(
                        capacity_df,
                        x="discharge_capacity",
                        title=f"{battery_id} - Capacity Distribution",
                        nbins=30,
                    )
                    fig_hist.update_layout(height=300)
                    st.plotly_chart(fig_hist, use_container_width=True)

                with col2:
                    fig_box = px.box(
                        capacity_df,
                        y="discharge_capacity",
                        title=f"{battery_id} - Capacity Box Plot",
                    )
                    fig_box.update_layout(height=300)
                    st.plotly_chart(fig_box, use_container_width=True)

                st.markdown("---")

    # === dQ/dV ANALYSIS ===
    elif viz_type == "dQ/dV Analysis":
        from scipy.signal import savgol_filter

        st.subheader("dQ/dV (Differential Capacity) Analysis")

        st.info(
            """
        **What is dQ/dV?**
        Differential capacity analysis shows the rate of change of capacity with voltage.
        Peaks in dQ/dV plots correspond to phase transitions in the battery material,
        making this useful for degradation tracking and chemistry diagnostics.
        """
        )

        # Get available cycles for selected batteries
        try:
            max_cycle_query = f"""
                SELECT MAX(CAST(test_id AS INTEGER)) as max_cycle
                FROM cycles
                WHERE battery_id IN ({','.join([f"'{b}'" for b in selected_batteries])})
            """
            max_cycle_result = pd.read_sql_query(max_cycle_query, conn)
            max_cycle = (
                int(max_cycle_result["max_cycle"].iloc[0]) if not max_cycle_result.empty else 100
            )
            max_cycle = max(max_cycle, 1)
        except Exception:
            max_cycle = 100

        # ====== UI CONTROLS ======
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_cycles = st.multiselect(
                "Select Cycles to Analyze",
                options=list(range(0, max_cycle + 1)),
                default=[0, int(max_cycle / 2), max_cycle] if max_cycle > 0 else [0],
                help="Select specific cycles for dQ/dV analysis",
            )
        with col2:
            analysis_type = st.radio(
                "Analysis Type",
                ["Charge", "Discharge", "Both"],
                help="Choose charge, discharge, or both",
            )

        # Advanced processing options
        with st.expander("Advanced Processing Options"):
            col1, col2, col3 = st.columns(3)

            with col1:
                smooth = st.checkbox("Apply Savitzky-Golay smoothing", value=True)
                if smooth:
                    window = st.slider("Smoothing window (odd)", 5, 51, 11, step=2)
                    poly = st.slider("Polynomial order", 1, 5, 3)

            with col2:
                remove_cv = st.checkbox("Remove CV region (ΔV < 1 mV)", value=True)
                outlier_threshold = st.slider("Outlier threshold (×median)", 5, 50, 10)

            with col3:
                export_csv = st.checkbox("Enable CSV export", value=False)
                show_qv_plot = st.checkbox("Show Q vs V plot", value=True)
                show_vt_plot = st.checkbox("Show V vs Time plot", value=True)

        if not selected_cycles:
            st.warning("Please select at least one cycle to analyze")
        else:
            # Storage for CSV export
            all_export_data = []

            for battery_id in selected_batteries:
                st.markdown(f"### {battery_id}")

                try:
                    # Query cycle data
                    cycles_query = f"""
                        SELECT test_id, time_seconds, voltage, current, capacity_ah
                        FROM cycles
                        WHERE battery_id = '{battery_id}'
                          AND CAST(test_id AS INTEGER) IN ({','.join(map(str, selected_cycles))})
                        ORDER BY test_id, time_seconds
                    """
                    cycles_data = pd.read_sql_query(cycles_query, conn)

                    if cycles_data.empty:
                        st.warning(f"No data found for {battery_id} in selected cycles")
                        continue

                    # Process each cycle
                    fig_dqdv = go.Figure()
                    fig_qv = go.Figure()
                    fig_vt = go.Figure()

                    colors = px.colors.qualitative.Plotly

                    for idx, cycle in enumerate(selected_cycles):
                        cycle_df = cycles_data[cycles_data["test_id"] == cycle].copy()

                        if cycle_df.empty:
                            continue

                        # Separate charge and discharge based on current
                        if analysis_type in ["Charge", "Both"]:
                            df_charge = cycle_df[cycle_df["current"] > 0].copy()

                            if len(df_charge) > 5:
                                # Sort by voltage to ensure monotonic
                                df_charge = df_charge.sort_values("voltage")

                                # Get voltage and capacity
                                V = df_charge["voltage"].values
                                # Use cumulative capacity if available, else derive from current
                                if (
                                    "capacity_ah" in df_charge.columns
                                    and not df_charge["capacity_ah"].isna().all()
                                ):
                                    Q = df_charge["capacity_ah"].values * 1000  # Convert to mAh
                                else:
                                    # Estimate capacity from current and time
                                    Q = np.cumsum(
                                        df_charge["current"].values
                                        * np.diff(np.insert(df_charge["time_seconds"].values, 0, 0))
                                        / 3600
                                        * 1000
                                    )

                                # Compute raw dQ/dV
                                dV = np.diff(V)
                                dQ = np.diff(Q)

                                # Remove CV region (very small ΔV)
                                if remove_cv:
                                    cv_mask = np.abs(dV) > 0.001  # ΔV > 1 mV
                                    dV = dV[cv_mask]
                                    dQ = dQ[cv_mask]
                                    V_mid = ((V[:-1] + V[1:]) / 2)[cv_mask]
                                else:
                                    V_mid = (V[:-1] + V[1:]) / 2

                                # Compute dQ/dV
                                mask = (dV != 0) & np.isfinite(dV) & np.isfinite(dQ)
                                dV_clean = dV[mask]
                                dQ_clean = dQ[mask]
                                V_mid_clean = V_mid[mask] if remove_cv else V_mid[mask]

                                if len(dV_clean) > 0:
                                    dQdV_raw = dQ_clean / dV_clean

                                    # Filter outliers
                                    median_dqdv = np.median(np.abs(dQdV_raw))
                                    outlier_mask = (
                                        np.abs(dQdV_raw) < median_dqdv * outlier_threshold
                                    )
                                    dQdV_filtered = dQdV_raw[outlier_mask]
                                    V_filtered = V_mid_clean[outlier_mask]

                                    # Apply Savitzky-Golay smoothing
                                    if smooth and len(dQdV_filtered) >= window:
                                        try:
                                            dQdV_final = savgol_filter(
                                                dQdV_filtered,
                                                window_length=window,
                                                polyorder=poly,
                                            )
                                        except Exception:
                                            dQdV_final = dQdV_filtered
                                    else:
                                        dQdV_final = dQdV_filtered

                                    # Add to dQ/dV plot
                                    fig_dqdv.add_trace(
                                        go.Scatter(
                                            x=V_filtered,
                                            y=dQdV_final,
                                            mode="lines",
                                            name=f"Cycle {cycle} - Charge",
                                            line={"width": 2, "color": colors[idx % len(colors)]},
                                            legendgroup=f"cycle_{cycle}",
                                            hovertemplate="V: %{x:.3f}V<br>dQ/dV: %{y:.1f} mAh/V",
                                        )
                                    )

                                    # Store for CSV export
                                    if export_csv:
                                        export_df = pd.DataFrame(
                                            {
                                                "battery_id": battery_id,
                                                "cycle": cycle,
                                                "mode": "Charge",
                                                "V_mid": V_filtered,
                                                "dQdV": dQdV_final,
                                            }
                                        )
                                        all_export_data.append(export_df)

                                    # Add Q vs V
                                    if show_qv_plot:
                                        fig_qv.add_trace(
                                            go.Scatter(
                                                x=V,
                                                y=Q,
                                                mode="lines",
                                                name=f"Cycle {cycle} - Charge",
                                                line={
                                                    "width": 2,
                                                    "color": colors[idx % len(colors)],
                                                    "dash": "solid",
                                                },
                                                legendgroup=f"cycle_{cycle}",
                                            )
                                        )

                        if analysis_type in ["Discharge", "Both"]:
                            df_discharge = cycle_df[cycle_df["current"] < 0].copy()

                            if len(df_discharge) > 5:
                                # Sort by voltage descending for discharge
                                df_discharge = df_discharge.sort_values("voltage", ascending=False)

                                # Get voltage and capacity
                                V = df_discharge["voltage"].values
                                if (
                                    "capacity_ah" in df_discharge.columns
                                    and not df_discharge["capacity_ah"].isna().all()
                                ):
                                    Q = df_discharge["capacity_ah"].values * 1000
                                else:
                                    Q = np.cumsum(
                                        np.abs(df_discharge["current"].values)
                                        * np.diff(
                                            np.insert(
                                                df_discharge["time_seconds"].values,
                                                0,
                                                0,
                                            )
                                        )
                                        / 3600
                                        * 1000
                                    )

                                # Compute raw dQ/dV
                                dV = np.diff(V)
                                dQ = np.diff(Q)

                                # Remove CV region (very small ΔV)
                                if remove_cv:
                                    cv_mask = np.abs(dV) > 0.001  # ΔV > 1 mV
                                    dV = dV[cv_mask]
                                    dQ = dQ[cv_mask]
                                    V_mid = ((V[:-1] + V[1:]) / 2)[cv_mask]
                                else:
                                    V_mid = (V[:-1] + V[1:]) / 2

                                # Compute dQ/dV
                                mask = (dV != 0) & np.isfinite(dV) & np.isfinite(dQ)
                                dV_clean = dV[mask]
                                dQ_clean = dQ[mask]
                                V_mid_clean = V_mid[mask] if remove_cv else V_mid[mask]

                                if len(dV_clean) > 0:
                                    dQdV_raw = dQ_clean / dV_clean

                                    # Filter outliers
                                    median_dqdv = np.median(np.abs(dQdV_raw))
                                    outlier_mask = (
                                        np.abs(dQdV_raw) < median_dqdv * outlier_threshold
                                    )
                                    dQdV_filtered = np.abs(dQdV_raw[outlier_mask])
                                    V_filtered = V_mid_clean[outlier_mask]

                                    # Apply Savitzky-Golay smoothing
                                    if smooth and len(dQdV_filtered) >= window:
                                        try:
                                            dQdV_final = savgol_filter(
                                                dQdV_filtered,
                                                window_length=window,
                                                polyorder=poly,
                                            )
                                        except Exception:
                                            dQdV_final = dQdV_filtered
                                    else:
                                        dQdV_final = dQdV_filtered

                                    # Add to dQ/dV plot
                                    fig_dqdv.add_trace(
                                        go.Scatter(
                                            x=V_filtered,
                                            y=dQdV_final,
                                            mode="lines",
                                            name=f"Cycle {cycle} - Discharge",
                                            line={
                                                "width": 2,
                                                "color": colors[idx % len(colors)],
                                                "dash": "dash",
                                            },
                                            legendgroup=f"cycle_{cycle}",
                                            hovertemplate="V: %{x:.3f}V<br>dQ/dV: %{y:.1f} mAh/V",
                                        )
                                    )

                                    # Store for CSV export
                                    if export_csv:
                                        export_df = pd.DataFrame(
                                            {
                                                "battery_id": battery_id,
                                                "cycle": cycle,
                                                "mode": "Discharge",
                                                "V_mid": V_filtered,
                                                "dQdV": dQdV_final,
                                            }
                                        )
                                        all_export_data.append(export_df)

                                    # Add Q vs V
                                    if show_qv_plot:
                                        fig_qv.add_trace(
                                            go.Scatter(
                                                x=V,
                                                y=Q,
                                                mode="lines",
                                                name=f"Cycle {cycle} - Discharge",
                                                line={
                                                    "width": 2,
                                                    "color": colors[idx % len(colors)],
                                                    "dash": "dash",
                                                },
                                                legendgroup=f"cycle_{cycle}",
                                            )
                                        )

                        # Add voltage vs time
                        if show_vt_plot:
                            fig_vt.add_trace(
                                go.Scatter(
                                    x=cycle_df["time_seconds"],
                                    y=cycle_df["voltage"],
                                    mode="lines",
                                    name=f"Cycle {cycle}",
                                    line={"width": 2, "color": colors[idx % len(colors)]},
                                    legendgroup=f"cycle_{cycle}",
                                )
                            )

                    # Update layouts and display
                    if fig_dqdv.data:
                        fig_dqdv.update_layout(
                            title=f"{battery_id} - dQ/dV vs Voltage",
                            xaxis_title="Voltage (V)",
                            yaxis_title="dQ/dV (mAh/V)",
                            height=500,
                            showlegend=True,
                            template="plotly_white",
                            hovermode="x unified",
                            legend={
                                "orientation": "v",
                                "yanchor": "top",
                                "y": 1,
                                "xanchor": "right",
                                "x": 1,
                            },
                        )
                        st.plotly_chart(fig_dqdv, use_container_width=True)

                        # Show secondary plots if enabled
                        if show_qv_plot or show_vt_plot:
                            col1, col2 = st.columns(2)

                            with col1:
                                if show_qv_plot and fig_qv.data:
                                    fig_qv.update_layout(
                                        title="Capacity vs Voltage",
                                        xaxis_title="Voltage (V)",
                                        yaxis_title="Capacity (mAh)",
                                        height=400,
                                        showlegend=True,
                                        template="plotly_white",
                                    )
                                    st.plotly_chart(fig_qv, use_container_width=True)

                            with col2:
                                if show_vt_plot and fig_vt.data:
                                    fig_vt.update_layout(
                                        title="Voltage vs Time",
                                        xaxis_title="Time (seconds)",
                                        yaxis_title="Voltage (V)",
                                        height=400,
                                        showlegend=True,
                                        template="plotly_white",
                                    )
                                    st.plotly_chart(fig_vt, use_container_width=True)

                        # Add interpretation guide
                        with st.expander("How to Interpret dQ/dV Plots"):
                            st.markdown(
                                """
                            **Understanding dQ/dV Analysis:**

                            - **Peaks**: Represent phase transitions in the electrode materials
                            - **Peak Position**: Changes indicate structural changes or degradation
                            - **Peak Height**: Decreases suggest loss of active material
                            - **Peak Broadening**: Indicates increased resistance or non-uniformity
                            - **New Peaks**: May indicate formation of new phases or degradation products

                            **Solid Line**: Charge process
                            **Dashed Line**: Discharge process

                            **Typical Features:**
                            - NMC/LCO: Multiple peaks around 3.7-4.0V
                            - LFP: Sharp peak around 3.4V
                            - Graphite (anode): Peaks around 0.1-0.2V

                            **Processing Options:**
                            - **Savitzky-Golay Smoothing**: Reduces noise while preserving peak shapes
                            - **CV Region Removal**: Filters out constant voltage (CV) steps where ΔV < 1mV
                            - **Outlier Filtering**: Removes spikes that exceed threshold × median value
                            """
                            )
                    else:
                        st.warning(
                            f"Could not generate dQ/dV plot for {battery_id}. Data may not have sufficient voltage variation."
                        )

                except Exception as e:
                    st.error(f"Error analyzing {battery_id}: {str(e)}")
                    import traceback

                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())

                st.markdown("---")

            # CSV Export (after all batteries processed)
            if export_csv and all_export_data:
                st.markdown("### Export Processed dQ/dV Data")
                export_df = pd.concat(all_export_data, ignore_index=True)

                # Show preview
                st.dataframe(export_df.head(20), use_container_width=True)
                st.caption(f"Total rows: {len(export_df)}")

                # Download button
                csv_data = export_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download processed dQ/dV CSV",
                    data=csv_data,
                    file_name=f"dQdV_processed_{'-'.join(selected_batteries)}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

    # === COMPARATIVE ANALYSIS ===
    elif viz_type == "Comparative Analysis":
        st.subheader("Comparative Analysis")

        if len(selected_batteries) < 2:
            st.warning("Please select at least 2 batteries for comparative analysis")
        else:
            # Collect all data
            comparison_data = []
            for battery_id in selected_batteries:
                capacity_df, resistance_df = load_battery_data(battery_id)
                if not capacity_df.empty:
                    comparison_data.append(
                        {
                            "Battery": battery_id,
                            "Total Cycles": len(capacity_df),
                            "Initial Capacity (Ah)": capacity_df["discharge_capacity"].iloc[0],
                            "Final Capacity (Ah)": capacity_df["discharge_capacity"].iloc[-1],
                            "Capacity Fade (%)": (
                                (
                                    capacity_df["discharge_capacity"].iloc[0]
                                    - capacity_df["discharge_capacity"].iloc[-1]
                                )
                                / capacity_df["discharge_capacity"].iloc[0]
                                * 100
                            ),
                            "Avg Re (Ω)": (
                                resistance_df["Re"].mean()
                                if not resistance_df.empty and "Re" in resistance_df.columns
                                else 0
                            ),
                        }
                    )

            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)

            # Bar charts
            col1, col2 = st.columns(2)

            with col1:
                fig_cycles = px.bar(
                    comparison_df,
                    x="Battery",
                    y="Total Cycles",
                    title="Total Cycles Comparison",
                    color="Total Cycles",
                    color_continuous_scale="Blues",
                )
                st.plotly_chart(fig_cycles, use_container_width=True)

            with col2:
                fig_fade = px.bar(
                    comparison_df,
                    x="Battery",
                    y="Capacity Fade (%)",
                    title="Capacity Fade Comparison",
                    color="Capacity Fade (%)",
                    color_continuous_scale="Reds",
                )
                st.plotly_chart(fig_fade, use_container_width=True)

    else:
        st.info(f"Visualization type '{viz_type}' is coming soon!")

# Export options
st.markdown("---")
st.subheader("Export Options")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Export Chart as PNG", use_container_width=True):
        st.info("Chart export functionality coming soon!")
with col2:
    if st.button("Export Data as CSV", use_container_width=True):
        st.info("Data export functionality coming soon!")
with col3:
    if st.button("Generate Report", use_container_width=True):
        st.info("Report generation coming soon!")
