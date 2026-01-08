"""
Data Explorer Page
==================
Query, filter, and explore battery testing data.

Features:
- SQL query interface
- Data filtering
- Column selection
- Data export
- Statistics and summaries
"""

import sqlite3
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Page config
st.set_page_config(page_title="Data Explorer - AmpereData", page_icon=None, layout="wide")

# Title
st.title("Data Explorer")
st.markdown("Query and explore your battery testing datasets.")

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


# Get list of tables
@st.cache_data(ttl=60)
def get_tables():
    query = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    tables = pd.read_sql_query(query, conn)
    return tables["name"].tolist()


tables = get_tables()

# Sidebar
with st.sidebar:
    st.subheader("Database Tables")
    selected_table = st.selectbox("Select Table", tables)

    st.markdown("---")

    st.subheader("Quick Stats")
    if selected_table:
        try:
            count_query = f"SELECT COUNT(*) as count FROM {selected_table}"
            count_df = pd.read_sql_query(count_query, conn)
            st.metric("Total Rows", f"{count_df['count'].iloc[0]:,}")

            # Get column info
            col_query = f"PRAGMA table_info({selected_table})"
            col_df = pd.read_sql_query(col_query, conn)
            st.metric("Total Columns", len(col_df))
        except Exception as e:
            st.error(f"Error: {e}")

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Browse Data", "SQL Query", "Analytics", "Export", "Manage Batteries"]
)

# === BROWSE DATA TAB ===
with tab1:
    st.subheader(f"Browse: {selected_table}")

    # Get table schema
    try:
        schema_query = f"PRAGMA table_info({selected_table})"
        schema_df = pd.read_sql_query(schema_query, conn)

        # Column selection
        all_columns = schema_df["name"].tolist()
        selected_columns = st.multiselect(
            "Select Columns to Display",
            all_columns,
            default=all_columns[:10] if len(all_columns) > 10 else all_columns,
        )

        if not selected_columns:
            st.warning("Please select at least one column")
        else:
            # Filters
            with st.expander("Filters"):
                col1, col2 = st.columns(2)
                with col1:
                    limit = st.number_input(
                        "Rows to display",
                        min_value=10,
                        max_value=10000,
                        value=100,
                        step=10,
                    )
                with col2:
                    offset = st.number_input(
                        "Start from row",
                        min_value=0,
                        max_value=1000000,
                        value=0,
                        step=100,
                    )

            # Load data
            columns_str = ", ".join(selected_columns)
            query = f"SELECT {columns_str} FROM {selected_table} LIMIT {limit} OFFSET {offset}"

            with st.spinner("Loading data..."):
                df = pd.read_sql_query(query, conn)

            # Display data
            st.dataframe(df, use_container_width=True, height=500)

            # Data info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows Displayed", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
            with col4:
                st.metric("Missing Values", df.isnull().sum().sum())

    except Exception as e:
        st.error(f"Error loading data: {e}")

# === SQL QUERY TAB ===
with tab2:
    st.subheader("SQL Query Interface")

    # Query examples
    with st.expander("Example Queries"):
        st.code(
            """
-- Get all batteries
SELECT * FROM batteries LIMIT 10;

-- Count cycles per battery
SELECT battery_id, COUNT(*) as cycle_count
FROM cycle_statistics
GROUP BY battery_id
ORDER BY cycle_count DESC;

-- Average capacity by battery
SELECT battery_id, AVG(discharge_capacity) as avg_capacity
FROM cycle_statistics
GROUP BY battery_id;

-- Recent experiments
SELECT * FROM experiments
WHERE start_date > date('now', '-30 days')
ORDER BY start_date DESC;
        """,
            language="sql",
        )

    # Query input
    default_query = f"SELECT * FROM {selected_table} LIMIT 100"
    user_query = st.text_area(
        "Enter SQL Query",
        value=default_query,
        height=150,
        help="Write your SQL query here. Be careful with UPDATE/DELETE operations.",
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        execute_button = st.button("Execute", type="primary", use_container_width=True)
    with col2:
        if st.button("Clear", use_container_width=True):
            st.rerun()

    if execute_button:
        try:
            # Safety check for destructive operations
            if any(
                keyword in user_query.upper() for keyword in ["DROP", "DELETE", "TRUNCATE", "ALTER"]
            ):
                st.error(
                    "Destructive operations (DROP, DELETE, TRUNCATE, ALTER) are not allowed in the UI."
                )
            else:
                with st.spinner("Executing query..."):
                    result_df = pd.read_sql_query(user_query, conn)

                st.success(f"Query executed successfully. Returned {len(result_df)} rows.")

                # Display results
                st.dataframe(result_df, use_container_width=True, height=400)

                # Result stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", len(result_df))
                with col2:
                    st.metric("Columns", len(result_df.columns))
                with col3:
                    st.metric(
                        "Size",
                        f"{result_df.memory_usage(deep=True).sum() / 1024:.1f} KB",
                    )

                # Quick visualizations
                if len(result_df) > 0:
                    st.markdown("---")
                    st.subheader("Quick Visualization")

                    # Get numeric columns
                    numeric_cols = result_df.select_dtypes(include=["number"]).columns.tolist()

                    if numeric_cols:
                        col1, col2 = st.columns(2)
                        with col1:
                            x_col = st.selectbox("X-axis", result_df.columns.tolist(), index=0)
                        with col2:
                            y_col = st.selectbox(
                                "Y-axis",
                                numeric_cols,
                                index=0 if numeric_cols else None,
                            )

                        if x_col and y_col:
                            fig = px.scatter(
                                result_df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}"
                            )
                            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Query Error: {e}")
            st.exception(e)

# === ANALYTICS TAB ===
with tab3:
    st.subheader("Data Analytics")

    try:
        # Load data for analytics
        query = f"SELECT * FROM {selected_table} LIMIT 1000"
        df = pd.read_sql_query(query, conn)

        if df.empty:
            st.warning("No data available for analytics")
        else:
            # Summary statistics
            st.markdown("### Summary Statistics")
            st.dataframe(df.describe(), use_container_width=True)

            # Data types
            st.markdown("### Column Data Types")
            types_df = pd.DataFrame(
                {
                    "Column": df.columns,
                    "Type": df.dtypes.astype(str),
                    "Non-Null": df.count(),
                    "Null": df.isnull().sum(),
                    "Unique": df.nunique(),
                }
            )
            st.dataframe(types_df, use_container_width=True)

            # Correlations for numeric columns
            numeric_df = df.select_dtypes(include=["number"])
            if not numeric_df.empty and len(numeric_df.columns) > 1:
                st.markdown("### Correlation Matrix")
                corr_matrix = numeric_df.corr()
                fig = px.imshow(
                    corr_matrix,
                    labels=dict(color="Correlation"),
                    color_continuous_scale="RdBu_r",
                    aspect="auto",
                    title="Feature Correlations",
                )
                st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error loading analytics: {e}")

# === EXPORT TAB ===
with tab4:
    st.subheader("Export Data")

    st.markdown("Export data from the selected table in various formats.")

    # Export options
    col1, col2 = st.columns(2)

    with col1:
        export_limit = st.number_input(
            "Number of rows to export",
            min_value=1,
            max_value=1000000,
            value=1000,
            step=100,
        )

    with col2:
        export_format = st.selectbox("Export Format", ["CSV", "JSON", "Excel", "Parquet"])

    # Column selection for export
    schema_query = f"PRAGMA table_info({selected_table})"
    schema_df = pd.read_sql_query(schema_query, conn)
    all_columns = schema_df["name"].tolist()

    export_columns = st.multiselect("Select columns to export", all_columns, default=all_columns)

    if st.button("Prepare Export", type="primary", use_container_width=True):
        try:
            # Load data
            columns_str = ", ".join(export_columns) if export_columns else "*"
            query = f"SELECT {columns_str} FROM {selected_table} LIMIT {export_limit}"

            with st.spinner("Preparing export..."):
                export_df = pd.read_sql_query(query, conn)

            st.success(f"Prepared {len(export_df)} rows for export")

            # Preview
            st.markdown("### Preview")
            st.dataframe(export_df.head(10), use_container_width=True)

            # Download buttons
            st.markdown("### Download")

            col1, col2, col3 = st.columns(3)

            with col1:
                csv_data = export_df.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    csv_data,
                    f"{selected_table}_export.csv",
                    "text/csv",
                    use_container_width=True,
                )

            with col2:
                json_data = export_df.to_json(orient="records", indent=2)
                st.download_button(
                    "Download JSON",
                    json_data,
                    f"{selected_table}_export.json",
                    "application/json",
                    use_container_width=True,
                )

            with col3:
                # Excel export would require openpyxl
                st.button(
                    "Download Excel",
                    disabled=True,
                    use_container_width=True,
                    help="Excel export coming soon",
                )

        except Exception as e:
            st.error(f"Export error: {e}")

# === MANAGE BATTERIES TAB ===
with tab5:
    st.subheader("Battery Management")

    # Load all batteries
    @st.cache_data(ttl=10)
    def load_all_batteries():
        try:
            query = """
                SELECT
                    b.battery_id,
                    b.manufacturer,
                    b.model,
                    bm.build_id,
                    bm.serial_number,
                    bm.status,
                    bs.build_name,
                    bs.build_type
                FROM batteries b
                LEFT JOIN battery_metadata bm ON b.battery_id = bm.battery_id
                LEFT JOIN build_sheets bs ON bm.build_id = bs.build_id
                ORDER BY b.battery_id
            """
            return pd.read_sql_query(query, conn)
        except Exception:
            return pd.read_sql_query(
                "SELECT DISTINCT battery_id FROM batteries ORDER BY battery_id", conn
            )

    batteries_df = load_all_batteries()

    if batteries_df.empty:
        st.warning("No batteries found in database.")
    else:
        # Action selector
        action = st.radio(
            "Select Action",
            ["Rename Battery", "Delete Battery", "Batch Operations", "View Battery Info"],
            horizontal=True,
        )

        st.markdown("---")

        if action == "Rename Battery":
            st.markdown("### Rename Battery")
            st.info(
                "Rename a battery ID across all tables. This updates cycles, capacity_fade, batteries, and battery_metadata tables."
            )

            col1, col2 = st.columns(2)

            with col1:
                old_id = st.selectbox(
                    "Select Battery to Rename", batteries_df["battery_id"].tolist()
                )

            with col2:
                new_id = st.text_input("New Battery ID*", placeholder="e.g., NMC811_Cell_01")

            if st.button("Rename Battery", type="primary"):
                if not new_id:
                    st.error("Please provide a new battery ID")
                elif old_id == new_id:
                    st.warning("New ID is the same as old ID")
                else:
                    try:
                        # Check if new_id already exists
                        check_query = (
                            f"SELECT battery_id FROM batteries WHERE battery_id = '{new_id}'"
                        )
                        check_df = pd.read_sql_query(check_query, conn)

                        if not check_df.empty:
                            st.error(f"Battery ID '{new_id}' already exists!")
                        else:
                            # Perform rename across all tables
                            cursor = conn.cursor()

                            cursor.execute(
                                "UPDATE batteries SET battery_id = ? WHERE battery_id = ?",
                                (new_id, old_id),
                            )
                            cursor.execute(
                                "UPDATE cycles SET battery_id = ? WHERE battery_id = ?",
                                (new_id, old_id),
                            )
                            cursor.execute(
                                "UPDATE capacity_fade SET battery_id = ? WHERE battery_id = ?",
                                (new_id, old_id),
                            )

                            try:
                                cursor.execute(
                                    "UPDATE battery_metadata SET battery_id = ? WHERE battery_id = ?",
                                    (new_id, old_id),
                                )
                            except Exception:
                                pass  # Table might not exist

                            conn.commit()

                            st.success(f"Battery renamed from '{old_id}' to '{new_id}'")
                            st.cache_data.clear()
                            st.rerun()

                    except Exception as e:
                        st.error(f"Error renaming battery: {str(e)}")
                        conn.rollback()

        elif action == "Delete Battery":
            st.markdown("### Delete Battery")
            st.warning(
                "This will delete all data associated with the battery (cycles, capacity_fade, metadata). This action cannot be undone."
            )

            battery_to_delete = st.selectbox(
                "Select Battery to Delete", batteries_df["battery_id"].tolist()
            )

            # Show battery info
            battery_info = batteries_df[batteries_df["battery_id"] == battery_to_delete]
            if not battery_info.empty:
                st.markdown("#### Battery Information:")
                info = battery_info.iloc[0]
                st.json(
                    {
                        "Battery ID": info["battery_id"],
                        "Build": info.get("build_name", "N/A"),
                        "Serial": info.get("serial_number", "N/A"),
                        "Status": info.get("status", "N/A"),
                    }
                )

            # Count data points
            try:
                cycles_count = pd.read_sql_query(
                    f"SELECT COUNT(*) as count FROM cycles WHERE battery_id = '{battery_to_delete}'",
                    conn,
                )["count"].iloc[0]
                st.metric("Data Points in cycles table", f"{cycles_count:,}")
            except Exception:
                pass

            # Confirmation
            confirm_delete = st.checkbox(
                "I understand this will permanently delete all data for this battery"
            )

            if st.button("Delete Battery", type="secondary", disabled=not confirm_delete):
                try:
                    cursor = conn.cursor()

                    cursor.execute("DELETE FROM cycles WHERE battery_id = ?", (battery_to_delete,))
                    cursor.execute(
                        "DELETE FROM capacity_fade WHERE battery_id = ?", (battery_to_delete,)
                    )
                    cursor.execute(
                        "DELETE FROM batteries WHERE battery_id = ?", (battery_to_delete,)
                    )

                    try:
                        cursor.execute(
                            "DELETE FROM battery_metadata WHERE battery_id = ?",
                            (battery_to_delete,),
                        )
                    except Exception:
                        pass

                    conn.commit()

                    st.success(f"Battery '{battery_to_delete}' deleted successfully")
                    st.cache_data.clear()
                    st.rerun()

                except Exception as e:
                    st.error(f"Error deleting battery: {str(e)}")
                    conn.rollback()

        elif action == "Batch Operations":
            st.markdown("### Batch Operations")

            st.markdown("#### Batch Rename")
            st.info("Upload a CSV with columns: `old_id`, `new_id`")

            batch_file = st.file_uploader("Upload CSV for Batch Rename", type=["csv"])

            if batch_file:
                try:
                    rename_df = pd.read_csv(batch_file)

                    if "old_id" not in rename_df.columns or "new_id" not in rename_df.columns:
                        st.error("CSV must have 'old_id' and 'new_id' columns")
                    else:
                        st.dataframe(rename_df, use_container_width=True)

                        if st.button("Execute Batch Rename"):
                            success_count = 0
                            error_count = 0
                            errors = []

                            progress_bar = st.progress(0)
                            status_text = st.empty()

                            for idx, row in rename_df.iterrows():
                                status_text.text(f"Renaming {row['old_id']} → {row['new_id']}...")

                                try:
                                    cursor = conn.cursor()

                                    # Check if new_id exists
                                    check_query = f"SELECT battery_id FROM batteries WHERE battery_id = '{row['new_id']}'"
                                    check_df = pd.read_sql_query(check_query, conn)

                                    if not check_df.empty:
                                        errors.append(
                                            f"{row['old_id']}: New ID '{row['new_id']}' already exists"
                                        )
                                        error_count += 1
                                    else:
                                        cursor.execute(
                                            "UPDATE batteries SET battery_id = ? WHERE battery_id = ?",
                                            (row["new_id"], row["old_id"]),
                                        )
                                        cursor.execute(
                                            "UPDATE cycles SET battery_id = ? WHERE battery_id = ?",
                                            (row["new_id"], row["old_id"]),
                                        )
                                        cursor.execute(
                                            "UPDATE capacity_fade SET battery_id = ? WHERE battery_id = ?",
                                            (row["new_id"], row["old_id"]),
                                        )

                                        try:
                                            cursor.execute(
                                                "UPDATE battery_metadata SET battery_id = ? WHERE battery_id = ?",
                                                (row["new_id"], row["old_id"]),
                                            )
                                        except Exception:
                                            pass

                                        conn.commit()
                                        success_count += 1

                                except Exception as e:
                                    errors.append(f"{row['old_id']}: {str(e)}")
                                    error_count += 1
                                    conn.rollback()

                                progress_bar.progress((idx + 1) / len(rename_df))

                            status_text.text("Batch rename complete!")

                            st.success(f"Renamed {success_count} batteries ({error_count} errors)")

                            if errors:
                                with st.expander("View Errors"):
                                    for error in errors:
                                        st.error(error)

                            st.cache_data.clear()

                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")

        elif action == "View Battery Info":
            st.markdown("### Battery Information")

            selected_battery = st.selectbox("Select Battery", batteries_df["battery_id"].tolist())

            if selected_battery:
                battery_info = batteries_df[batteries_df["battery_id"] == selected_battery]

                if not battery_info.empty:
                    info = battery_info.iloc[0]

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.markdown("**Basic Information**")
                        st.markdown(f"**Battery ID:** {info['battery_id']}")
                        if pd.notna(info.get("build_name")):
                            st.markdown(f"**Build:** {info['build_name']}")
                        if pd.notna(info.get("build_type")):
                            st.markdown(f"**Type:** {info['build_type']}")
                        if pd.notna(info.get("serial_number")):
                            st.markdown(f"**Serial:** {info['serial_number']}")

                    with col2:
                        st.markdown("**Database Stats**")

                        try:
                            cycles_count = pd.read_sql_query(
                                f"SELECT COUNT(*) as count FROM cycles WHERE battery_id = '{selected_battery}'",
                                conn,
                            )["count"].iloc[0]
                            st.metric("Cycles Data Points", f"{cycles_count:,}")
                        except Exception:
                            st.metric("Cycles Data Points", "N/A")

                        try:
                            capacity_count = pd.read_sql_query(
                                f"SELECT COUNT(*) as count FROM capacity_fade WHERE battery_id = '{selected_battery}'",
                                conn,
                            )["count"].iloc[0]
                            st.metric("Capacity Records", f"{capacity_count:,}")
                        except Exception:
                            st.metric("Capacity Records", "N/A")

                    with col3:
                        st.markdown("**Quick Actions**")

                        if st.button("Visualize"):
                            st.switch_page("pages/2_Visualizations.py")

                        if st.button("Edit Build Sheet"):
                            st.switch_page("pages/6_Build_Sheet.py")

    # Link to build sheet page
    st.markdown("---")
    if st.button("Manage Build Sheets & Metadata", use_container_width=True):
        st.switch_page("pages/6_Build_Sheet.py")

# Footer
st.markdown("---")
st.caption("Tip: Use the SQL Query tab for advanced filtering and aggregations")
