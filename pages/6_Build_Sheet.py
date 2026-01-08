"""
Build Sheet Management
======================
Manage battery metadata and build configurations.

Features:
- Create and edit build sheets
- Import metadata from Excel/CSV
- Link metadata to battery test data
- Group batteries by build type
- Export build sheets
"""

import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Page config
st.set_page_config(page_title="Build Sheet Management - AmpereData", page_icon=None, layout="wide")

# Title
st.title("Build Sheet Management")
st.markdown("Manage battery metadata and build configurations.")

# Database path
DB_PATH = "nasa_amperedata_full.db"


# Initialize database tables
def init_build_sheet_tables():
    """Create build_sheets and battery_metadata tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create build_sheets table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS build_sheets (
            build_id TEXT PRIMARY KEY,
            build_name TEXT NOT NULL,
            build_type TEXT,
            cathode_material TEXT,
            anode_material TEXT,
            electrolyte TEXT,
            separator TEXT,
            nominal_capacity_ah REAL,
            nominal_voltage_v REAL,
            form_factor TEXT,
            manufacturer TEXT,
            build_date TEXT,
            notes TEXT,
            created_at TEXT,
            updated_at TEXT
        )
    """
    )

    # Create battery_metadata table (links batteries to build sheets)
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS battery_metadata (
            battery_id TEXT PRIMARY KEY,
            build_id TEXT,
            serial_number TEXT,
            position_in_build TEXT,
            assembly_date TEXT,
            first_test_date TEXT,
            status TEXT,
            notes TEXT,
            created_at TEXT,
            updated_at TEXT,
            FOREIGN KEY (battery_id) REFERENCES batteries(battery_id),
            FOREIGN KEY (build_id) REFERENCES build_sheets(build_id)
        )
    """
    )

    conn.commit()
    conn.close()


# Initialize tables
init_build_sheet_tables()


# Database functions
@st.cache_data(ttl=10)
def load_build_sheets():
    """Load all build sheets."""
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM build_sheets ORDER BY created_at DESC", conn)
        return df
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()


@st.cache_data(ttl=10)
def load_battery_metadata():
    """Load all battery metadata."""
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(
            """
            SELECT
                bm.*,
                bs.build_name,
                bs.build_type,
                bs.cathode_material,
                bs.anode_material
            FROM battery_metadata bm
            LEFT JOIN build_sheets bs ON bm.build_id = bs.build_id
            ORDER BY bm.created_at DESC
        """,
            conn,
        )
        return df
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()


@st.cache_data(ttl=10)
def load_available_batteries():
    """Load batteries that don't have metadata yet."""
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(
            """
            SELECT DISTINCT b.battery_id
            FROM batteries b
            LEFT JOIN battery_metadata bm ON b.battery_id = bm.battery_id
            WHERE bm.battery_id IS NULL
            ORDER BY b.battery_id
        """,
            conn,
        )
        return df["battery_id"].tolist()
    except Exception:
        return []
    finally:
        conn.close()


def save_build_sheet(build_data):
    """Save or update a build sheet."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # Check if build exists
        cursor.execute(
            "SELECT build_id FROM build_sheets WHERE build_id = ?", (build_data["build_id"],)
        )
        exists = cursor.fetchone()

        if exists:
            # Update existing
            cursor.execute(
                """
                UPDATE build_sheets SET
                    build_name = ?,
                    build_type = ?,
                    cathode_material = ?,
                    anode_material = ?,
                    electrolyte = ?,
                    separator = ?,
                    nominal_capacity_ah = ?,
                    nominal_voltage_v = ?,
                    form_factor = ?,
                    manufacturer = ?,
                    build_date = ?,
                    notes = ?,
                    updated_at = ?
                WHERE build_id = ?
            """,
                (
                    build_data["build_name"],
                    build_data.get("build_type"),
                    build_data.get("cathode_material"),
                    build_data.get("anode_material"),
                    build_data.get("electrolyte"),
                    build_data.get("separator"),
                    build_data.get("nominal_capacity_ah"),
                    build_data.get("nominal_voltage_v"),
                    build_data.get("form_factor"),
                    build_data.get("manufacturer"),
                    build_data.get("build_date"),
                    build_data.get("notes"),
                    datetime.now().isoformat(),
                    build_data["build_id"],
                ),
            )
        else:
            # Insert new
            cursor.execute(
                """
                INSERT INTO build_sheets (
                    build_id, build_name, build_type, cathode_material, anode_material,
                    electrolyte, separator, nominal_capacity_ah, nominal_voltage_v,
                    form_factor, manufacturer, build_date, notes, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    build_data["build_id"],
                    build_data["build_name"],
                    build_data.get("build_type"),
                    build_data.get("cathode_material"),
                    build_data.get("anode_material"),
                    build_data.get("electrolyte"),
                    build_data.get("separator"),
                    build_data.get("nominal_capacity_ah"),
                    build_data.get("nominal_voltage_v"),
                    build_data.get("form_factor"),
                    build_data.get("manufacturer"),
                    build_data.get("build_date"),
                    build_data.get("notes"),
                    datetime.now().isoformat(),
                    datetime.now().isoformat(),
                ),
            )

        conn.commit()
        return True, "Build sheet saved successfully!"
    except Exception as e:
        return False, f"Error saving build sheet: {str(e)}"
    finally:
        conn.close()


def link_battery_to_build(battery_id, metadata):
    """Link a battery to a build sheet with metadata."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # Check if metadata exists
        cursor.execute(
            "SELECT battery_id FROM battery_metadata WHERE battery_id = ?", (battery_id,)
        )
        exists = cursor.fetchone()

        if exists:
            # Update existing
            cursor.execute(
                """
                UPDATE battery_metadata SET
                    build_id = ?,
                    serial_number = ?,
                    position_in_build = ?,
                    assembly_date = ?,
                    first_test_date = ?,
                    status = ?,
                    notes = ?,
                    updated_at = ?
                WHERE battery_id = ?
            """,
                (
                    metadata.get("build_id"),
                    metadata.get("serial_number"),
                    metadata.get("position_in_build"),
                    metadata.get("assembly_date"),
                    metadata.get("first_test_date"),
                    metadata.get("status", "Active"),
                    metadata.get("notes"),
                    datetime.now().isoformat(),
                    battery_id,
                ),
            )
        else:
            # Insert new
            cursor.execute(
                """
                INSERT INTO battery_metadata (
                    battery_id, build_id, serial_number, position_in_build,
                    assembly_date, first_test_date, status, notes, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    battery_id,
                    metadata.get("build_id"),
                    metadata.get("serial_number"),
                    metadata.get("position_in_build"),
                    metadata.get("assembly_date"),
                    metadata.get("first_test_date"),
                    metadata.get("status", "Active"),
                    metadata.get("notes"),
                    datetime.now().isoformat(),
                    datetime.now().isoformat(),
                ),
            )

        conn.commit()
        return True, "Battery metadata saved successfully!"
    except Exception as e:
        return False, f"Error saving battery metadata: {str(e)}"
    finally:
        conn.close()


def rename_battery(old_id, new_id):
    """Rename a battery across all tables."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # Check if new_id already exists
        cursor.execute("SELECT battery_id FROM batteries WHERE battery_id = ?", (new_id,))
        if cursor.fetchone():
            return False, f"Battery ID '{new_id}' already exists!"

        # Update in all tables
        cursor.execute("UPDATE batteries SET battery_id = ? WHERE battery_id = ?", (new_id, old_id))
        cursor.execute("UPDATE cycles SET battery_id = ? WHERE battery_id = ?", (new_id, old_id))
        cursor.execute(
            "UPDATE capacity_fade SET battery_id = ? WHERE battery_id = ?", (new_id, old_id)
        )
        cursor.execute(
            "UPDATE battery_metadata SET battery_id = ? WHERE battery_id = ?", (new_id, old_id)
        )

        conn.commit()
        return True, f"Battery renamed from '{old_id}' to '{new_id}'"
    except Exception as e:
        conn.rollback()
        return False, f"Error renaming battery: {str(e)}"
    finally:
        conn.close()


# Sidebar navigation
with st.sidebar:
    st.subheader("Build Sheet Tools")
    page = st.radio(
        "Select Action",
        [
            "View Build Sheets",
            "Create New Build Sheet",
            "Link Batteries to Build",
            "Import from Excel",
            "Rename Batteries",
            "View Battery Groups",
        ],
    )

# Clear cache button
if st.sidebar.button("Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# Main content based on selected page
if page == "View Build Sheets":
    st.subheader("Existing Build Sheets")

    build_sheets = load_build_sheets()

    if build_sheets.empty:
        st.info("No build sheets found. Create one to get started!")
    else:
        # Show build sheets table
        st.dataframe(
            build_sheets[
                [
                    "build_id",
                    "build_name",
                    "build_type",
                    "cathode_material",
                    "anode_material",
                    "nominal_capacity_ah",
                    "created_at",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        # Select build sheet to view details
        st.markdown("---")
        selected_build = st.selectbox(
            "Select Build Sheet for Details", build_sheets["build_id"].tolist()
        )

        if selected_build:
            build_data = build_sheets[build_sheets["build_id"] == selected_build].iloc[0]

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Build Information")
                st.markdown(f"**Build ID:** {build_data['build_id']}")
                st.markdown(f"**Build Name:** {build_data['build_name']}")
                st.markdown(f"**Build Type:** {build_data.get('build_type', 'N/A')}")
                st.markdown(f"**Cathode:** {build_data.get('cathode_material', 'N/A')}")
                st.markdown(f"**Anode:** {build_data.get('anode_material', 'N/A')}")
                st.markdown(f"**Electrolyte:** {build_data.get('electrolyte', 'N/A')}")
                st.markdown(f"**Separator:** {build_data.get('separator', 'N/A')}")

            with col2:
                st.markdown("### Specifications")
                st.markdown(
                    f"**Nominal Capacity:** {build_data.get('nominal_capacity_ah', 'N/A')} Ah"
                )
                st.markdown(f"**Nominal Voltage:** {build_data.get('nominal_voltage_v', 'N/A')} V")
                st.markdown(f"**Form Factor:** {build_data.get('form_factor', 'N/A')}")
                st.markdown(f"**Manufacturer:** {build_data.get('manufacturer', 'N/A')}")
                st.markdown(f"**Build Date:** {build_data.get('build_date', 'N/A')}")

            if build_data.get("notes"):
                st.markdown("### Notes")
                st.info(build_data["notes"])

            # Show batteries linked to this build
            st.markdown("---")
            st.markdown("### Linked Batteries")
            metadata = load_battery_metadata()
            linked = metadata[metadata["build_id"] == selected_build]

            if linked.empty:
                st.warning("No batteries linked to this build sheet yet.")
            else:
                st.dataframe(
                    linked[["battery_id", "serial_number", "status", "assembly_date"]],
                    use_container_width=True,
                    hide_index=True,
                )

elif page == "Create New Build Sheet":
    st.subheader("Create New Build Sheet")

    with st.form("new_build_sheet"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Basic Information")
            build_id = st.text_input(
                "Build ID*", placeholder="e.g., BUILD_001", help="Unique identifier"
            )
            build_name = st.text_input("Build Name*", placeholder="e.g., NMC811 Prototype v1")
            build_type = st.selectbox(
                "Build Type", ["Prototype", "Production", "Research", "Validation", "Other"]
            )
            manufacturer = st.text_input("Manufacturer", placeholder="e.g., ACME Batteries")
            build_date = st.date_input("Build Date")

        with col2:
            st.markdown("#### Materials")
            cathode = st.text_input("Cathode Material", placeholder="e.g., NMC811")
            anode = st.text_input("Anode Material", placeholder="e.g., Graphite")
            electrolyte = st.text_input("Electrolyte", placeholder="e.g., 1M LiPF6 in EC:DMC")
            separator = st.text_input("Separator", placeholder="e.g., Celgard 2325")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Specifications")
            nominal_capacity = st.number_input("Nominal Capacity (Ah)", min_value=0.0, step=0.1)
            nominal_voltage = st.number_input("Nominal Voltage (V)", min_value=0.0, step=0.1)

        with col2:
            st.markdown("#### Physical")
            form_factor = st.selectbox(
                "Form Factor",
                [
                    "Cylindrical (18650)",
                    "Cylindrical (21700)",
                    "Pouch",
                    "Prismatic",
                    "Coin Cell",
                    "Other",
                ],
            )

        notes = st.text_area("Notes (optional)", placeholder="Additional information...")

        submitted = st.form_submit_button("Create Build Sheet", use_container_width=True)

        if submitted:
            if not build_id or not build_name:
                st.error("Please provide Build ID and Build Name")
            else:
                build_data = {
                    "build_id": build_id,
                    "build_name": build_name,
                    "build_type": build_type,
                    "cathode_material": cathode if cathode else None,
                    "anode_material": anode if anode else None,
                    "electrolyte": electrolyte if electrolyte else None,
                    "separator": separator if separator else None,
                    "nominal_capacity_ah": nominal_capacity if nominal_capacity > 0 else None,
                    "nominal_voltage_v": nominal_voltage if nominal_voltage > 0 else None,
                    "form_factor": form_factor,
                    "manufacturer": manufacturer if manufacturer else None,
                    "build_date": build_date.isoformat() if build_date else None,
                    "notes": notes if notes else None,
                }

                success, message = save_build_sheet(build_data)
                if success:
                    st.success(message)
                    st.cache_data.clear()
                    st.balloons()
                else:
                    st.error(message)

elif page == "Link Batteries to Build":
    st.subheader("Link Batteries to Build Sheet")

    # Load data
    build_sheets = load_build_sheets()
    available_batteries = load_available_batteries()

    if build_sheets.empty:
        st.warning("No build sheets available. Please create one first.")
    elif not available_batteries:
        st.info("All batteries are already linked to build sheets.")
    else:
        with st.form("link_battery"):
            col1, col2 = st.columns(2)

            with col1:
                battery_id = st.selectbox("Select Battery*", available_batteries)
                build_id = st.selectbox("Link to Build Sheet*", build_sheets["build_id"].tolist())

            with col2:
                serial_number = st.text_input("Serial Number", placeholder="e.g., SN-12345")
                position = st.text_input("Position in Build", placeholder="e.g., Cell #1")

            col1, col2 = st.columns(2)

            with col1:
                assembly_date = st.date_input("Assembly Date")
                first_test_date = st.date_input("First Test Date")

            with col2:
                status = st.selectbox("Status", ["Active", "Testing", "Retired", "Failed"])

            notes = st.text_area(
                "Notes", placeholder="Additional information about this battery..."
            )

            submitted = st.form_submit_button("Link Battery", use_container_width=True)

            if submitted:
                metadata = {
                    "build_id": build_id,
                    "serial_number": serial_number if serial_number else None,
                    "position_in_build": position if position else None,
                    "assembly_date": assembly_date.isoformat() if assembly_date else None,
                    "first_test_date": first_test_date.isoformat() if first_test_date else None,
                    "status": status,
                    "notes": notes if notes else None,
                }

                success, message = link_battery_to_build(battery_id, metadata)
                if success:
                    st.success(message)
                    st.cache_data.clear()
                else:
                    st.error(message)

elif page == "Import from Excel":
    st.subheader("Import Build Sheet from Excel")

    st.markdown(
        """
    **Import Format:**

    Upload an Excel file with one or both of these sheets:

    1. **build_sheets** - Build sheet information
       - Columns: build_id, build_name, build_type, cathode_material, anode_material, etc.

    2. **battery_metadata** - Battery-to-build linkage
       - Columns: battery_id, build_id, serial_number, status, etc.
    """
    )

    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx", "xls"])

    if uploaded_file:
        try:
            # Read Excel file
            excel_file = pd.ExcelFile(uploaded_file)
            sheet_names = excel_file.sheet_names

            st.success(f"Found {len(sheet_names)} sheets: {', '.join(sheet_names)}")

            # Process build_sheets
            if "build_sheets" in sheet_names:
                st.markdown("### Build Sheets Import")
                builds_df = pd.read_excel(uploaded_file, sheet_name="build_sheets")
                st.dataframe(builds_df, use_container_width=True)

                if st.button("Import Build Sheets"):
                    success_count = 0
                    error_count = 0

                    for _, row in builds_df.iterrows():
                        build_data = row.to_dict()
                        success, _ = save_build_sheet(build_data)
                        if success:
                            success_count += 1
                        else:
                            error_count += 1

                    st.success(f"Imported {success_count} build sheets ({error_count} errors)")
                    st.cache_data.clear()

            # Process battery_metadata
            if "battery_metadata" in sheet_names:
                st.markdown("### Battery Metadata Import")
                metadata_df = pd.read_excel(uploaded_file, sheet_name="battery_metadata")
                st.dataframe(metadata_df, use_container_width=True)

                if st.button("Import Battery Metadata"):
                    success_count = 0
                    error_count = 0

                    for _, row in metadata_df.iterrows():
                        battery_id = row["battery_id"]
                        metadata = row.to_dict()
                        success, _ = link_battery_to_build(battery_id, metadata)
                        if success:
                            success_count += 1
                        else:
                            error_count += 1

                    st.success(
                        f"Imported {success_count} battery metadata records ({error_count} errors)"
                    )
                    st.cache_data.clear()

        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            import traceback

            with st.expander("Error Details"):
                st.code(traceback.format_exc())

    # Download template
    st.markdown("---")
    st.markdown("### Download Template")

    # Create template Excel
    template_builds = pd.DataFrame(
        {
            "build_id": ["BUILD_001"],
            "build_name": ["Example Build"],
            "build_type": ["Prototype"],
            "cathode_material": ["NMC811"],
            "anode_material": ["Graphite"],
            "electrolyte": ["1M LiPF6"],
            "separator": ["Celgard"],
            "nominal_capacity_ah": [2.5],
            "nominal_voltage_v": [3.7],
            "form_factor": ["18650"],
            "manufacturer": ["ACME"],
            "build_date": ["2024-01-01"],
            "notes": ["Example build sheet"],
        }
    )

    template_metadata = pd.DataFrame(
        {
            "battery_id": ["CELL_001"],
            "build_id": ["BUILD_001"],
            "serial_number": ["SN-12345"],
            "position_in_build": ["Cell #1"],
            "assembly_date": ["2024-01-01"],
            "first_test_date": ["2024-01-02"],
            "status": ["Active"],
            "notes": ["Example battery"],
        }
    )

    # Create Excel writer
    import io

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        template_builds.to_excel(writer, sheet_name="build_sheets", index=False)
        template_metadata.to_excel(writer, sheet_name="battery_metadata", index=False)

    st.download_button(
        label="Download Excel Template",
        data=buffer.getvalue(),
        file_name="build_sheet_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

elif page == "Rename Batteries":
    st.subheader("Rename Batteries")

    st.info(
        """
    **Rename batteries to more meaningful names for easier identification and export.**

    This will update the battery ID across all tables (batteries, cycles, capacity_fade, battery_metadata).
    """
    )

    # Load all batteries
    conn = sqlite3.connect(DB_PATH)
    all_batteries = pd.read_sql_query(
        "SELECT DISTINCT battery_id FROM batteries ORDER BY battery_id", conn
    )
    conn.close()

    if all_batteries.empty:
        st.warning("No batteries found in database.")
    else:
        with st.form("rename_battery"):
            col1, col2 = st.columns(2)

            with col1:
                old_id = st.selectbox(
                    "Select Battery to Rename", all_batteries["battery_id"].tolist()
                )

            with col2:
                new_id = st.text_input("New Battery ID*", placeholder="e.g., NMC811_Cell_01")

            submitted = st.form_submit_button("Rename Battery", use_container_width=True)

            if submitted:
                if not new_id:
                    st.error("Please provide a new battery ID")
                elif old_id == new_id:
                    st.warning("New ID is the same as old ID")
                else:
                    success, message = rename_battery(old_id, new_id)
                    if success:
                        st.success(message)
                        st.cache_data.clear()
                        st.rerun()
                    else:
                        st.error(message)

        # Batch rename section
        st.markdown("---")
        st.markdown("### Batch Rename")

        st.markdown("Upload a CSV with columns: `old_id`, `new_id`")

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
                            success, message = rename_battery(row["old_id"], row["new_id"])

                            if success:
                                success_count += 1
                            else:
                                error_count += 1
                                errors.append(f"{row['old_id']}: {message}")

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

elif page == "View Battery Groups":
    st.subheader("Battery Groups by Build Type")

    metadata = load_battery_metadata()

    if metadata.empty:
        st.warning("No battery metadata found. Link batteries to build sheets to see groups.")
    else:
        # Group by build_type
        if "build_type" in metadata.columns:
            st.markdown("### Group by Build Type")

            build_types = metadata["build_type"].dropna().unique()

            for build_type in build_types:
                with st.expander(
                    f"{build_type} ({len(metadata[metadata['build_type'] == build_type])} batteries)"
                ):
                    group_df = metadata[metadata["build_type"] == build_type]

                    # Show summary
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Total Batteries", len(group_df))

                    with col2:
                        unique_builds = group_df["build_id"].nunique()
                        st.metric("Unique Builds", unique_builds)

                    with col3:
                        active = len(group_df[group_df["status"] == "Active"])
                        st.metric("Active", active)

                    with col4:
                        if "cathode_material" in group_df.columns:
                            cathodes = group_df["cathode_material"].dropna().unique()
                            st.metric("Cathode Types", len(cathodes))

                    # Show detailed table
                    st.dataframe(
                        group_df[
                            [
                                "battery_id",
                                "build_name",
                                "cathode_material",
                                "anode_material",
                                "serial_number",
                                "status",
                            ]
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )

                    # Export group
                    csv = group_df.to_csv(index=False)
                    st.download_button(
                        label=f"Export {build_type} Group",
                        data=csv,
                        file_name=f"battery_group_{build_type}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

        # Group by cathode material
        st.markdown("---")
        st.markdown("### Group by Cathode Material")

        if "cathode_material" in metadata.columns:
            cathodes = metadata["cathode_material"].dropna().unique()

            for cathode in cathodes:
                count = len(metadata[metadata["cathode_material"] == cathode])
                st.markdown(f"**{cathode}:** {count} batteries")

# Footer
st.markdown("---")
st.caption("Build Sheet Management System | AmpereData v1.0")
