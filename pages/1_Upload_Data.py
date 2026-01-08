"""
Upload Data Page
================
Upload and process battery testing data files with intelligent column mapping.

Supported formats:
- BioLogic (.mpt, .mps)
- Neware (.nda, .ndax, .csv)
- Generic CSV/TXT
- Excel (.xlsx)

Features:
- Automatic column detection
- Interactive column mapping interface
- Data preview and validation
- Database storage
"""

import sqlite3
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import authentication
from utils.auth import require_auth, show_logout_button  # noqa: E402

# Import column mapper
from utils.column_mapper import ColumnMapper, StandardSchema, get_column_preview  # noqa: E402

# Import parsers directly
try:
    from backend.parsers.biologic import BiologicMPTParser

    BIOLOGIC_AVAILABLE = True
except ImportError:
    BIOLOGIC_AVAILABLE = False

try:
    from backend.parsers.neware import HAS_NEWARE_NDA, NewareNDAParser

    NEWARE_AVAILABLE = True
    NEWARE_BINARY_SUPPORT = HAS_NEWARE_NDA
except ImportError:
    NEWARE_AVAILABLE = False
    NEWARE_BINARY_SUPPORT = False

PARSERS_AVAILABLE = BIOLOGIC_AVAILABLE or NEWARE_AVAILABLE

# Page config
st.set_page_config(page_title="Upload Data - AmpereData", page_icon=None, layout="wide")

# Authentication
require_auth()

# Title
st.title("Upload Battery Testing Data")
st.markdown(
    "Upload files from BioLogic, Neware, or generic CSV formats with intelligent column mapping."
)

# Show logout button
show_logout_button()

# Initialize session state
if "upload_history" not in st.session_state:
    st.session_state.upload_history = []
if "current_df" not in st.session_state:
    st.session_state.current_df = None
if "column_mapper" not in st.session_state:
    st.session_state.column_mapper = None
if "mapping_step" not in st.session_state:
    st.session_state.mapping_step = "upload"  # 'upload', 'sheet_select', 'mapping', 'complete'
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None
if "excel_sheets" not in st.session_state:
    st.session_state.excel_sheets = None
if "selected_sheet" not in st.session_state:
    st.session_state.selected_sheet = None
if "temp_file_path" not in st.session_state:
    st.session_state.temp_file_path = None

# Database path
DB_PATH = "nasa_amperedata_full.db"


def store_data_in_database(df: pd.DataFrame, battery_id: str, file_name: str):
    """Store mapped data in the database."""
    try:
        conn = sqlite3.connect(DB_PATH)

        # Ensure battery_id is in the DataFrame (add if missing)
        if "battery_id" not in df.columns:
            df = df.copy()
            df["battery_id"] = battery_id

        # Store time-series data in cycles table
        # cycles table schema: battery_id, test_id, cycle_type, timestamp, time_seconds, voltage, current, temperature, capacity_ah, metadata
        if "time_s" in df.columns and "voltage" in df.columns:
            cycles_df = pd.DataFrame()
            cycles_df["battery_id"] = battery_id

            # ADD CYCLE NUMBER (test_id) - CRITICAL for voltage profile, dQ/dV, etc.
            if "cycle_number" in df.columns:
                cycles_df["test_id"] = df["cycle_number"]
            else:
                # If no cycle column, assign all data to cycle 1
                cycles_df["test_id"] = 1

            cycles_df["time_seconds"] = df["time_s"]
            cycles_df["voltage"] = df["voltage"]
            cycles_df["current"] = df["current"] if "current" in df.columns else None
            cycles_df["temperature"] = (
                df["temperature_c"] if "temperature_c" in df.columns else None
            )
            cycles_df["capacity_ah"] = (
                df["discharge_capacity_ah"] if "discharge_capacity_ah" in df.columns else None
            )
            cycles_df["timestamp"] = datetime.now().isoformat()

            # Only include columns that exist in the table
            cycles_df = cycles_df.dropna(axis=1, how="all")  # Drop columns that are all None
            cycles_df.to_sql("cycles", conn, if_exists="append", index=False)

        # Store capacity fade data if available
        # capacity_fade table schema: battery_id, cycle_number, test_id, capacity_ah, retention_percent, timestamp
        if "cycle_number" in df.columns and (
            "discharge_capacity_ah" in df.columns or "charge_capacity_ah" in df.columns
        ):
            # Group by cycle to get one row per cycle
            capacity_df = (
                df.groupby("cycle_number")
                .agg(
                    {
                        "discharge_capacity_ah": (
                            "mean" if "discharge_capacity_ah" in df.columns else lambda x: None
                        ),
                        "charge_capacity_ah": (
                            "mean" if "charge_capacity_ah" in df.columns else lambda x: None
                        ),
                    }
                )
                .reset_index()
            )

            # Use discharge capacity as the main capacity metric
            capacity_df["battery_id"] = battery_id
            capacity_df["capacity_ah"] = (
                capacity_df["discharge_capacity_ah"]
                if "discharge_capacity_ah" in capacity_df.columns
                else capacity_df.get("charge_capacity_ah")
            )
            capacity_df["timestamp"] = datetime.now().isoformat()

            # Calculate retention percent if we have initial capacity
            if not capacity_df["capacity_ah"].isna().all():
                initial_capacity = capacity_df["capacity_ah"].iloc[0]
                if initial_capacity and initial_capacity > 0:
                    capacity_df["retention_percent"] = (
                        capacity_df["capacity_ah"] / initial_capacity
                    ) * 100

            # Select only columns that exist in the table
            final_capacity_df = capacity_df[
                ["battery_id", "cycle_number", "capacity_ah", "timestamp"]
            ].copy()
            if "retention_percent" in capacity_df.columns:
                final_capacity_df["retention_percent"] = capacity_df["retention_percent"]

            final_capacity_df.to_sql("capacity_fade", conn, if_exists="append", index=False)

        # Create/update batteries table entry
        cursor = conn.cursor()
        try:
            # Check if battery exists
            cursor.execute("SELECT battery_id FROM batteries WHERE battery_id = ?", (battery_id,))
            if not cursor.fetchone():
                # Insert new battery
                cursor.execute(
                    """
                    INSERT INTO batteries (battery_id, experiment_id, manufacturer, model)
                    VALUES (?, ?, ?, ?)
                """,
                    (battery_id, None, "Unknown", file_name),
                )
        except sqlite3.OperationalError:
            # batteries table might not exist, skip it
            pass

        conn.commit()
        conn.close()
        return True, "Data stored successfully in database!"
    except Exception as e:
        import traceback

        error_details = traceback.format_exc()
        return False, f"Database error: {str(e)}\n\nDetails:\n{error_details}"


# Main UI Flow
if st.session_state.mapping_step == "upload":
    # STEP 1: FILE UPLOAD
    st.subheader("Step 1: Upload File")

    # Show parser availability status
    with st.expander("Parser Status", expanded=False):
        if BIOLOGIC_AVAILABLE:
            st.markdown("**BioLogic** (.mpt, .mps) - Fully supported")
        else:
            st.markdown("**BioLogic** - Not available")

        if NEWARE_AVAILABLE:
            if NEWARE_BINARY_SUPPORT:
                st.markdown("**Neware** (.nda, .ndax, .csv) - Fully supported")
            else:
                st.markdown("**Neware** (.csv only) - Binary files need `NewareNDA` library")
                st.markdown("   → Install: `pip install NewareNDA`")
        else:
            st.markdown("**Neware** - Not available")

        st.markdown("**CSV/TXT/Excel** - Fully supported (pandas)")

    # File format selection
    file_format = st.selectbox(
        "Select data format",
        [
            "Auto-detect",
            "BioLogic (.mpt, .mps)",
            "Neware (.nda, .ndax, .csv)",
            "Generic CSV/TXT",
            "Excel (.xlsx)",
        ],
        help="Choose the instrument format or let the system auto-detect",
    )

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["mpt", "mps", "nda", "ndax", "csv", "txt", "xlsx", "xls"],
        help="Drag and drop or click to browse",
    )

    if uploaded_file is not None:
        # Display file info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("File Name", uploaded_file.name)
        with col2:
            st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
        with col3:
            file_ext = Path(uploaded_file.name).suffix
            st.metric("Format", file_ext.upper())

        st.markdown("---")

        # Process button
        if st.button("Parse File & Configure Mapping", type="primary", use_container_width=True):
            with st.spinner("Parsing file..."):
                try:
                    # Save uploaded file temporarily
                    temp_dir = Path(tempfile.gettempdir()) / "amperedata_uploads"
                    temp_dir.mkdir(exist_ok=True)
                    temp_file = temp_dir / uploaded_file.name

                    with open(temp_file, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # Step 1: Detect format
                    status_text.text("Detecting file format...")
                    progress_bar.progress(20)

                    # Determine parser type
                    if file_format == "Auto-detect":
                        if file_ext in [".mpt", ".mps"]:
                            parser_type = "biologic"
                        elif file_ext in [".nda", ".ndax"]:
                            parser_type = "neware"
                        else:
                            parser_type = "csv"
                    else:
                        parser_type_map = {
                            "BioLogic (.mpt, .mps)": "biologic",
                            "Neware (.nda, .ndax, .csv)": "neware",
                            "Generic CSV/TXT": "csv",
                            "Excel (.xlsx)": "csv",
                        }
                        parser_type = parser_type_map.get(file_format, "csv")

                    # Step 2: Parse file
                    status_text.text(f"Parsing with {parser_type} parser...")
                    progress_bar.progress(50)

                    df = None

                    # Handle BioLogic files
                    if file_ext in [".mpt", ".mps"] and parser_type == "biologic":
                        if not BIOLOGIC_AVAILABLE:
                            st.error("BioLogic parser not available. Please check dependencies.")
                            st.stop()

                        # Parse BioLogic file
                        parser = BiologicMPTParser()
                        biologic_data, biologic_metadata = parser.parse_file(temp_file)

                        # Convert to DataFrame
                        df = pd.DataFrame(
                            {
                                "time_s": biologic_data.time_s,
                                "voltage": biologic_data.voltage_V,
                                "current": biologic_data.current_A,
                                "cycle_number": biologic_data.cycle_number,
                            }
                        )

                        # Add optional columns if available
                        if biologic_data.charge_capacity_Ah:
                            df["charge_capacity_ah"] = biologic_data.charge_capacity_Ah
                        if biologic_data.discharge_capacity_Ah:
                            df["discharge_capacity_ah"] = biologic_data.discharge_capacity_Ah
                        if biologic_data.energy_Wh:
                            df["energy_wh"] = biologic_data.energy_Wh

                        status_text.text("BioLogic file parsed successfully!")

                    # Handle Neware binary files
                    elif file_ext in [".nda", ".ndax"] and parser_type == "neware":
                        if not NEWARE_AVAILABLE:
                            st.error("Neware parser not available. Please check dependencies.")
                            st.stop()

                        if not NEWARE_BINARY_SUPPORT:
                            st.error(
                                "Neware binary files require NewareNDA library. Install with: pip install NewareNDA"
                            )
                            st.stop()

                        # Parse Neware binary file
                        parser = NewareNDAParser()
                        neware_data, neware_metadata = parser.parse_file(temp_file)

                        # Convert time from hours to seconds
                        time_s = [t * 3600 for t in neware_data.time_h]

                        # Convert current from mA to A
                        current_a = [i / 1000 for i in neware_data.current_ma]

                        # Convert capacities from mAh to Ah
                        charge_cap_ah = [c / 1000 for c in neware_data.capacitance_chg_mah]
                        discharge_cap_ah = [d / 1000 for d in neware_data.capacitance_dchg_mah]

                        # Convert to DataFrame
                        df = pd.DataFrame(
                            {
                                "time_s": time_s,
                                "voltage": neware_data.voltage_v,
                                "current": current_a,
                                "cycle_number": neware_data.cycle_id,
                                "step_id": neware_data.step_id,
                                "charge_capacity_ah": charge_cap_ah,
                                "discharge_capacity_ah": discharge_cap_ah,
                            }
                        )

                        # Add optional temperature data if available
                        if hasattr(neware_data, "avg_temp_c") and neware_data.avg_temp_c:
                            df["temperature_c"] = neware_data.avg_temp_c
                        elif hasattr(neware_data, "temperature_c") and neware_data.temperature_c:
                            df["temperature_c"] = neware_data.temperature_c

                        status_text.text("Neware file parsed successfully!")

                    # Handle CSV/TXT/Excel files
                    elif file_ext in [".csv", ".txt", ".xlsx", ".xls"]:
                        if file_ext in [".xlsx", ".xls"]:
                            # For Excel files, detect sheets first
                            status_text.text("Detecting Excel sheets...")
                            excel_file = pd.ExcelFile(temp_file)
                            sheet_names = excel_file.sheet_names

                            if len(sheet_names) > 1:
                                # Multiple sheets - need user selection
                                st.session_state.excel_sheets = sheet_names
                                st.session_state.uploaded_file_name = uploaded_file.name
                                st.session_state.temp_file_path = str(temp_file)
                                st.session_state.mapping_step = "sheet_select"
                                progress_bar.progress(100)
                                status_text.text(
                                    f"Found {len(sheet_names)} sheets. Please select one to continue."
                                )
                                st.success(
                                    f"Excel file loaded. Found {len(sheet_names)} sheets. Please select one."
                                )
                                st.rerun()
                            else:
                                # Single sheet - load directly
                                df = pd.read_excel(temp_file, sheet_name=sheet_names[0])
                                status_text.text(f"Loaded sheet: {sheet_names[0]}")
                        else:
                            # Try different encodings and delimiters for CSV/TXT
                            try:
                                df = pd.read_csv(temp_file)
                            except Exception:
                                try:
                                    df = pd.read_csv(temp_file, encoding="latin1")
                                except Exception:
                                    df = pd.read_csv(temp_file, sep="\t")

                            status_text.text("File loaded successfully!")

                    else:
                        st.error(f"Unsupported file type: {file_ext}")
                        st.stop()

                    progress_bar.progress(100)

                    if df is not None and not df.empty:
                        # Store in session state
                        st.session_state.current_df = df
                        st.session_state.uploaded_file_name = uploaded_file.name

                        # For BioLogic/Neware, columns are already mapped, skip to complete
                        if parser_type in ["biologic", "neware"]:
                            st.session_state.mapping_step = "complete"
                            st.session_state.column_mapper = (
                                ColumnMapper()
                            )  # Empty mapper for already-mapped data
                        else:
                            # For CSV/Excel, go to mapping step
                            st.session_state.mapping_step = "mapping"
                            st.session_state.column_mapper = ColumnMapper()

                        st.success(
                            f"File parsed. Found {len(df)} rows and {len(df.columns)} columns."
                        )
                        st.rerun()
                    else:
                        st.error("Failed to parse file or file is empty.")

                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
                    import traceback

                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())

elif st.session_state.mapping_step == "sheet_select":
    # STEP 1.5: EXCEL SHEET SELECTION
    st.subheader("Step 1.5: Select Excel Sheet")
    st.markdown(f"**File:** {st.session_state.uploaded_file_name}")

    if st.session_state.excel_sheets is None:
        st.warning("No sheets detected. Please go back and upload a file.")
        if st.button("← Back to Upload"):
            st.session_state.mapping_step = "upload"
            st.session_state.excel_sheets = None
            st.session_state.temp_file_path = None
            st.rerun()
        st.stop()

    st.markdown("---")

    st.info(
        f"This Excel file contains **{len(st.session_state.excel_sheets)} sheets**. "
        "Please select which sheet contains the battery testing data."
    )

    # Display sheets with preview
    sheet_names = st.session_state.excel_sheets
    temp_file = st.session_state.temp_file_path

    # Create sheet selection UI
    st.markdown("### Available Sheets:")

    # Show sheet previews
    for idx, sheet_name in enumerate(sheet_names):
        with st.expander(f"Sheet {idx + 1}: {sheet_name}", expanded=(idx == 0)):
            try:
                # Load preview of this sheet (first 5 rows)
                preview_df = pd.read_excel(temp_file, sheet_name=sheet_name, nrows=5)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows (preview)", len(preview_df))
                with col2:
                    st.metric("Columns", len(preview_df.columns))
                with col3:
                    st.metric("Non-empty cells", preview_df.notna().sum().sum())

                st.dataframe(preview_df, use_container_width=True)
                st.caption("Showing first 5 rows")

                # Select button for this sheet
                if st.button(
                    f"Use Sheet: {sheet_name}",
                    key=f"select_sheet_{idx}",
                    type="primary",
                    use_container_width=True,
                ):
                    # Load the selected sheet
                    with st.spinner(f"Loading sheet: {sheet_name}..."):
                        df = pd.read_excel(temp_file, sheet_name=sheet_name)

                        # Store in session state
                        st.session_state.current_df = df
                        st.session_state.selected_sheet = sheet_name
                        st.session_state.mapping_step = "mapping"
                        st.session_state.column_mapper = ColumnMapper()

                        st.success(
                            f"Loaded sheet '{sheet_name}'. Found {len(df)} rows and {len(df.columns)} columns."
                        )
                        st.rerun()

            except Exception as e:
                st.error(f"Error loading preview: {str(e)}")

    st.markdown("---")

    # Back button
    if st.button("← Back to Upload", use_container_width=True):
        st.session_state.mapping_step = "upload"
        st.session_state.excel_sheets = None
        st.session_state.temp_file_path = None
        st.session_state.selected_sheet = None
        st.rerun()

elif st.session_state.mapping_step == "mapping":
    # STEP 2: COLUMN MAPPING
    st.subheader("Step 2: Map Columns to Standard Schema")
    if st.session_state.selected_sheet:
        st.markdown(
            f"**File:** {st.session_state.uploaded_file_name} → Sheet: `{st.session_state.selected_sheet}`"
        )
    else:
        st.markdown(f"**File:** {st.session_state.uploaded_file_name}")

    df = st.session_state.current_df
    mapper = st.session_state.column_mapper

    if df is None:
        st.warning("No data loaded. Please go back and upload a file.")
        if st.button("← Back to Upload"):
            st.session_state.mapping_step = "upload"
            st.rerun()
        st.stop()

    st.markdown("---")

    # Show data preview
    with st.expander("Data Preview", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)
        st.caption(f"Showing first 10 of {len(df)} rows")

    # Auto-detect mapping
    if "auto_mapping_done" not in st.session_state or not st.session_state.auto_mapping_done:
        with st.spinner("Auto-detecting column mappings..."):
            detected_mapping = mapper.auto_detect_mapping(df)

            # Apply high-confidence mappings automatically
            for std_col, (user_col, confidence) in detected_mapping.items():
                if user_col and confidence > 0.8:
                    mapper.set_mapping(user_col, std_col)

            st.session_state.auto_mapping_done = True

    # Load/Save mapping options
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### Configure Column Mappings")
        st.markdown(
            "Map your file columns to the standard schema. **Required columns** are marked with *"
        )
    with col2:
        # Load saved mapping
        saved_mappings = ColumnMapper.list_saved_mappings()
        if not saved_mappings.empty:
            with st.popover("Load Mapping", use_container_width=True):
                st.markdown("**Saved Mappings:**")
                for _, row in saved_mappings.iterrows():
                    if st.button(
                        f"{row['mapping_name']}",
                        key=f"load_{row['mapping_name']}",
                        use_container_width=True,
                    ):
                        try:
                            mapper = ColumnMapper.load_from_database(row["mapping_name"])
                            st.session_state.column_mapper = mapper
                            st.success(f"Loaded mapping: {row['mapping_name']}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to load: {str(e)}")

    # Get standard columns
    std_columns = StandardSchema.get_all_columns()
    user_columns = ["(Not Mapped)"] + list(df.columns)

    # Group by category
    categories = {}
    for col in std_columns:
        if col.category not in categories:
            categories[col.category] = []
        categories[col.category].append(col)

    # Display mapping UI by category
    tabs = st.tabs([cat.title() for cat in categories.keys()])

    for tab, (_category, cols) in zip(tabs, categories.items()):
        with tab:
            for std_col in cols:
                col1, col2, col3, col4 = st.columns([1, 2, 2, 1])

                with col1:
                    if std_col.required:
                        st.markdown(f"**{std_col.name}** *")
                    else:
                        st.markdown(f"**{std_col.name}**")

                with col2:
                    st.caption(f"{std_col.description}")
                    if std_col.unit:
                        st.caption(f"Unit: {std_col.unit}")

                with col3:
                    # Check if already mapped
                    current_mapping = None
                    for user_col, mapped_std_col in mapper.mapping.items():
                        if mapped_std_col == std_col.name:
                            current_mapping = user_col
                            break

                    # Selectbox for mapping
                    default_idx = 0
                    if current_mapping:
                        try:
                            default_idx = user_columns.index(current_mapping)
                        except ValueError:
                            default_idx = 0

                    selected = st.selectbox(
                        "Map to",
                        user_columns,
                        index=default_idx,
                        key=f"map_{std_col.name}",
                        label_visibility="collapsed",
                    )

                    # Update mapping
                    if selected != "(Not Mapped)":
                        # Remove old mapping if exists
                        if current_mapping and current_mapping != selected:
                            if current_mapping in mapper.mapping:
                                del mapper.mapping[current_mapping]
                        mapper.set_mapping(selected, std_col.name)
                    else:
                        if current_mapping and current_mapping in mapper.mapping:
                            del mapper.mapping[current_mapping]

                with col4:
                    # Show preview button
                    if current_mapping or (selected and selected != "(Not Mapped)"):
                        col_to_preview = selected if selected != "(Not Mapped)" else current_mapping
                        if st.button(
                            "View",
                            key=f"preview_{std_col.name}",
                            help="Preview column data",
                        ):
                            preview = get_column_preview(df, col_to_preview)
                            st.info(
                                f"**{col_to_preview}**\n\nType: {preview.get('dtype')}\nSamples: {preview.get('samples')}"
                            )

    # Show mapping summary
    st.markdown("---")
    st.markdown("### Mapping Summary")

    summary_df = mapper.get_mapping_summary()
    if not summary_df.empty:
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    else:
        st.info("No columns mapped yet. Please map at least the required columns to proceed.")

    # Validate mapping
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        if st.button("Validate & Apply Mapping", type="primary", use_container_width=True):
            # Apply mapping
            df_mapped = mapper.apply_mapping(df)

            # Validate
            validation_result = mapper.validate_mapping(df_mapped)

            if validation_result["errors"]:
                st.error("Mapping validation failed:")
                for error in validation_result["errors"]:
                    st.error(f"• {error}")
            else:
                st.success("Mapping validated successfully!")

                if validation_result["warnings"]:
                    st.warning("Warnings:")
                    for warning in validation_result["warnings"]:
                        st.warning(f"• {warning}")

                # Update session state
                st.session_state.current_df = df_mapped
                st.session_state.mapping_step = "complete"
                st.rerun()

    with col2:
        # Save mapping button
        with st.popover("Save Mapping", use_container_width=True):
            st.markdown("**Save this mapping for reuse:**")
            mapping_name = st.text_input("Mapping Name", placeholder="e.g., My Lab CSV Format")
            mapping_desc = st.text_area(
                "Description (optional)", placeholder="Describe this mapping..."
            )
            if st.button("Save", type="primary", disabled=not mapping_name):
                try:
                    mapper.save_to_database(mapping_name, mapping_desc)
                    st.success(f"Saved: {mapping_name}")
                except Exception as e:
                    st.error(f"Failed to save: {str(e)}")

    with col3:
        if st.button("← Back", use_container_width=True):
            # Check if we came from sheet selection
            if st.session_state.selected_sheet and st.session_state.temp_file_path:
                # Go back to sheet selection
                st.session_state.mapping_step = "sheet_select"
            else:
                # Go back to upload
                st.session_state.mapping_step = "upload"
                st.session_state.excel_sheets = None
                st.session_state.temp_file_path = None
                st.session_state.selected_sheet = None
            st.session_state.auto_mapping_done = False
            st.rerun()

elif st.session_state.mapping_step == "complete":
    # STEP 3: COMPLETE & STORE
    st.subheader("Step 3: Review & Store Data")
    if st.session_state.selected_sheet:
        st.markdown(
            f"**File:** {st.session_state.uploaded_file_name} → Sheet: `{st.session_state.selected_sheet}`"
        )
    else:
        st.markdown(f"**File:** {st.session_state.uploaded_file_name}")

    df = st.session_state.current_df

    if df is None:
        st.warning("No data loaded.")
        if st.button("← Start Over"):
            st.session_state.mapping_step = "upload"
            st.rerun()
        st.stop()

    st.success(f"Data ready. {len(df)} rows x {len(df.columns)} columns")

    # Show mapped data preview
    with st.expander("Mapped Data Preview", expanded=True):
        st.dataframe(df.head(20), use_container_width=True)

    # Show column info
    with st.expander("Column Information"):
        col_info = []
        for col in df.columns:
            col_info.append(
                {
                    "Column": col,
                    "Type": str(df[col].dtype),
                    "Non-Null": f"{df[col].count()} / {len(df)}",
                    "Sample": str(df[col].iloc[0]) if len(df) > 0 else "N/A",
                }
            )
        st.dataframe(pd.DataFrame(col_info), use_container_width=True, hide_index=True)

    st.markdown("---")

    # Storage options
    st.markdown("### Storage Options")

    # Check if battery_id column exists in mapped data
    has_battery_id_column = "battery_id" in df.columns

    col1, col2 = st.columns(2)
    with col1:
        if has_battery_id_column and not df["battery_id"].isna().all():
            # Use battery_id from data
            unique_ids = df["battery_id"].dropna().unique()
            if len(unique_ids) == 1:
                battery_id = st.text_input(
                    "Battery/Cell ID",
                    value=str(unique_ids[0]),
                    help="Battery ID from your data file",
                )
            else:
                battery_id = st.selectbox(
                    "Battery/Cell ID (multiple found)",
                    options=unique_ids.tolist(),
                    help="Multiple battery IDs found in file. Select one or upload files separately.",
                )
                st.warning(
                    f"Multiple battery IDs detected: {len(unique_ids)}. Consider uploading each battery separately."
                )
        else:
            # Auto-generate battery_id based on filename and timestamp
            import re

            # Clean filename for use in ID
            clean_name = re.sub(
                r"[^a-zA-Z0-9_-]", "_", Path(st.session_state.uploaded_file_name).stem
            )
            clean_name = clean_name[:20]  # Limit length
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_id = f"{clean_name}_{timestamp}" if clean_name else f"BAT_{timestamp}"

            battery_id = st.text_input(
                "Battery/Cell ID",
                value=default_id,
                help="Auto-generated from filename + timestamp (you can edit this)",
            )
            if not has_battery_id_column:
                st.info("No battery_id column mapped. Auto-generated ID from filename.")

    with col2:
        store_in_db = st.checkbox("Store in Database", value=True, help="Save to NASA database")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Save Data", type="primary", use_container_width=True):
            if store_in_db:
                with st.spinner("Storing data in database..."):
                    success, message = store_data_in_database(
                        df, battery_id, st.session_state.uploaded_file_name
                    )
                    if success:
                        st.success(message)
                        st.balloons()

                        # Add to history
                        st.session_state.upload_history.append(
                            {
                                "file_name": st.session_state.uploaded_file_name,
                                "battery_id": battery_id,
                                "timestamp": datetime.now().isoformat(),
                                "rows": len(df),
                                "columns": len(df.columns),
                            }
                        )
                    else:
                        st.error(message)
            else:
                st.info("Data not stored (storage disabled)")

    with col2:
        # Download mapped data as CSV
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"{battery_id}_mapped.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col3:
        if st.button("Upload Another", use_container_width=True):
            # Reset session state
            st.session_state.current_df = None
            st.session_state.column_mapper = None
            st.session_state.mapping_step = "upload"
            st.session_state.uploaded_file_name = None
            st.session_state.auto_mapping_done = False
            st.rerun()

    # Show link to visualizations
    st.markdown("---")
    st.markdown("### Next Steps")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("View Visualizations", use_container_width=True):
            st.switch_page("pages/2_Visualizations.py")
    with col2:
        if st.button("Explore Data", use_container_width=True):
            st.switch_page("pages/3_Data_Explorer.py")

# Sidebar: Upload History
st.sidebar.markdown("### Upload History")
if st.session_state.upload_history:
    for _i, item in enumerate(reversed(st.session_state.upload_history[-5:])):
        with st.sidebar.expander(f"{item['file_name']}", expanded=False):
            st.caption(f"**Battery ID:** {item['battery_id']}")
            st.caption(f"**Time:** {item['timestamp'][:19]}")
            st.caption(f"**Rows:** {item['rows']}, **Columns:** {item['columns']}")
else:
    st.sidebar.info("No uploads yet")

# Sidebar: Help
with st.sidebar.expander("Help & Tips"):
    st.markdown(
        """
    **Column Mapping Tips:**

    1. **Auto-Detection**: The system automatically suggests mappings based on common column names.

    2. **Required Columns** (*):
       - time_s (time in seconds)
       - cycle_number
       - voltage
       - current

    3. **Optional but Recommended**:
       - temperature_c
       - charge_capacity_ah
       - discharge_capacity_ah

    4. **Preview Data**: Click the View button to see sample values before mapping.

    5. **Common Patterns**:
       - Time: "time", "elapsed_time", "t"
       - Voltage: "V", "volt", "Ecell"
       - Current: "I", "amp", "current"
    """
    )
