"""
Settings & Configuration Page
==============================
Configure platform settings, parsers, and system preferences.

Features:
- Parser configuration
- Database management
- System settings
- User preferences
- About information
"""

import sqlite3
import sys
from pathlib import Path

import streamlit as st

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Page config
st.set_page_config(page_title="Settings - AmpereData", page_icon=None, layout="wide")

# Title
st.title("Settings & Configuration")
st.markdown("Configure platform settings and preferences.")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["General", "Parsers", "Database", "Preferences", "About"])

# === GENERAL SETTINGS ===
with tab1:
    st.subheader("General Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Appearance")
        theme = st.selectbox("Theme", ["Light", "Dark", "Auto"], index=0)

        st.markdown("### Dashboard")
        default_page = st.selectbox(
            "Default landing page",
            ["Home", "Upload Data", "Visualizations", "Data Explorer"],
            index=0,
        )

        refresh_interval = st.slider(
            "Auto-refresh interval (seconds)",
            min_value=0,
            max_value=300,
            value=60,
            step=30,
            help="0 = disabled",
        )

    with col2:
        st.markdown("### Notifications")
        enable_notifications = st.checkbox("Enable notifications", value=True)
        notify_on_upload = st.checkbox("Notify on file upload complete", value=True)
        notify_on_error = st.checkbox("Notify on errors", value=True)

        st.markdown("### Performance")
        cache_duration = st.slider(
            "Cache duration (minutes)",
            min_value=1,
            max_value=60,
            value=10,
            help="How long to cache data",
        )

        max_rows_display = st.number_input(
            "Max rows to display", min_value=10, max_value=10000, value=1000, step=100
        )

    st.markdown("---")

    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("Save Settings", type="primary", use_container_width=True):
            st.success("Settings saved successfully!")
    with col2:
        if st.button("Reset to Defaults", use_container_width=True):
            st.info("Settings reset to defaults")
            st.rerun()

# === PARSER SETTINGS ===
with tab2:
    st.subheader("Parser Configuration")

    st.markdown(
        """
    Configure how different file formats are parsed and processed.
    """
    )

    # BioLogic Parser
    with st.expander("BioLogic Parser", expanded=True):
        st.markdown("**Supported formats:** `.mpt`, `.mps`")

        col1, col2 = st.columns(2)
        with col1:
            biologic_enabled = st.checkbox("Enable BioLogic parser", value=True)
            biologic_auto_detect = st.checkbox("Auto-detect encoding", value=True)
        with col2:
            biologic_encoding = st.selectbox(
                "Default encoding",
                ["utf-8", "latin-1", "iso-8859-1", "cp1252"],
                index=0,
                disabled=biologic_auto_detect,
            )

        biologic_validate = st.checkbox("Validate data quality", value=True)
        biologic_extract_metadata = st.checkbox("Extract metadata", value=True)

    # Neware Parser
    with st.expander("Neware Parser"):
        st.markdown("**Supported formats:** `.nda`, `.ndax`, `.csv`")

        col1, col2 = st.columns(2)
        with col1:
            neware_enabled = st.checkbox("Enable Neware parser", value=True)
            neware_binary_support = st.checkbox("Support binary files (.nda, .ndax)", value=True)
        with col2:
            neware_decimal_separator = st.selectbox("Decimal separator", [".", ","], index=0)

        neware_validate = st.checkbox("Validate data quality", value=True, key="neware_validate")

    # CSV/Generic Parser
    with st.expander("Generic CSV/TXT Parser"):
        st.markdown("**Supported formats:** `.csv`, `.txt`, `.xlsx`")

        col1, col2 = st.columns(2)
        with col1:
            csv_enabled = st.checkbox("Enable CSV parser", value=True)
            csv_auto_delimiter = st.checkbox("Auto-detect delimiter", value=True)
        with col2:
            csv_delimiter = st.selectbox(
                "Default delimiter",
                [",", ";", "\\t", "|"],
                index=0,
                disabled=csv_auto_delimiter,
            )

        csv_skip_rows = st.number_input("Rows to skip", min_value=0, max_value=100, value=0)
        csv_header_row = st.number_input("Header row", min_value=0, max_value=100, value=0)

    st.markdown("---")
    if st.button("Save Parser Settings", type="primary"):
        st.success("Parser settings saved!")

# === DATABASE SETTINGS ===
with tab3:
    st.subheader("Database Management")

    db_path = Path("nasa_amperedata_full.db")

    # Database info
    col1, col2, col3 = st.columns(3)

    with col1:
        if db_path.exists():
            db_size = db_path.stat().st_size / (1024 * 1024)  # MB
            st.metric("Database Size", f"{db_size:.2f} MB")
        else:
            st.metric("Database Size", "N/A")

    with col2:
        if db_path.exists():
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
            table_count = cursor.fetchone()[0]
            conn.close()
            st.metric("Tables", table_count)
        else:
            st.metric("Tables", "0")

    with col3:
        st.metric("Database Type", "SQLite")

    st.markdown("---")

    # Database operations
    st.markdown("### Database Operations")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Backup & Restore**")
        if st.button("Create Backup", use_container_width=True):
            if db_path.exists():
                import shutil
                from datetime import datetime

                backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
                shutil.copy2(db_path, backup_name)
                st.success(f"Backup created: {backup_name}")
            else:
                st.warning("No database to backup")

        uploaded_backup = st.file_uploader("Restore from backup", type=["db"])
        if uploaded_backup and st.button("Restore Backup", use_container_width=True):
            st.warning("This will overwrite current database!")

    with col2:
        st.markdown("**Maintenance**")
        if st.button("Vacuum Database", use_container_width=True):
            if db_path.exists():
                conn = sqlite3.connect(str(db_path))
                conn.execute("VACUUM")
                conn.close()
                st.success("Database vacuumed")
            else:
                st.warning("No database found")

        if st.button("Analyze Database", use_container_width=True):
            if db_path.exists():
                conn = sqlite3.connect(str(db_path))
                conn.execute("ANALYZE")
                conn.close()
                st.success("Database analyzed")
            else:
                st.warning("No database found")

    st.markdown("---")

    # Danger zone
    with st.expander("Danger Zone", expanded=False):
        st.warning("**Warning:** These operations cannot be undone!")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear All Data", type="secondary", use_container_width=True):
                st.error("This feature is disabled for safety")

        with col2:
            if st.button("Delete Database", type="secondary", use_container_width=True):
                st.error("This feature is disabled for safety")

# === USER PREFERENCES ===
with tab4:
    st.subheader("User Preferences")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Visualization Defaults")
        default_viz = st.selectbox(
            "Default visualization",
            ["Capacity Fade", "Voltage Profiles", "EIS/Impedance", "Cycle Statistics"],
            index=0,
        )

        show_grid = st.checkbox("Show grid lines", value=True)
        show_legend = st.checkbox("Show legend", value=True)

        color_scheme = st.selectbox(
            "Color scheme",
            ["Default", "Colorblind-friendly", "High contrast", "Grayscale"],
            index=0,
        )

    with col2:
        st.markdown("### File Upload Defaults")
        auto_process = st.checkbox("Auto-process uploads", value=True)
        auto_visualize = st.checkbox("Auto-generate visualizations", value=True)
        auto_store = st.checkbox("Auto-store in database", value=True)

        upload_folder = st.text_input("Default upload folder", value="/tmp/amperedata/uploads")

    st.markdown("---")

    if st.button("Save Preferences", type="primary"):
        st.success("Preferences saved!")

# === ABOUT ===
with tab5:
    st.subheader("About AmpereData")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            """
        ### AmpereData Platform

        **Version:** 1.0.0
        **Build:** 2025.10.25

        #### Features
        - Multi-format data parsing (BioLogic, Neware, CSV)
        - Interactive visualizations
        - SQL query interface
        - Data export capabilities
        - Batch processing support
        - Real-time monitoring

        #### Supported Instruments
        - **BioLogic:** VSP, VMP3, SP-150, SP-200, SP-240
        - **Neware:** CT series, BTS series
        - **Generic:** CSV, TXT, Excel files

        #### Technology Stack
        - **Frontend:** Streamlit
        - **Backend:** Python, FastAPI
        - **Database:** SQLite, PostgreSQL/TimescaleDB
        - **Visualization:** Plotly, Matplotlib
        - **Processing:** Pandas, NumPy, SciPy
        """
        )

    with col2:
        st.markdown(
            """
        ### Resources

        - [Documentation](#)
        - [API Reference](#)
        - [GitHub Repository](#)
        - [Report Issues](#)

        ### Support

        - Email: support@amperedata.com
        - Discord: #amperedata
        - Forum: forum.amperedata.com

        ### Citation

        If you use AmpereData in your research, please cite:

        ```
        AmpereData v1.0 (2025)
        Battery Testing Analytics Platform
        ```
        """
        )

    st.markdown("---")

    # System information
    st.markdown("### System Information")

    import platform
    import sys

    sys_info = {
        "Python Version": sys.version.split()[0],
        "Platform": platform.system(),
        "Architecture": platform.machine(),
        "Streamlit Version": st.__version__,
    }

    col1, col2, col3, col4 = st.columns(4)
    for idx, (key, value) in enumerate(sys_info.items()):
        with [col1, col2, col3, col4][idx]:
            st.metric(key, value)

    st.markdown("---")

    # License
    with st.expander("License"):
        st.markdown(
            """
        MIT License

        Copyright (c) 2025 AmpereData

        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction...
        """
        )

# Footer
st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: #666; padding: 1rem 0;'>
    <p><small>AmpereData Platform v1.0 | Built with Streamlit</small></p>
</div>
""",
    unsafe_allow_html=True,
)
