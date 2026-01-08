"""
AmpereData Streamlit Application
=================================
Main entry point for the multi-page Streamlit UI.

Features:
- File upload and processing
- Interactive visualizations
- Data exploration
- Batch processing
- System monitoring
- User authentication and access control
"""

import sys
from pathlib import Path

import streamlit as st

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import authentication
from utils.auth import init_session_state, require_auth, show_logout_button

# Page configuration
st.set_page_config(
    page_title="AmpereData Platform",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/yourusername/amperedata",
        "Report a bug": "https://github.com/yourusername/amperedata/issues",
        "About": "# AmpereData\nBattery Testing Data Analytics Platform",
    },
)

# Authentication check
init_session_state()
require_auth()  # Require login to access platform

# Custom CSS for professional styling
st.markdown(
    """
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    h1 {
        color: #1a1a1a;
        padding-bottom: 1rem;
        border-bottom: 1px solid #e0e0e0;
        font-weight: 600;
    }
    .metric-card {
        background: #ffffff;
        padding: 1.25rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
        transition: box-shadow 0.2s ease;
    }
    .metric-card:hover {
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .metric-card h3 {
        color: #1a1a1a;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .metric-card p {
        color: #666;
        font-size: 0.9rem;
        margin: 0.25rem 0;
    }
    .metric-card small {
        color: #999;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Title and description
st.title("AmpereData Platform")
st.markdown(
    """
**Professional Battery Testing Data Analytics**
Upload, analyze, and visualize electrochemical battery testing data from multiple instruments.
"""
)

# Sidebar navigation
st.sidebar.title("Navigation")
st.sidebar.markdown("---")

# Show logout button in sidebar
show_logout_button()

# ELN Integration Section
st.sidebar.markdown("---")
st.sidebar.subheader("Lab Notebook")
st.sidebar.markdown(
    """
    <a href="http://localhost:3000" target="_blank"
       style="display: inline-flex; align-items: center; justify-content: center;
              padding: 0.6rem 1rem;
              background: linear-gradient(135deg, #030213 0%, #1a1a2e 100%);
              color: white;
              text-decoration: none;
              border-radius: 8px;
              width: 100%;
              font-weight: 500;
              box-shadow: 0 2px 4px rgba(0,0,0,0.1);
              transition: all 0.2s ease;">
        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 8px;">
            <path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20"></path>
        </svg>
        Open AmpereData ELN
    </a>
    """,
    unsafe_allow_html=True,
)
st.sidebar.caption("Electronic Lab Notebook for experiments, materials, and project management.")

# Main dashboard metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Platform Status", value="Online", delta="Active")

with col2:
    st.metric(label="Total Experiments", value="34", delta="+2 today")

with col3:
    st.metric(label="Data Points", value="2.5M", delta="+150K")

with col4:
    st.metric(label="Storage Used", value="1.2 GB", delta="+85 MB")

st.markdown("---")

# Quick access cards
st.subheader("Quick Access")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
    <div class="metric-card">
        <h3>Upload Data</h3>
        <p>Upload new battery testing files for analysis</p>
        <p><small>Supported: BioLogic, Neware, CSV, Excel</small></p>
    </div>
    """,
        unsafe_allow_html=True,
    )
    if st.button("Go to Upload →", key="upload_btn", use_container_width=True):
        st.switch_page("pages/1_Upload_Data.py")

with col2:
    st.markdown(
        """
    <div class="metric-card">
        <h3>View Dashboards</h3>
        <p>Interactive visualizations and analysis</p>
        <p><small>Cycle life, EIS, capacity fade, dQ/dV</small></p>
    </div>
    """,
        unsafe_allow_html=True,
    )
    if st.button("Go to Dashboard →", key="viz_btn", use_container_width=True):
        st.switch_page("pages/2_Visualizations.py")

with col3:
    st.markdown(
        """
    <div class="metric-card">
        <h3>Explore Data</h3>
        <p>Query and explore your datasets</p>
        <p><small>SQL queries, filters, exports</small></p>
    </div>
    """,
        unsafe_allow_html=True,
    )
    if st.button("Go to Explorer →", key="explore_btn", use_container_width=True):
        st.switch_page("pages/3_Data_Explorer.py")

# Second row of quick access cards
col4, col5, col6 = st.columns(3)

with col4:
    st.markdown(
        """
    <div class="metric-card">
        <h3>Live Dashboard</h3>
        <p>Real-time monitoring from connected cyclers</p>
        <p><small>Auto-refresh, alerts, streaming data</small></p>
    </div>
    """,
        unsafe_allow_html=True,
    )
    if st.button("Go to Live →", key="live_btn", use_container_width=True):
        st.switch_page("pages/7_Live_Dashboard.py")

with col5:
    st.markdown(
        """
    <div class="metric-card">
        <h3>Advanced Charts</h3>
        <p>In-depth analysis and visualizations</p>
        <p><small>Voltage profiles, dQ/dV, SOH tracking</small></p>
    </div>
    """,
        unsafe_allow_html=True,
    )
    if st.button("Go to Charts →", key="charts_btn", use_container_width=True):
        st.switch_page("pages/5_Advanced_Charts.py")

with col6:
    st.markdown(
        """
    <div class="metric-card">
        <h3>Build Sheets</h3>
        <p>Document cell builds and materials</p>
        <p><small>Track batches, link to test data</small></p>
    </div>
    """,
        unsafe_allow_html=True,
    )
    if st.button("Go to Build Sheets →", key="build_btn", use_container_width=True):
        st.switch_page("pages/6_Build_Sheet.py")

st.markdown("---")

# Recent activity
st.subheader("Recent Activity")

activity_data = [
    {
        "time": "10 minutes ago",
        "event": "NASA Battery B0005 - Analysis complete",
        "status": "Done",
    },
    {"time": "1 hour ago", "event": "Uploaded 3 BioLogic .mpt files", "status": "Done"},
    {
        "time": "2 hours ago",
        "event": "Generated EIS spectrum visualizations",
        "status": "Done",
    },
    {"time": "Today", "event": "Processed 34 battery datasets", "status": "Done"},
]

for activity in activity_data:
    col1, col2, col3 = st.columns([2, 6, 1])
    with col1:
        st.text(activity["time"])
    with col2:
        st.text(activity["event"])
    with col3:
        st.text(activity["status"])

st.markdown("---")

# System information
with st.expander("System Information"):
    sys_col1, sys_col2 = st.columns(2)

    with sys_col1:
        st.markdown(
            """
        **Platform Version:** 1.0.0
        **Database:** SQLite (nasa_amperedata_full.db)
        **Parsers Available:** BioLogic, Neware, Generic CSV
        **Features Enabled:** File Upload, Visualizations, Data Export
        """
        )

    with sys_col2:
        st.markdown(
            """
        **Python Version:** 3.11
        **Streamlit Version:** 1.x
        **Total Batteries:** 34
        **Total Cycles:** 2,500+
        """
        )

# Footer
st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p>AmpereData Platform v1.0 | Built with Streamlit</p>
    <p><small>For support, visit the documentation or contact your administrator</small></p>
</div>
""",
    unsafe_allow_html=True,
)
