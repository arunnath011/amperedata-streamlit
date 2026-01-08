"""Streamlit application for battery data visualization.

This module provides a comprehensive Streamlit web application for creating
interactive visualizations of battery testing data using the visualization framework.
"""

import io
import logging
from datetime import datetime, timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    import streamlit as st
    from plotly.subplots import make_subplots

    STREAMLIT_AVAILABLE = True
except ImportError:
    st = None
    go = None
    make_subplots = None
    STREAMLIT_AVAILABLE = False

from .components import create_chart
from .config import ChartConfigManager, TemplateManager, ThemeManager
from .models import (
    ChartConfig,
    ChartData,
    ChartTemplate,
    ChartType,
    ExportConfig,
    ExportFormat,
    VisualizationTheme,
)
from .templates import BatteryAnalysisTemplates
from .utils import ChartExporter, DataProcessor

logger = logging.getLogger(__name__)


class StreamlitVisualizationApp:
    """Main Streamlit application for battery data visualization."""

    def __init__(self):
        """Initialize the Streamlit application."""
        if not STREAMLIT_AVAILABLE:
            raise ImportError("Streamlit is not available. Install with: pip install streamlit")

        self.config_manager = ChartConfigManager()
        self.theme_manager = ThemeManager()
        self.template_manager = TemplateManager()
        self.battery_templates = BatteryAnalysisTemplates()
        self.data_processor = DataProcessor()
        self.chart_exporter = ChartExporter()

        # Initialize session state
        self._initialize_session_state()

    def _initialize_session_state(self) -> None:
        """Initialize Streamlit session state variables."""
        if "current_config" not in st.session_state:
            st.session_state.current_config = None
        if "current_data" not in st.session_state:
            st.session_state.current_data = None
        if "current_theme" not in st.session_state:
            st.session_state.current_theme = self.theme_manager.get_theme_by_name("Light")
        if "chart_history" not in st.session_state:
            st.session_state.chart_history = []
        if "real_time_enabled" not in st.session_state:
            st.session_state.real_time_enabled = False

    def run(self) -> None:
        """Run the main Streamlit application."""
        try:
            # Configure page
            st.set_page_config(
                page_title="AmpereData Visualization",
                page_icon="🔋",
                layout="wide",
                initial_sidebar_state="expanded",
            )

            # Apply custom CSS
            self._apply_custom_css()

            # Main header
            st.title("🔋 AmpereData Visualization Framework")
            st.markdown("Interactive battery data analysis and visualization")

            # Sidebar navigation
            page = self._render_sidebar()

            # Main content area
            if page == "Chart Builder":
                self._render_chart_builder()
            elif page == "Template Gallery":
                self._render_template_gallery()
            elif page == "Data Upload":
                self._render_data_upload()
            elif page == "Dashboard":
                self._render_dashboard()
            elif page == "Real-time Monitoring":
                self._render_real_time_monitoring()
            elif page == "Export & Share":
                self._render_export_share()
            elif page == "Settings":
                self._render_settings()

        except Exception as e:
            logger.error(f"Streamlit app error: {str(e)}")
            st.error(f"Application error: {str(e)}")

    def _apply_custom_css(self) -> None:
        """Apply custom CSS styling."""
        css = """
        <style>
        .main-header {
            background: linear-gradient(90deg, #1f77b4, #ff7f0e);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            margin-bottom: 2rem;
        }

        .metric-card {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #1f77b4;
            margin: 0.5rem 0;
        }

        .chart-container {
            background: white;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 1rem 0;
        }

        .sidebar-section {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }

        .template-card {
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .template-card:hover {
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            transform: translateY(-2px);
        }
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)

    def _render_sidebar(self) -> str:
        """Render the sidebar navigation."""
        st.sidebar.title("🔋 Navigation")

        pages = [
            "Chart Builder",
            "Template Gallery",
            "Data Upload",
            "Dashboard",
            "Real-time Monitoring",
            "Export & Share",
            "Settings",
        ]

        page = st.sidebar.selectbox("Select Page", pages)

        # Theme selector
        st.sidebar.markdown("---")
        st.sidebar.subheader("🎨 Theme")
        themes = self.theme_manager.list_themes()
        theme_names = [t.name for t in themes]

        current_theme_name = (
            st.session_state.current_theme.name if st.session_state.current_theme else "Light"
        )
        selected_theme_name = st.sidebar.selectbox(
            "Select Theme",
            theme_names,
            index=theme_names.index(current_theme_name) if current_theme_name in theme_names else 0,
        )

        if selected_theme_name != current_theme_name:
            st.session_state.current_theme = self.theme_manager.get_theme_by_name(
                selected_theme_name
            )
            st.experimental_rerun()

        # Quick stats
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Quick Stats")
        if st.session_state.current_data is not None:
            data = st.session_state.current_data
            st.sidebar.metric("Rows", len(data))
            st.sidebar.metric("Columns", len(data.columns))
            st.sidebar.metric("Memory Usage", f"{data.memory_usage(deep=True).sum() / 1024:.1f} KB")

        return page

    def _render_chart_builder(self) -> None:
        """Render the chart builder interface."""
        st.header("📈 Chart Builder")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Configuration")

            # Chart type selection
            chart_types = [ct.value for ct in ChartType]
            selected_type = st.selectbox("Chart Type", chart_types)

            # Data source
            data_source = st.radio("Data Source", ["Upload File", "Sample Data", "Current Data"])

            if data_source == "Upload File":
                uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "json"])
                if uploaded_file:
                    data = self._load_data_file(uploaded_file)
                    st.session_state.current_data = data

            elif data_source == "Sample Data":
                sample_type = st.selectbox(
                    "Sample Type", ["Cycling Data", "Capacity Fade", "Impedance"]
                )
                data = self._generate_sample_data(sample_type)
                st.session_state.current_data = data

            # Column mapping
            if st.session_state.current_data is not None:
                data = st.session_state.current_data
                st.subheader("Column Mapping")

                columns = list(data.columns)
                x_col = st.selectbox("X-axis", columns)
                y_col = st.selectbox("Y-axis", columns)

                # Optional columns
                color_col = st.selectbox("Color (optional)", ["None"] + columns)
                size_col = st.selectbox("Size (optional)", ["None"] + columns)

                # Chart styling
                st.subheader("Styling")
                color = st.color_picker("Primary Color", "#1f77b4")
                line_width = st.slider("Line Width", 0.5, 5.0, 2.0, 0.1)
                marker_size = st.slider("Marker Size", 0, 20, 6)
                opacity = st.slider("Opacity", 0.1, 1.0, 0.8, 0.1)

                # Create chart
                if st.button("Create Chart", type="primary"):
                    try:
                        config = self._create_chart_config(
                            selected_type,
                            data,
                            x_col,
                            y_col,
                            color_col if color_col != "None" else None,
                            size_col if size_col != "None" else None,
                            color,
                            line_width,
                            marker_size,
                            opacity,
                        )
                        st.session_state.current_config = config
                        st.success("Chart created successfully!")
                    except Exception as e:
                        st.error(f"Failed to create chart: {str(e)}")

        with col2:
            st.subheader("Preview")

            if st.session_state.current_config:
                try:
                    chart = create_chart(st.session_state.current_config)
                    figure = chart.render()

                    # Apply theme
                    if st.session_state.current_theme:
                        self._apply_theme_to_figure(figure, st.session_state.current_theme)

                    st.plotly_chart(figure, use_container_width=True)

                    # Chart actions
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        if st.button("Save Configuration"):
                            self._save_chart_config()
                    with col_b:
                        if st.button("Export Chart"):
                            self._export_chart(figure)
                    with col_c:
                        if st.button("Add to Dashboard"):
                            self._add_to_dashboard()

                except Exception as e:
                    st.error(f"Failed to render chart: {str(e)}")
            else:
                st.info("Configure and create a chart to see the preview")

    def _render_template_gallery(self) -> None:
        """Render the template gallery."""
        st.header("🎨 Template Gallery")

        # Filter options
        col1, col2, col3 = st.columns(3)
        with col1:
            category_filter = st.selectbox(
                "Category", ["All", "Battery Analysis", "Basic", "Analysis"]
            )
        with col2:
            chart_type_filter = st.selectbox("Chart Type", ["All"] + [ct.value for ct in ChartType])
        with col3:
            search_query = st.text_input("Search Templates")

        # Get templates
        templates = self.battery_templates.list_templates()

        # Apply filters
        if category_filter != "All":
            templates = [t for t in templates if t.category == category_filter]
        if chart_type_filter != "All":
            templates = [t for t in templates if t.chart_type.value == chart_type_filter]
        if search_query:
            templates = [
                t
                for t in templates
                if search_query.lower() in t.name.lower()
                or search_query.lower() in (t.description or "").lower()
            ]

        # Display templates
        if templates:
            cols = st.columns(3)
            for i, template in enumerate(templates):
                with cols[i % 3]:
                    self._render_template_card(template)
        else:
            st.info("No templates found matching the criteria")

    def _render_template_card(self, template: ChartTemplate) -> None:
        """Render a template card."""
        with st.container():
            st.markdown(
                f"""
            <div class="template-card">
                <h4>{template.name}</h4>
                <p><strong>Type:</strong> {template.chart_type.value}</p>
                <p><strong>Category:</strong> {template.category}</p>
                <p>{template.description}</p>
                <p><small>Tags: {', '.join(template.tags)}</small></p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            if st.button("Use Template", key=f"use_{template.id}"):
                self._apply_template(template)

    def _render_data_upload(self) -> None:
        """Render the data upload interface."""
        st.header("📁 Data Upload")

        upload_method = st.radio(
            "Upload Method", ["File Upload", "Database Connection", "API Connection"]
        )

        if upload_method == "File Upload":
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=["csv", "xlsx", "json", "parquet"],
                help="Supported formats: CSV, Excel, JSON, Parquet",
            )

            if uploaded_file:
                try:
                    data = self._load_data_file(uploaded_file)
                    st.session_state.current_data = data

                    st.success(
                        f"File uploaded successfully! {len(data)} rows, {len(data.columns)} columns"
                    )

                    # Data preview
                    st.subheader("Data Preview")
                    st.dataframe(data.head(100))

                    # Data info
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Data Info")
                        buffer = io.StringIO()
                        data.info(buf=buffer)
                        st.text(buffer.getvalue())

                    with col2:
                        st.subheader("Statistics")
                        st.dataframe(data.describe())

                except Exception as e:
                    st.error(f"Failed to load file: {str(e)}")

        elif upload_method == "Database Connection":
            st.subheader("Database Connection")

            st.selectbox("Database Type", ["PostgreSQL", "MySQL", "SQLite"])
            st.text_input("Host", "localhost")
            st.number_input("Port", value=5432)
            st.text_input("Database")
            st.text_input("Username")
            st.text_input("Password", type="password")
            st.text_area("SQL Query", "SELECT * FROM experiments LIMIT 1000")

            if st.button("Connect and Load"):
                st.info("Database connection functionality would be implemented here")

        elif upload_method == "API Connection":
            st.subheader("API Connection")

            st.text_input("API URL")
            st.text_area("Headers (JSON)", "{}")
            st.text_area("Parameters (JSON)", "{}")

            if st.button("Connect and Load"):
                st.info("API connection functionality would be implemented here")

    def _render_dashboard(self) -> None:
        """Render the dashboard interface."""
        st.header("📊 Dashboard")

        # Dashboard layout options
        col1, col2 = st.columns([1, 3])

        with col1:
            st.subheader("Dashboard Settings")

            layout_type = st.selectbox("Layout", ["Grid", "Tabs", "Accordion"])
            columns = st.slider("Columns", 1, 4, 2)

            # Chart selection
            st.subheader("Charts")
            if st.session_state.chart_history:
                selected_charts = st.multiselect(
                    "Select Charts",
                    options=range(len(st.session_state.chart_history)),
                    format_func=lambda x: f"Chart {x+1}",
                    default=list(range(min(4, len(st.session_state.chart_history)))),
                )
            else:
                st.info("No charts available. Create charts first.")
                selected_charts = []

        with col2:
            st.subheader("Dashboard Preview")

            if selected_charts and st.session_state.chart_history:
                if layout_type == "Grid":
                    self._render_grid_dashboard(selected_charts, columns)
                elif layout_type == "Tabs":
                    self._render_tab_dashboard(selected_charts)
                elif layout_type == "Accordion":
                    self._render_accordion_dashboard(selected_charts)
            else:
                st.info("Select charts to display in the dashboard")

    def _render_real_time_monitoring(self) -> None:
        """Render the real-time monitoring interface."""
        st.header("⚡ Real-time Monitoring")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Configuration")

            # Enable/disable real-time
            real_time_enabled = st.checkbox(
                "Enable Real-time Updates", value=st.session_state.real_time_enabled
            )
            st.session_state.real_time_enabled = real_time_enabled

            if real_time_enabled:
                st.slider("Update Interval (seconds)", 1, 60, 5)
                st.selectbox("Data Source", ["Simulated", "Database", "API"])

                # Chart configuration
                st.selectbox("Chart Type", ["Line", "Gauge", "Bar"])
                st.number_input("Max Data Points", 10, 1000, 100)

                if st.button("Start Monitoring"):
                    st.success("Real-time monitoring started!")

        with col2:
            st.subheader("Live Chart")

            if st.session_state.real_time_enabled:
                # Placeholder for real-time chart
                placeholder = st.empty()

                # Simulate real-time data
                if st.button("Generate Sample Data"):
                    data = self._generate_real_time_data()

                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=data["timestamp"],
                            y=data["voltage"],
                            mode="lines",
                            name="Voltage",
                        )
                    )
                    fig.update_layout(
                        title="Real-time Battery Voltage",
                        xaxis_title="Time",
                        yaxis_title="Voltage (V)",
                    )

                    placeholder.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Enable real-time updates to see live data")

    def _render_export_share(self) -> None:
        """Render the export and sharing interface."""
        st.header("📤 Export & Share")

        if st.session_state.current_config:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Export Options")

                export_format = st.selectbox("Format", [f.value for f in ExportFormat])
                width = st.number_input("Width (px)", 400, 2000, 800)
                height = st.number_input("Height (px)", 300, 1500, 600)
                scale = st.slider("Scale", 0.5, 3.0, 1.0, 0.1)

                filename = st.text_input("Filename", f"chart.{export_format}")

                if st.button("Export Chart", type="primary"):
                    try:
                        export_config = ExportConfig(
                            format=ExportFormat(export_format),
                            width=width,
                            height=height,
                            scale=scale,
                            filename=filename,
                        )

                        chart = create_chart(st.session_state.current_config)
                        chart.render()

                        # Export chart
                        exported_file = chart.export(export_config)
                        st.success(f"Chart exported to: {exported_file}")

                        # Provide download link
                        with open(exported_file, "rb") as f:
                            st.download_button(
                                label="Download File",
                                data=f.read(),
                                file_name=filename,
                                mime=self._get_mime_type(export_format),
                            )

                    except Exception as e:
                        st.error(f"Export failed: {str(e)}")

            with col2:
                st.subheader("Sharing Options")

                st.checkbox("Make Public")
                st.text_input("Password (optional)", type="password")
                st.date_input("Expiration Date")

                if st.button("Generate Share Link"):
                    # Generate a mock share link
                    share_id = f"chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    share_url = f"https://amperedata.com/share/{share_id}"

                    st.success("Share link generated!")
                    st.code(share_url)

                    # Embed code
                    st.subheader("Embed Code")
                    embed_code = f'<iframe src="{share_url}" width="800" height="600"></iframe>'
                    st.code(embed_code, language="html")
        else:
            st.info("Create a chart first to enable export and sharing options")

    def _render_settings(self) -> None:
        """Render the settings interface."""
        st.header("⚙️ Settings")

        tab1, tab2, tab3 = st.tabs(["General", "Themes", "Data"])

        with tab1:
            st.subheader("General Settings")

            st.checkbox("Auto-save configurations", True)
            st.checkbox("Show tooltips", True)
            st.checkbox("Enable animations", True)

            st.number_input("Default chart width", 400, 2000, 800)
            st.number_input("Default chart height", 300, 1500, 600)

            if st.button("Save General Settings"):
                st.success("Settings saved!")

        with tab2:
            st.subheader("Theme Management")

            # Theme editor
            st.write("Create Custom Theme")
            theme_name = st.text_input("Theme Name")

            col1, col2 = st.columns(2)
            with col1:
                st.color_picker("Primary Color", "#1f77b4")
                st.color_picker("Secondary Color", "#ff7f0e")
                st.color_picker("Success Color", "#2ca02c")

            with col2:
                st.color_picker("Background Color", "#ffffff")
                st.color_picker("Text Color", "#212529")
                st.color_picker("Grid Color", "#e9ecef")

            if st.button("Create Theme"):
                st.success(f"Theme '{theme_name}' created!")

        with tab3:
            st.subheader("Data Settings")

            st.number_input("Max rows to display", 100, 10000, 1000)
            st.checkbox("Enable data caching", True)
            st.checkbox("Auto-detect data types", True)

            # Data validation
            st.write("Data Validation Rules")
            st.checkbox("Validate data ranges")
            st.checkbox("Validate data types")

            if st.button("Save Data Settings"):
                st.success("Data settings saved!")

    def _load_data_file(self, uploaded_file) -> pd.DataFrame:
        """Load data from uploaded file."""
        try:
            file_extension = uploaded_file.name.split(".")[-1].lower()

            if file_extension == "csv":
                return pd.read_csv(uploaded_file)
            elif file_extension in ["xlsx", "xls"]:
                return pd.read_excel(uploaded_file)
            elif file_extension == "json":
                return pd.read_json(uploaded_file)
            elif file_extension == "parquet":
                return pd.read_parquet(uploaded_file)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")

        except Exception as e:
            logger.error(f"Failed to load data file: {str(e)}")
            raise

    def _generate_sample_data(self, sample_type: str) -> pd.DataFrame:
        """Generate sample data for testing."""
        np.random.seed(42)

        if sample_type == "Cycling Data":
            time = np.linspace(0, 10, 1000)
            voltage = 3.7 + 0.5 * np.sin(2 * np.pi * time / 2) + 0.1 * np.random.randn(1000)
            current = 1.0 + 0.3 * np.cos(2 * np.pi * time / 2) + 0.05 * np.random.randn(1000)
            capacity = np.cumsum(current * (time[1] - time[0]))

            return pd.DataFrame(
                {
                    "time": time,
                    "voltage": voltage,
                    "current": current,
                    "capacity": capacity,
                    "temperature": 25 + 5 * np.random.randn(1000),
                }
            )

        elif sample_type == "Capacity Fade":
            cycles = np.arange(1, 501)
            capacity = 2.5 * np.exp(-cycles / 1000) + 0.05 * np.random.randn(500)
            efficiency = 99.5 - 0.001 * cycles + 0.1 * np.random.randn(500)

            return pd.DataFrame(
                {
                    "cycle_number": cycles,
                    "discharge_capacity": capacity,
                    "coulombic_efficiency": efficiency / 100,
                }
            )

        elif sample_type == "Impedance":
            freq = np.logspace(-2, 5, 100)
            real_z = 0.1 + 0.05 / np.sqrt(freq)
            imag_z = -0.02 * np.sqrt(freq)

            return pd.DataFrame(
                {
                    "frequency": freq,
                    "real_impedance": real_z,
                    "imaginary_impedance": imag_z,
                }
            )

        else:
            # Default sample data
            x = np.linspace(0, 10, 100)
            y = np.sin(x) + 0.1 * np.random.randn(100)

            return pd.DataFrame({"x": x, "y": y})

    def _create_chart_config(
        self,
        chart_type: str,
        data: pd.DataFrame,
        x_col: str,
        y_col: str,
        color_col: Optional[str] = None,
        size_col: Optional[str] = None,
        color: str = "#1f77b4",
        line_width: float = 2.0,
        marker_size: int = 6,
        opacity: float = 0.8,
    ) -> ChartConfig:
        """Create chart configuration from parameters."""
        from .models import AxisConfig, ChartStyle, LayoutConfig

        # Prepare chart data
        chart_data = ChartData(
            x=data[x_col].tolist(),
            y=data[y_col].tolist(),
            color=data[color_col].tolist() if color_col else None,
            size=data[size_col].tolist() if size_col else None,
        )

        # Create configuration
        config = ChartConfig(
            type=ChartType(chart_type),
            title=f"{y_col} vs {x_col}",
            data=chart_data,
            style=ChartStyle(
                color=color,
                line_width=line_width,
                marker_size=marker_size,
                opacity=opacity,
            ),
            x_axis=AxisConfig(title=x_col.replace("_", " ").title(), show_grid=True),
            y_axis=AxisConfig(title=y_col.replace("_", " ").title(), show_grid=True),
            layout=LayoutConfig(
                title=f"{y_col} vs {x_col}",
                show_legend=color_col is not None,
                height=500,
            ),
        )

        return config

    def _apply_theme_to_figure(self, figure: Any, theme: VisualizationTheme) -> None:
        """Apply theme to Plotly figure."""
        figure.update_layout(
            paper_bgcolor=theme.colors.get("background", "#ffffff"),
            plot_bgcolor=theme.colors.get("surface", "#ffffff"),
            font={
                "family": theme.fonts.get("primary", "Arial"),
                "color": theme.colors.get("text", "#212529"),
            },
        )

        figure.update_xaxes(gridcolor=theme.colors.get("grid", "#e9ecef"))
        figure.update_yaxes(gridcolor=theme.colors.get("grid", "#e9ecef"))

    def _apply_template(self, template: ChartTemplate) -> None:
        """Apply template to create chart configuration."""
        try:
            config = self.template_manager.apply_template_to_config(template)
            st.session_state.current_config = config
            st.success(f"Template '{template.name}' applied successfully!")
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Failed to apply template: {str(e)}")

    def _save_chart_config(self) -> None:
        """Save current chart configuration."""
        if st.session_state.current_config:
            try:
                file_path = self.config_manager.save_config(st.session_state.current_config)
                st.success(f"Configuration saved to: {file_path}")
            except Exception as e:
                st.error(f"Failed to save configuration: {str(e)}")

    def _export_chart(self, figure: Any) -> None:
        """Export chart to file."""
        try:
            # Create temporary export
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chart_{timestamp}.png"

            figure.write_image(filename, width=800, height=600, scale=2)

            with open(filename, "rb") as f:
                st.download_button(
                    label="Download Chart",
                    data=f.read(),
                    file_name=filename,
                    mime="image/png",
                )

        except Exception as e:
            st.error(f"Export failed: {str(e)}")

    def _add_to_dashboard(self) -> None:
        """Add current chart to dashboard."""
        if st.session_state.current_config:
            st.session_state.chart_history.append(st.session_state.current_config)
            st.success("Chart added to dashboard!")

    def _render_grid_dashboard(self, selected_charts: list[int], columns: int) -> None:
        """Render grid-based dashboard."""
        for i in range(0, len(selected_charts), columns):
            cols = st.columns(columns)
            for j, col in enumerate(cols):
                if i + j < len(selected_charts):
                    chart_idx = selected_charts[i + j]
                    config = st.session_state.chart_history[chart_idx]

                    with col:
                        try:
                            chart = create_chart(config)
                            figure = chart.render()
                            st.plotly_chart(figure, use_container_width=True)
                        except Exception as e:
                            st.error(f"Chart {chart_idx + 1}: {str(e)}")

    def _render_tab_dashboard(self, selected_charts: list[int]) -> None:
        """Render tab-based dashboard."""
        tab_names = [f"Chart {i+1}" for i in selected_charts]
        tabs = st.tabs(tab_names)

        for i, tab in enumerate(tabs):
            with tab:
                chart_idx = selected_charts[i]
                config = st.session_state.chart_history[chart_idx]

                try:
                    chart = create_chart(config)
                    figure = chart.render()
                    st.plotly_chart(figure, use_container_width=True)
                except Exception as e:
                    st.error(f"Chart error: {str(e)}")

    def _render_accordion_dashboard(self, selected_charts: list[int]) -> None:
        """Render accordion-based dashboard."""
        for i, chart_idx in enumerate(selected_charts):
            config = st.session_state.chart_history[chart_idx]

            with st.expander(f"Chart {chart_idx + 1}: {config.title}", expanded=i == 0):
                try:
                    chart = create_chart(config)
                    figure = chart.render()
                    st.plotly_chart(figure, use_container_width=True)
                except Exception as e:
                    st.error(f"Chart error: {str(e)}")

    def _generate_real_time_data(self) -> pd.DataFrame:
        """Generate simulated real-time data."""
        now = datetime.now()
        timestamps = [now - timedelta(seconds=i) for i in range(60, 0, -1)]
        voltages = 3.7 + 0.1 * np.random.randn(60)

        return pd.DataFrame({"timestamp": timestamps, "voltage": voltages})

    def _get_mime_type(self, format: str) -> str:
        """Get MIME type for export format."""
        mime_types = {
            "png": "image/png",
            "jpeg": "image/jpeg",
            "svg": "image/svg+xml",
            "pdf": "application/pdf",
            "html": "text/html",
            "json": "application/json",
        }
        return mime_types.get(format, "application/octet-stream")


# Main entry point
def main():
    """Main entry point for the Streamlit app."""
    if not STREAMLIT_AVAILABLE:
        print("Streamlit is not available. Install with: pip install streamlit")
        return

    app = StreamlitVisualizationApp()
    app.run()


if __name__ == "__main__":
    main()
