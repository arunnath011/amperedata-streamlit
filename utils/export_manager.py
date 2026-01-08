"""
Export Manager - Advanced Data Export Utilities
================================================

Supports multiple export formats:
- CSV (standard, Excel-compatible)
- JSON (structured, nested)
- Excel (formatted, multi-sheet)
- HDF5 (large datasets)
- Parquet (compressed, fast)
- Publication-ready charts (PNG, SVG, PDF)
"""

import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


class ExportManager:
    """Manages data export in various formats."""

    SUPPORTED_FORMATS = ["csv", "json", "excel", "parquet", "hdf5"]

    def __init__(self, output_dir: str = "exports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_dataframe(
        self, df: pd.DataFrame, filename: str, format: str = "csv", **kwargs
    ) -> str:
        """
        Export DataFrame to specified format.

        Args:
            df: DataFrame to export
            filename: Output filename (without extension)
            format: Export format ('csv', 'json', 'excel', 'parquet', 'hdf5')
            **kwargs: Format-specific options

        Returns:
            Path to exported file
        """
        if format not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {format}. Choose from {self.SUPPORTED_FORMATS}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{filename}_{timestamp}"

        if format == "csv":
            return self._export_csv(df, base_filename, **kwargs)
        elif format == "json":
            return self._export_json(df, base_filename, **kwargs)
        elif format == "excel":
            return self._export_excel(df, base_filename, **kwargs)
        elif format == "parquet":
            return self._export_parquet(df, base_filename, **kwargs)
        elif format == "hdf5":
            return self._export_hdf5(df, base_filename, **kwargs)

    def _export_csv(
        self,
        df: pd.DataFrame,
        filename: str,
        compression: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Export as CSV with optional compression."""
        if compression:
            output_path = self.output_dir / f"{filename}.csv.{compression}"
        else:
            output_path = self.output_dir / f"{filename}.csv"

        df.to_csv(output_path, index=False, compression=compression, **kwargs)

        return str(output_path)

    def _export_json(
        self,
        df: pd.DataFrame,
        filename: str,
        orient: str = "records",
        indent: int = 2,
        **kwargs,
    ) -> str:
        """Export as JSON with structured format."""
        output_path = self.output_dir / f"{filename}.json"

        df.to_json(output_path, orient=orient, indent=indent, **kwargs)

        return str(output_path)

    def _export_excel(
        self, df: pd.DataFrame, filename: str, sheet_name: str = "Data", **kwargs
    ) -> str:
        """Export as Excel with formatting."""
        output_path = self.output_dir / f"{filename}.xlsx"

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False, **kwargs)

            # Auto-adjust column widths
            worksheet = writer.sheets[sheet_name]
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(cell.value)
                    except Exception:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width

        return str(output_path)

    def _export_parquet(
        self, df: pd.DataFrame, filename: str, compression: str = "snappy", **kwargs
    ) -> str:
        """Export as Parquet (compressed columnar format)."""
        output_path = self.output_dir / f"{filename}.parquet"

        df.to_parquet(output_path, compression=compression, index=False, **kwargs)

        return str(output_path)

    def _export_hdf5(self, df: pd.DataFrame, filename: str, key: str = "data", **kwargs) -> str:
        """Export as HDF5 (efficient for large datasets)."""
        output_path = self.output_dir / f"{filename}.h5"

        df.to_hdf(output_path, key=key, mode="w", index=False, **kwargs)

        return str(output_path)

    def export_multi_sheet_excel(
        self,
        data_dict: dict[str, pd.DataFrame],
        filename: str,
        format_options: Optional[dict] = None,
    ) -> str:
        """
        Export multiple DataFrames as Excel workbook.

        Args:
            data_dict: Dict of sheet_name -> DataFrame
            filename: Output filename
            format_options: Optional formatting options

        Returns:
            Path to exported file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"{filename}_{timestamp}.xlsx"

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            for sheet_name, df in data_dict.items():
                df.to_excel(
                    writer, sheet_name=sheet_name[:31], index=False
                )  # Excel has 31 char limit

                # Auto-adjust columns
                worksheet = writer.sheets[sheet_name[:31]]
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(cell.value)
                        except Exception:
                            pass
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[column_letter].width = adjusted_width

        return str(output_path)

    def export_archive(
        self, files: list[str], archive_name: str, compression: str = "ZIP_DEFLATED"
    ) -> str:
        """
        Create compressed archive of multiple files.

        Args:
            files: List of file paths to include
            archive_name: Output archive name
            compression: Compression method

        Returns:
            Path to created archive
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = self.output_dir / f"{archive_name}_{timestamp}.zip"

        compression_method = getattr(zipfile, compression)

        with zipfile.ZipFile(archive_path, "w", compression_method) as zipf:
            for file_path in files:
                zipf.write(file_path, Path(file_path).name)

        return str(archive_path)

    def create_export_bundle(
        self,
        data: pd.DataFrame,
        filename: str,
        formats: list[str] = None,
    ) -> str:
        """
        Create bundle with data in multiple formats.

        Args:
            data: DataFrame to export
            filename: Base filename
            formats: List of formats to include

        Returns:
            Path to bundle archive
        """
        if formats is None:
            formats = ["csv", "json", "excel"]
        exported_files = []

        for fmt in formats:
            if fmt in self.SUPPORTED_FORMATS:
                file_path = self.export_dataframe(data, filename, fmt)
                exported_files.append(file_path)

        # Create archive
        archive_path = self.export_archive(exported_files, f"{filename}_bundle")

        # Clean up individual files
        for file_path in exported_files:
            Path(file_path).unlink()

        return archive_path


class ChartExporter:
    """Export Plotly charts in various formats."""

    def __init__(self, output_dir: str = "exports/charts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_static_image(
        self,
        fig,
        filename: str,
        format: str = "png",
        width: int = 1920,
        height: int = 1080,
        scale: float = 2.0,
    ) -> str:
        """
        Export chart as static image.

        Args:
            fig: Plotly figure
            filename: Output filename
            format: Image format ('png', 'jpeg', 'svg', 'pdf')
            width: Image width in pixels
            height: Image height in pixels
            scale: Resolution scale factor

        Returns:
            Path to exported image
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"{filename}_{timestamp}.{format}"

        try:
            fig.write_image(
                str(output_path), format=format, width=width, height=height, scale=scale
            )
            return str(output_path)
        except Exception:
            # Fallback to HTML if image export fails (requires kaleido)
            html_path = self.output_dir / f"{filename}_{timestamp}.html"
            fig.write_html(str(html_path))
            return str(html_path)

    def export_interactive_html(
        self,
        fig,
        filename: str,
        include_plotlyjs: str = "cdn",
        config: Optional[dict] = None,
    ) -> str:
        """
        Export chart as interactive HTML.

        Args:
            fig: Plotly figure
            filename: Output filename
            include_plotlyjs: How to include plotly.js ('cdn', 'directory', True)
            config: Chart configuration options

        Returns:
            Path to exported HTML
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"{filename}_{timestamp}.html"

        default_config = {
            "displayModeBar": True,
            "displaylogo": False,
            "toImageButtonOptions": {
                "format": "png",
                "filename": filename,
                "height": 1080,
                "width": 1920,
                "scale": 2,
            },
        }

        if config:
            default_config.update(config)

        fig.write_html(str(output_path), include_plotlyjs=include_plotlyjs, config=default_config)

        return str(output_path)


def create_data_report(
    data: pd.DataFrame,
    report_name: str,
    include_stats: bool = True,
    include_plots: bool = True,
) -> str:
    """
    Create comprehensive data report with statistics and visualizations.

    Args:
        data: DataFrame to analyze
        report_name: Name for the report
        include_stats: Include statistical summary
        include_plots: Include distribution plots

    Returns:
        Path to report file
    """
    export_mgr = ExportManager(output_dir="exports/reports")

    # Create report sections
    report_data = {}

    # Basic info
    info_df = pd.DataFrame(
        {
            "Metric": ["Rows", "Columns", "Memory Usage", "Missing Values"],
            "Value": [
                len(data),
                len(data.columns),
                f"{data.memory_usage(deep=True).sum() / 1024:.2f} KB",
                data.isnull().sum().sum(),
            ],
        }
    )
    report_data["Info"] = info_df

    # Data types
    types_df = pd.DataFrame(
        {
            "Column": data.columns,
            "Type": data.dtypes.astype(str),
            "Non-Null": data.count(),
            "Null": data.isnull().sum(),
        }
    )
    report_data["Columns"] = types_df

    # Statistics
    if include_stats:
        stats_df = data.describe().T
        stats_df["column"] = stats_df.index
        report_data["Statistics"] = stats_df

    # Create Excel report
    output_path = export_mgr.export_multi_sheet_excel(report_data, report_name)

    return output_path
