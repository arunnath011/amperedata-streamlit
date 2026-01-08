# ⚡ AmpereData - Battery Data Analytics Platform

<div align="center">

**An open-source, comprehensive battery testing data analytics platform for researchers, engineers, and data scientists.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/arunnath011/amperedata-streamlit/actions/workflows/ci.yml/badge.svg)](https://github.com/arunnath011/amperedata-streamlit/actions/workflows/ci.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

[Features](#-features) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [Contributing](#-contributing) • [Troubleshooting](#-troubleshooting)

</div>

---

## 📖 Introduction

**AmpereData** is a powerful, web-based analytics platform designed specifically for battery testing data. Built with [Streamlit](https://streamlit.io/), it provides researchers and engineers with intuitive tools to:

- **Import** data from multiple battery cycler formats (BioLogic, Neware, Arbin, generic CSV/Excel)
- **Visualize** electrochemical data with interactive, publication-ready charts
- **Analyze** capacity fade, impedance spectra (EIS), and differential capacity (dQ/dV)
- **Predict** remaining useful life (RUL) using machine learning models
- **Export** processed data and reports in various formats

Whether you're conducting academic research, performing quality control in manufacturing, or developing new battery technologies, AmpereData streamlines your data workflow from raw files to actionable insights.

### Why AmpereData?

| Challenge | AmpereData Solution |
|-----------|---------------------|
| Multiple cycler formats | Universal parser supporting 10+ formats |
| Manual data cleaning | Intelligent column mapping & auto-detection |
| Complex visualizations | 15+ interactive chart types, one-click generation |
| Scattered analysis tools | All-in-one platform with consistent interface |
| No ML expertise | Pre-trained RUL prediction models ready to use |

---

## ✨ Features

### 📁 Universal Data Import
- **BioLogic** (.mpt, .mps) - Full support for EC-Lab files
- **Neware** (.nda, .ndax, .csv) - Native and exported formats
- **Arbin** (.res, .csv) - Research and exported formats
- **Generic CSV/Excel** - With intelligent column mapping
- **Batch upload** - Process multiple files simultaneously

### 📊 Interactive Visualizations
- **Voltage Profiles** - Charge/discharge curves with cycle selection
- **Capacity Fade** - Retention tracking with trend analysis
- **Coulombic Efficiency** - CE vs. cycle number
- **dQ/dV Analysis** - Differential capacity with peak detection
- **EIS Plots** - Nyquist and Bode diagrams
- **Rate Capability** - Multi-rate comparison
- **3D Surface Plots** - Voltage-capacity-cycle visualization
- **Heatmaps** - Correlation and degradation maps

### 🔬 Electrochemical Analysis
- Automatic cycle detection and segmentation
- Capacity, energy, and efficiency calculations
- Impedance fitting with equivalent circuit models
- State of Health (SOH) estimation
- Degradation mechanism identification

### 🤖 Machine Learning
- **RUL Prediction** - XGBoost model with 98% R² accuracy
- **Anomaly Detection** - Identify abnormal cycling behavior
- **Batch Comparison** - Statistical analysis across cells
- Pre-trained models included, no ML expertise required

### 📤 Export & Reporting
- CSV, Excel, JSON export formats
- Publication-ready figure export (PNG, SVG, PDF)
- Automated report generation
- Custom templates support

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Git

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/arunnath011/amperedata-streamlit.git
cd amperedata-streamlit

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
streamlit run streamlit_app.py
```

The application will open in your browser at **http://localhost:8501**

### Default Credentials

| Username | Password | Role |
|----------|----------|------|
| `admin` | `admin123` | Administrator |

⚠️ **Security Note**: Change the default password immediately after first login!

---

## 📚 Documentation

### Supported File Formats

| Format | Extensions | Parser | Notes |
|--------|------------|--------|-------|
| BioLogic EC-Lab | `.mpt`, `.mps` | `BiologicParser` | Full metadata support |
| Neware Binary | `.nda`, `.ndax` | `NewareParser` | Requires `NewareNDA` |
| Neware CSV | `.csv` | `NewareCSVParser` | Exported from BTSDA |
| Arbin | `.res`, `.csv` | `GenericCSVParser` | With column mapping |
| Generic CSV | `.csv`, `.txt` | `GenericCSVParser` | Auto-detect or manual mapping |
| Excel | `.xlsx`, `.xls` | `GenericCSVParser` | Multi-sheet support |

### Project Structure

```
amperedata-streamlit/
├── streamlit_app.py          # 🏠 Main entry point
├── pages/                    # 📄 Streamlit pages
│   ├── 1_Upload_Data.py      #    Data upload interface
│   ├── 2_Visualizations.py   #    Chart generation
│   ├── 3_Data_Explorer.py    #    SQL queries & filters
│   ├── 4_Settings.py         #    App configuration
│   ├── 5_Advanced_Charts.py  #    Complex visualizations
│   ├── 6_Build_Sheet.py      #    Cell documentation
│   └── 7_Live_Dashboard.py   #    Real-time monitoring
├── backend/                  # ⚙️ Backend services
│   ├── parsers/              #    File format parsers
│   ├── etl/                  #    ETL pipeline
│   └── ml/                   #    Machine learning models
├── frontend/                 # 🎨 Frontend components
│   ├── dashboard/            #    Dashboard builder
│   ├── electrochemical/      #    Analysis processors
│   └── visualization/        #    Chart components
├── utils/                    # 🔧 Utilities
│   ├── auth.py               #    Authentication
│   ├── column_mapper.py      #    Column mapping
│   └── export_manager.py     #    Data export
├── tests/                    # 🧪 Test suite
│   ├── unit/                 #    Unit tests
│   └── integration/          #    Integration tests
├── .github/workflows/        # 🔄 CI/CD pipelines
└── data/                     # 📂 Sample datasets
```

### Configuration

Create a `.streamlit/secrets.toml` file for sensitive configuration:

```toml
[database]
host = "localhost"
port = 5432
name = "amperedata"

[auth]
secret_key = "your-secret-key-here"
token_expiry = 3600
```

Environment variables can also be set in `.env`:

```bash
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
DATABASE_URL=postgresql://user:pass@localhost:5432/amperedata
```

---

## 🐳 Docker Deployment

### Using Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Services

| Service | Port | Description |
|---------|------|-------------|
| Streamlit | 8501 | Web application |
| PostgreSQL | 5432 | Database |
| Redis | 6379 | Caching layer |

### Production Deployment

For production deployments, consider:

1. **Reverse Proxy**: Use nginx or Traefik for SSL termination
2. **Environment Variables**: Never commit secrets to git
3. **Database Backups**: Configure automated PostgreSQL backups
4. **Monitoring**: Add Prometheus/Grafana for metrics

---

## 🔧 Troubleshooting

### Common Issues

<details>
<summary><strong>❌ ModuleNotFoundError: No module named 'xxx'</strong></summary>

**Cause**: Missing dependency or virtual environment not activated.

**Solution**:
```bash
# Ensure virtual environment is active
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Reinstall dependencies
pip install -r requirements.txt
```
</details>

<details>
<summary><strong>❌ Streamlit: Connection refused on localhost:8501</strong></summary>

**Cause**: Port already in use or Streamlit not running.

**Solution**:
```bash
# Check if port is in use
lsof -i :8501  # macOS/Linux
netstat -ano | findstr :8501  # Windows

# Kill the process or use a different port
streamlit run streamlit_app.py --server.port 8502
```
</details>

<details>
<summary><strong>❌ BioLogic file parsing fails</strong></summary>

**Cause**: Encoding issues or unsupported file version.

**Solution**:
- Ensure the file is exported from EC-Lab (not EC-Soft)
- Try re-exporting with UTF-8 encoding
- Check if the file opens correctly in EC-Lab
</details>

<details>
<summary><strong>❌ Neware .nda files not recognized</strong></summary>

**Cause**: `NewareNDA` library not installed.

**Solution**:
```bash
pip install NewareNDA
```
</details>

<details>
<summary><strong>❌ Database connection failed</strong></summary>

**Cause**: Database service not running or incorrect credentials.

**Solution**:
```bash
# If using Docker
docker-compose up -d postgres

# Check connection
psql -h localhost -U amperedata -d amperedata
```
</details>

<details>
<summary><strong>❌ Charts not rendering / blank page</strong></summary>

**Cause**: Browser cache or JavaScript errors.

**Solution**:
1. Hard refresh: `Ctrl+Shift+R` (Windows/Linux) or `Cmd+Shift+R` (Mac)
2. Clear browser cache
3. Try a different browser
4. Check browser console for errors (F12 → Console)
</details>

### Getting Help

If your issue isn't listed above:

1. **Search existing issues**: [GitHub Issues](https://github.com/arunnath011/amperedata-streamlit/issues)
2. **Create a new issue** with:
   - Python version (`python --version`)
   - OS and version
   - Full error traceback
   - Steps to reproduce
3. **Join discussions**: [GitHub Discussions](https://github.com/arunnath011/amperedata-streamlit/discussions)

---

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### Ways to Contribute

- 🐛 **Report bugs** - Open an issue with detailed reproduction steps
- 💡 **Suggest features** - Share ideas in Discussions
- 📝 **Improve documentation** - Fix typos, add examples
- 🔧 **Submit code** - Bug fixes, new features, tests
- 🌍 **Translations** - Help localize the interface

### Development Setup

```bash
# 1. Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/amperedata-streamlit.git
cd amperedata-streamlit

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install development dependencies
pip install -r requirements.txt
pip install pre-commit pytest pytest-cov black isort ruff

# 4. Install pre-commit hooks
pre-commit install

# 5. Create a feature branch
git checkout -b feature/your-feature-name
```

### Code Style

We use automated tools to maintain code quality:

- **Black** - Code formatting (line length: 100)
- **isort** - Import sorting
- **Ruff** - Fast Python linter
- **Pre-commit** - Automated checks on commit

```bash
# Format code manually
black --line-length=100 .
isort --profile=black --line-length=100 .

# Run linter
ruff check .

# Run all pre-commit hooks
pre-commit run --all-files
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=backend --cov=frontend --cov=utils --cov-report=html

# Run specific test file
pytest tests/unit/test_visualization_core.py -v

# Run tests matching pattern
pytest -k "test_chart" -v
```

### Pull Request Process

1. **Ensure tests pass**: `pytest tests/`
2. **Check code style**: `pre-commit run --all-files`
3. **Update documentation** if needed
4. **Write descriptive commit messages**:
   ```
   feat: add support for Arbin .res files

   - Implement ArbinParser class
   - Add unit tests for parser
   - Update documentation
   ```
5. **Open PR** against `main` branch
6. **Respond to review feedback**

### Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

| Prefix | Description |
|--------|-------------|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `docs:` | Documentation changes |
| `style:` | Code style (formatting, no logic change) |
| `refactor:` | Code refactoring |
| `test:` | Adding/updating tests |
| `chore:` | Maintenance tasks |

---

## 🗺️ Roadmap

### Version 1.0 (Current)
- [x] Multi-format data import
- [x] Interactive visualizations
- [x] Basic ML predictions
- [x] Export functionality

### Version 1.1 (Planned)
- [ ] Real-time data streaming
- [ ] Collaborative features
- [ ] Advanced EIS fitting
- [ ] Custom report templates

### Version 2.0 (Future)
- [ ] Cloud deployment option
- [ ] API for external integrations
- [ ] Mobile-responsive design
- [ ] Plugin system for custom analyzers

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 AmpereData Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software...
```

---

## 🙏 Acknowledgments

- **[NASA](https://www.nasa.gov/)** - Open battery testing datasets used for model training
- **[BioLogic](https://www.biologic.net/)** - EC-Lab file format documentation
- **[Streamlit](https://streamlit.io/)** - Amazing framework for data apps
- **[Plotly](https://plotly.com/)** - Interactive visualization library
- **Open Source Community** - All the amazing libraries we depend on

---

## 📬 Contact

- **GitHub Issues**: [Bug reports & feature requests](https://github.com/arunnath011/amperedata-streamlit/issues)
- **GitHub Discussions**: [Questions & community](https://github.com/arunnath011/amperedata-streamlit/discussions)

---

<div align="center">

**Made with ⚡ by the AmpereData Community**

If you find this project useful, please consider giving it a ⭐!

</div>
