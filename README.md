# AmpereData - Battery Data Analytics

Open-source battery testing data analytics platform for researchers and engineers.

![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## Features

- **Universal Data Parsing** - BioLogic (.mpt), Neware (.nda/.ndax), CSV, Excel
- **Intelligent Column Mapping** - Auto-detect and map columns to standard schema
- **Interactive Visualizations** - 15+ chart types (Plotly-based)
- **Data Explorer** - SQL queries, filters, exports
- **ML Predictions** - RUL forecasting with XGBoost (98% R² accuracy)
- **Batch Processing** - Process multiple files at once

## Quick Start

```bash
# Clone repository
git clone https://github.com/arunnath011/amperedata-streamlit.git
cd amperedata-streamlit

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py

# Open browser: http://localhost:8501
```

## Default Login

- **Username**: `admin`
- **Password**: `admin123`

⚠️ Change the default password after first login!

## Supported File Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| BioLogic | .mpt, .mps | Full support |
| Neware | .nda, .ndax | Requires `NewareNDA` library |
| Neware CSV | .csv | Full support |
| Generic CSV | .csv, .txt | With column mapping |
| Excel | .xlsx, .xls | Multi-sheet support |

## Project Structure

```
amperedata-streamlit/
├── streamlit_app.py      # Main entry point
├── pages/                # Streamlit pages
│   ├── 1_Upload_Data.py
│   ├── 2_Visualizations.py
│   ├── 3_Data_Explorer.py
│   ├── 4_Settings.py
│   ├── 5_Advanced_Charts.py
│   ├── 6_Build_Sheet.py
│   └── 7_Live_Dashboard.py
├── backend/
│   ├── parsers/          # File parsers
│   ├── etl/              # ETL pipeline
│   └── ml/               # ML models
├── utils/                # Utilities
│   ├── auth.py          # Authentication
│   └── column_mapper.py # Column mapping
└── data/                 # Sample data
```

## Docker Deployment

```bash
docker-compose up -d
```

Services:
- Streamlit UI: http://localhost:8501
- PostgreSQL: localhost:5432
- Redis: localhost:6379

## Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- NASA for open battery testing datasets
- BioLogic for data format documentation
- Open source community

