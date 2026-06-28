# AstroExo: AI-Powered Exoplanet Detection Platform

AstroExo is a professional-grade scientific platform designed to detect and classify exoplanet transits from TESS (Transiting Exoplanet Survey Satellite) light curves. Developed for the Bharatiya Antariksh Hackathon, AstroExo seamlessly bridges the gap between raw astrophysical data and human-interpretable scientific discovery.

## 🎯 Problem Statement

Analyzing raw light curve data to detect exoplanets is a mathematically complex and time-consuming process. Traditional methods require astronomers to manually filter noise, remove stellar variability, and visually inspect phase-folded curves to distinguish true exoplanets from false positives like eclipsing binaries or instrumental artifacts. AstroExo solves this by providing a fully automated, AI-driven pipeline that ingests astronomical catalogs, processes the signals, and provides an explainable classification in seconds.

## 🚀 Key Features

* **Intelligent Ingestion:** Supports automated parsing of CSV catalogs and lightcurves. Automatically detects primary identifiers (TIC ID) and presents a clean preview.
* **Astrophysical Pipeline:** Automatically cleans data, removes stellar variability (Savitzky-Golay detrending), and isolates transit signatures.
* **Transit Detection & Phase Folding:** Implements a rigorous Box Least Squares (BLS) periodogram to identify periodic transit dips and automatically calculates the transit depth, duration, and orbital period, folding the lightcurve for analysis.
* **Machine Learning Classification:** Utilizes a trained Random Forest AI model (with XGBoost capabilities) to classify candidates into distinct categories (e.g., Exoplanet Transit, Eclipsing Binary).
* **Explainable AI (XAI):** A state-of-the-art XAI dashboard explicitly lists the mathematical evidence, feature importance, and risk indicators driving the AI's decision.
* **Interactive Scientific Visualizations:** Provides a 7-chart Plotly visualization suite covering the entire pipeline: Raw Data, Quality Filtering, Detrending, Transit Detection, BLS Power Spectrum, Phase Folding, and Binned Modeling.
* **Publication-Ready Reports:** Generates dark-themed, publication-quality PDF reports via the backend (ReportLab), complete with embedded charts, scientific parameters, and XAI summaries.
* **Robust Error Handling:** Intercepts invalid datasets, flat signals, and incompatible catalogs with user-friendly descriptive dialogs instead of generic backend failures.

## 🏗️ System Architecture

AstroExo features a decoupled architecture, separating high-performance numerical computation from interactive frontend visualization.

```text
Frontend (React + Vite)
      │
      ▼
FastAPI (REST Interface)
      │
      ▼
Data Ingestion (CSV / File Uploads)
      │
      ▼
Detrending & Quality Filter
      │
      ▼
Transit Detection (BLS) & Phase Folding
      │
      ▼
Feature Extraction (Morphology & SNR)
      │
      ▼
AI Classification (Random Forest / XGBoost)
      │
      ▼
XAI Dashboard & PDF Generation (ReportLab)
```

## 📂 Project Structure

```text
AstroExo/
├── start.bat / start.sh      # Automated startup scripts
├── requirements.txt          # Python dependencies
├── backend/                  # Python FastAPI Backend
│   ├── app.py                # Main API routing and application entry point
│   ├── classifier.py         # Random Forest / XGBoost training and classification
│   ├── config.py             # Global backend configuration
│   ├── data_loader.py        # CSV parsing and TESS MAST data retrieval
│   ├── detrend.py            # Savitzky-Golay filtering and variability removal
│   ├── feature_engineering.py# Scientific feature extraction for ML
│   ├── parameter_fit.py      # Estimation of physical/orbital parameters
│   ├── phase_fold.py         # Light curve phase folding and binning
│   ├── quality_filter.py     # Outlier rejection and signal cleaning
│   ├── transit_search.py     # Box Least Squares transit detection algorithm
│   └── utils.py              # Backend utility functions
├── frontend/                 # React + Vite Frontend
│   ├── index.html            # Main HTML entry point
│   ├── package.json          # Node dependencies
│   ├── vite.config.ts        # Vite bundler configuration
│   └── src/
│       ├── App.tsx           # Main application dashboard and UI orchestration
│       ├── api.ts            # Axios API client for backend communication
│       ├── index.css         # Global styling and AstroExo design system
│       └── main.tsx          # React DOM mounting
└── models/                   # Serialized ML Models
    └── classifier.pkl        # Pre-trained Random Forest model
```

## 🛠️ Technology Stack

* **Frontend:** React 19, TypeScript, Vite, Plotly.js, Lucide-React
* **Backend:** Python 3.10+, FastAPI, Uvicorn, Pandas, Numpy, Scipy, Scikit-Learn, XGBoost
* **Reporting:** ReportLab (Backend PDF Generation)

## 💻 Installation & Quick Start

### Prerequisites
* Python 3.10 or higher
* Node.js 18 or higher

### One-Click Startup (Recommended)
You can launch the entire stack (both frontend and backend) using the provided startup scripts:
* **Windows:** Double-click `start.bat` or run it from the command prompt.
* **macOS/Linux:** Run `./start.sh` from the terminal.

The script will automatically install dependencies and start both servers.

### Manual Setup

**Backend Setup:**
```bash
python -m venv venv
# Activate the virtual environment:
# On Windows: venv\Scripts\activate
# On macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

**Frontend Setup:**
```bash
cd frontend
npm install
npm run dev
```

## 📊 Usage Guide

1. **Launch the Platform:** Open the AstroExo Dashboard at `http://localhost:5173`.
2. **Select Data Source:** 
   * **Upload CSV:** Click the upload area to provide an astronomical light curve or catalog. The system will process it or present a preview catalog to select from.
   * **Sample Targets:** Enter a target ID (e.g., a TIC ID) and click "Analyze", or use the pre-defined target workflows.
3. **Monitor Processing:** The dashboard cleanly resets and displays a floating status indicator tracking the pipeline stages (Searching MAST, Detrending, Classifying, etc.).
4. **Review Results:** Once complete, interact with the scientific charts to investigate the light curve morphology and BLS periodogram.
5. **Explainable AI:** Scroll down to review the AI Decision Summary, Supporting Scientific Evidence, Risk Indicators, and Feature Importance.
6. **Export Report:** Click the **"Export PDF Report"** button in the header to generate and download a comprehensive, publication-ready summary from the backend.

## 🔭 Future Enhancements

* **GPU Acceleration:** Currently, the BLS periodogram operates on the CPU. Future iterations will migrate the search grid to CuPy/JAX for processing long-cadence multi-sector data.
* **Deep Learning Integration:** While Random Forests provide excellent XAI capabilities, a parallel CNN trained directly on phase-folded fluxes is planned to augment confidence scoring.
* **Database Persistence:** Currently, analysis results are transient in memory. Future builds will incorporate PostgreSQL for cataloging confirmed candidates.

## 📄 License

This project was developed for the Bharatiya Antariksh Hackathon 2026. All rights reserved.
