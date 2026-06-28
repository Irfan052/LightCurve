# AstroExo: AI-Powered Exoplanet Detection Platform

AstroExo is a professional-grade scientific platform designed to detect and classify exoplanet transits from TESS (Transiting Exoplanet Survey Satellite) light curves. Developed for the Bharatiya Antariksh Hackathon, AstroExo seamlessly bridges the gap between raw astrophysical data and human-interpretable scientific discovery.

## 🎯 Problem Statement

Analyzing raw light curve data to detect exoplanets is a mathematically complex and time-consuming process. Traditional methods require astronomers to manually filter noise, remove stellar variability, and visually inspect phase-folded curves to distinguish true exoplanets from false positives like eclipsing binaries or instrumental artifacts. AstroExo solves this by providing a fully automated, AI-driven pipeline that ingests astronomical catalogs, processes the signals, and provides an explainable classification in seconds.

## 🚀 Key Features

* **Intelligent Ingestion:** Supports automated parsing of CSV catalogs. Automatically detects primary identifiers (TIC ID, Gaia ID) and presents a clean preview table.
* **Astrophysical Pipeline:** Automatically cleans data, removes stellar variability (Savitzky-Golay detrending), and isolates transit signatures.
* **Transit Detection:** Implements a rigorous Box Least Squares (BLS) periodogram to identify periodic transit dips and automatically calculates the transit depth, duration, and orbital period.
* **Machine Learning Classification:** Utilizes a trained Random Forest AI model to classify candidates into distinct categories (e.g., Exoplanet Transit, Eclipsing Binary).
* **Explainable AI (XAI):** A state-of-the-art XAI dashboard explicitly lists the mathematical evidence, feature importance, and risk indicators driving the AI's decision.
* **Interactive Scientific Visualizations:** Provides a 7-chart Plotly visualization suite covering the entire pipeline: Raw Data, Quality Filtering, Detrending, Transit Detection, BLS Power Spectrum, Phase Folding, and Binned Modeling.
* **Publication-Ready Reports:** Generates dark-themed, publication-quality PDF reports complete with embedded charts, scientific parameters, and XAI summaries.
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
AI Classification (Random Forest)
      │
      ▼
XAI Dashboard & PDF Generation
```

## 📂 Project Structure

```text
AstroExo/
├── backend/                  # Python FastAPI Backend
│   ├── app.py                # Main API routing and application entry point
│   ├── ai_models.py          # Random Forest loading and feature extraction
│   ├── bls_search.py         # Box Least Squares transit detection algorithm
│   ├── data_ingestion.py     # CSV parsing and target identification logic
│   ├── detrending.py         # Savitzky-Golay filtering and variability removal
│   ├── quality_filter.py     # Outlier rejection and signal cleaning
│   └── requirements.txt      # Python dependencies
└── frontend/                 # React + Vite Frontend
    ├── index.html            # Main HTML entry point
    ├── package.json          # Node dependencies
    ├── src/
    │   ├── App.tsx           # Main application dashboard and UI orchestration
    │   ├── api.ts            # Axios API client for backend communication
    │   ├── index.css         # Global styling and AstroExo design system
    │   └── main.tsx          # React DOM mounting
```

## 🛠️ Technology Stack

* **Frontend:** React 19, TypeScript, Vite, Plotly.js, HTML2Canvas, Lucide-React
* **Backend:** Python 3.10+, FastAPI, Uvicorn, Pandas, Numpy, Scipy, Scikit-Learn
* **Reporting:** jsPDF (Frontend PDF Compilation)

## 💻 Installation Instructions

### Prerequisites
* Python 3.10 or higher
* Node.js 18 or higher

### Backend Setup
```bash
cd backend
python -m venv venv
# Activate the virtual environment:
# On Windows: venv\Scripts\activate
# On macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
python -m uvicorn app:app --reload
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

## 📊 Usage Guide

1. **Launch the Platform:** Open the AstroExo Dashboard at `http://localhost:5173`.
2. **Select Data Source:** 
   * **Upload CSV:** Click the upload area to provide an astronomical catalog (must contain target identifiers). The system will preview the catalog. Click "Analyze" on any row to begin processing.
   * **Sample Targets:** Click "Analyze Synthetic Demo Data" in the left sidebar to execute a predefined, successful exoplanet detection workflow.
3. **Monitor Processing:** The dashboard cleanly resets and displays a floating status indicator tracking the pipeline stages (Searching MAST, Downloading light curve, Detrending, Classifying, etc.).
4. **Review Results:** Once complete, interact with the 7 scientific charts to investigate the light curve morphology and BLS periodogram.
5. **Explainable AI:** Scroll down to review the AI Decision Summary, Supporting Scientific Evidence, Risk Indicators, and Random Forest Feature Importance.
6. **Export Report:** Click the **"Export PDF Report"** button in the header to generate and download a comprehensive, publication-ready summary of the findings.

## 🔭 Future Enhancements

* **GPU Acceleration:** Currently, the BLS periodogram operates on the CPU. Future iterations will migrate the search grid to CuPy/JAX for processing long-cadence multi-sector data.
* **Deep Learning Integration:** While Random Forests provide excellent XAI capabilities, a parallel CNN trained directly on phase-folded fluxes is planned to augment confidence scoring.
* **Database Persistence:** Currently, analysis results are transient in memory. Future builds will incorporate PostgreSQL for cataloging confirmed candidates.

## 📄 License

This project was developed for the Bharatiya Antariksh Hackathon 2026. All rights reserved.
