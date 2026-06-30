# AstroExo AI Detection Dashboard

## 1. Project Overview
Analyzing raw light curve data to detect exoplanets is a mathematically complex and time-consuming process. Traditional methods require astronomers to manually filter noise, remove stellar variability, and visually inspect phase-folded curves to distinguish true exoplanets from false positives like eclipsing binaries or instrumental artifacts. AstroExo solves this problem by providing a fully automated, AI-driven pipeline that ingests TESS (Transiting Exoplanet Survey Satellite) light-curve observations, processes the signals, and provides an explainable exoplanet classification in seconds.

## 2. Features
* **TIC ID Analysis:** Directly analyze targets using their TESS Input Catalog (TIC) identifiers.
* **CSV Catalog Upload:** Support for uploading astronomical catalogs and light curves in CSV format.
* **Automatic Catalog Validation:** Intelligent parsing and validation of uploaded data formats.
* **Light Curve Preprocessing:** Automated preparation and cleaning of raw light curves.
* **Quality Filtering:** Advanced filtering utilizing TESS quality flags to reject outliers and anomalies.
* **Detrending:** Savitzky-Golay filtering to remove stellar variability and instrumental noise.
* **Transit Detection (BLS):** Box Least Squares (BLS) algorithm to identify periodic transit signatures.
* **Feature Extraction:** Scientific extraction of morphological and signal-to-noise metrics.
* **AI-assisted Classification:** Random Forest machine learning model to classify candidates.
* **Scientific Parameter Estimation:** Calculation of transit depth, duration, and orbital period.
* **Interactive Visualizations:** Suite of dynamic Plotly charts for deep data exploration.
* **PDF Report Generation:** Automated creation of publication-ready scientific reports.
* **Robust Error Handling:** Graceful management of invalid inputs, missing data, and processing failures.
* **Sample Targets:** Built-in demonstration targets for immediate platform evaluation.

## 3. Technology Stack
* **Frontend:** React, TypeScript, Vite, Plotly
* **Backend:** Python, FastAPI, Uvicorn
* **Machine Learning:** Scikit-Learn (Random Forest), XGBoost
* **Scientific Libraries:** Pandas, NumPy, SciPy
* **Data Sources:** TESS (Transiting Exoplanet Survey Satellite) observations, MAST (Mikulski Archive for Space Telescopes)

## 4. Project Architecture
The AstroExo platform utilizes a decoupled architecture where a React-based frontend communicates with a Python FastAPI backend. 
1. **Frontend:** Provides an interactive UI for users to upload data, input TIC IDs, and visualize results using Plotly charts. It orchestrates the analysis workflow.
2. **Backend:** Acts as the computational engine. It exposes RESTful endpoints to trigger data retrieval, run the scientific pipeline, and generate PDF reports.
3. **Machine Learning Pipeline:** Integrated into the backend, it processes extracted light curve features through a pre-trained Random Forest model to determine the likelihood of an exoplanet transit.
4. **Data Flow:** User Request -> FastAPI -> Data Loader (CSV/MAST) -> Quality Filter -> Detrending -> Transit Search -> Phase Folding -> Feature Engineering -> AI Classifier -> JSON Response -> Frontend Visualization.

## 5. Folder Structure
```text
AstroExo/
├── backend/                  # Python FastAPI Backend and ML Pipeline
│   ├── app.py                # Main API routing
│   ├── classifier.py         # AI classification logic
│   ├── data_loader.py        # Data ingestion from CSV/MAST
│   ├── detrend.py            # Light curve detrending
│   ├── feature_engineering.py# Scientific feature extraction
│   ├── parameter_fit.py      # Scientific parameter estimation
│   ├── phase_fold.py         # Phase folding logic
│   ├── quality_filter.py     # Outlier and quality filtering
│   ├── transit_search.py     # BLS transit detection
│   └── utils.py              # Utility functions
├── data/                     # Sample datasets and catalogs
├── docs/                     # Project documentation
├── frontend/                 # React + Vite Frontend
│   ├── public/               # Static assets
│   ├── src/                  # React components and logic
│   ├── package.json          # Node dependencies
│   └── vite.config.ts        # Vite configuration
├── models/                   # Serialized Machine Learning models
├── .gitignore                # Git ignore rules
├── DEMO_SCRIPT.md            # Demo script
├── README.md                 # Project documentation
├── TESTING_CHECKLIST.md      # Testing checklists
├── requirements.txt          # Python dependencies
├── start.bat                 # Windows startup script
└── start.sh                  # Linux/macOS startup script
```

## 6. Installation
The application is designed to be run locally on your machine.

### Prerequisites
* Python 3.10+
* Node.js 18+

### Automated Startup
To automatically install dependencies and start both frontend and backend servers:
* **Windows:** Double-click `start.bat` or run it from the command prompt.
* **Linux/macOS:** Run `./start.sh` from the terminal.

### Manual Startup
If you prefer to start the services manually:

**1. Backend**
```bash
python -m venv venv
# Activate the virtual environment:
# Windows: venv\Scripts\activate
# Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

**2. Frontend**
```bash
cd frontend
npm install
npm run dev
```
The dashboard will be available at `http://localhost:5173`.

## 7. Usage
* **Analyze a TIC ID:** Enter a valid TESS Input Catalog (TIC) ID in the input field and click "Analyze" to fetch and process data directly from MAST.
* **Upload a CSV Catalog:** Drag and drop or select a CSV catalog containing light curve data or target lists to initiate automated analysis.
* **Use Sample Targets:** Utilize the pre-configured sample targets available on the dashboard for quick demonstrations of the pipeline's capabilities.
* **Export PDF Reports:** After a successful analysis, click the export button to generate and download a comprehensive, publication-ready PDF report detailing the findings.

## 8. Machine Learning Pipeline
The analytical core of AstroExo follows a strict sequence:
1. **Data Loading:** Ingests raw observation data from CSV uploads or MAST queries.
2. **Quality Filtering:** Applies TESS bitmask filtering and outlier rejection to clean the signal.
3. **Detrending:** Uses Savitzky-Golay filtering to remove low-frequency stellar variability.
4. **Transit Search (BLS):** Executes a Box Least Squares periodogram to find periodic dips.
5. **Phase Folding:** Folds the time-series data based on the discovered period to align transits.
6. **Feature Engineering:** Extracts critical morphological and physical features (e.g., depth, SNR).
7. **Random Forest Classification:** Analyzes the extracted features using a trained Random Forest model.
8. **Scientific Parameter Estimation:** Derives final physical parameters for the potential exoplanet.

## 9. Output
Upon completion, the dashboard presents:
* **AI Prediction:** The final classification (e.g., Exoplanet Candidate, Eclipsing Binary).
* **Confidence Score:** The probability assigned to the prediction by the AI model.
* **Scientific Parameters:** Calculated values like Transit Period, Depth, and Duration.
* **Observation Metadata:** Details about the target star and TESS sector observations.
* **Interactive Charts:** A suite of dynamic Plotly graphs illustrating raw data, detrending, the BLS periodogram, and the phase-folded model.
* **PDF Report:** A downloadable, formatted document summarizing all evidence and metrics.

## 10. Error Handling
AstroExo incorporates robust mechanisms to handle failures gracefully:
* **Invalid TIC IDs:** Alerts the user if the entered ID does not exist or has no recorded observations.
* **Missing TESS Observations:** Informs the user if light curve data cannot be retrieved for a target.
* **Invalid Uploads:** Validates CSV structure and notifies the user of formatting or data errors.
* **Report Generation Restrictions:** Prevents report generation if the analysis fails or data is incomplete, providing clear feedback on the required steps.

## 11. Future Improvements
* Implementation of GPU acceleration for the BLS transit search to analyze long-cadence, multi-sector targets significantly faster.
* Integration of a deep learning model (e.g., Convolutional Neural Network) operating directly on the folded flux arrays to complement the Random Forest classifier.
* Expansion of the pipeline to support light curve data from other missions, such as Kepler and K2.

## 12. License
This project was developed for the Bharatiya Antariksh Hackathon 2026. All rights reserved.

## 13. Author
AstroExo Team
