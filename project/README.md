# AstroExo AI: Exoplanet Detection from Noisy Light Curves

A complete, hackathon-ready data processing and machine learning pipeline to detect exoplanetary transit events from noisy TESS light curves. The system filters out stellar variability and instrumental artifacts using robust astrophysical algorithms (Sigma-Clipping, Savitzky-Golay filtering, Box Least Squares periodograms), extracts diagnostic transit parameters, and classifies events using a Random Forest classifier.

---

## 🌌 System Architecture

```
[Raw Observations] (TESS Sector / Uploaded CSV)
       │
       ▼
[Quality Filter] (Remove NaNs, Asymmetric Sigma-Clipping)
       │
       ▼
[Detrend & Flatten] (Savitzky-Golay low-frequency filtering)
       │
       ▼
[Transit Search] (Box Least Squares Periodogram Peak) ──► [BLS Power Spectrum Plot]
       │
       ▼
[Phase Fold & Bin] (Fold time-series at peak period)  ──► [Folded Light Curve Plot]
       │
       ├──────────────────────────────┐
       ▼                              ▼
[Feature Engineering]         [Parameter Estimation]
  - Shape Score (U vs V)        - Orbital Period (Days)
  - Odd-Even Depth Diff         - Transit Depth & Duration
  - Secondary Eclipse Ratio     - Est. Planet Radius (R_Earth)
  - Out-of-transit Scatter      - Semi-major axis (AU)
       │                              │
       ▼                              │
[ML Classifier]                       │
  - Random Forest                     │
       │                              │
       ▼                              ▼
[Classification Label & Confidence] ◄─┘
       │
       ▼
[FastAPI JSON Response] ──► [Interactive Web Dashboard]
```

---

## 📂 Project Directory Structure

```
project/
├── backend/
│   ├── app.py                 # FastAPI application & API endpoints
│   ├── config.py              # Configuration constants & directories
│   ├── data_loader.py         # Lightkurve MAST downloader & synthetic generator
│   ├── quality_filter.py      # NaN cleaning & asymmetric sigma-clipping
│   ├── detrend.py             # Savitzky-Golay flattening filter
│   ├── transit_search.py      # Astropy Box Least Squares search
│   ├── phase_fold.py          # Phase folding & binning routines
│   ├── feature_engineering.py # Diagnostic features (U/V ratio, odd-even diff)
│   ├── classifier.py          # Random Forest training & inference pipeline
│   ├── parameter_fit.py       # Physical & confidence parameters estimator
│   ├── visualize.py           # Matplotlib PNG & Base64 encoders
│   ├── utils.py               # Standard logger & converters
│   └── test_pipeline.py       # Comprehensive unittest suite
├── frontend/
│   ├── index.html             # Premium space-themed glassmorphism interface
│   ├── style.css              # Custom styling, glow effects, responsive grid
│   └── script.js              # State orchestration & API fetch hooks
├── models/
│   └── classifier.pkl         # Cached trained Random Forest model (auto-generated)
├── data/
│   ├── raw/                   # Raw light curve directories
│   ├── processed/             # Cleaned light curve data
│   └── results/               # Matplotlib output charts (PNG)
├── requirements.txt           # PIP dependencies manifest
└── README.md                  # Setup & pipeline documentation
```

---

## ⚡ Deployment & Run Instructions

### Prerequisites
- Python 3.9+ (Python 3.13 supported)
- internet connection (to download TESS observations; mock data fallback is automatic)

### 1. Installation
Clone or copy the project files to your directory and install the requirements:
```bash
pip install -r requirements.txt
```

### 2. Running Unit Tests
Validate the scientific pipeline and trigger initial model training:
```bash
python -m unittest backend.test_pipeline
```

### 3. Running the Backend Server
Start the FastAPI server on port `8000`:
```bash
uvicorn backend.app:app --reload --port 8000
```
- Open your browser to `http://127.0.0.1:8000/docs` to view the interactive Swagger API documentation.

### 4. Running the Frontend Dashboard
Simply open the `frontend/index.html` file in any modern browser, or serve it using a lightweight HTTP server:
```bash
# Using Python to host the frontend locally:
python -m http.server 3000 --directory frontend
```
Navigate to `http://127.0.0.1:3000` to view the UI.

---

## 🛰️ API Integration Example

### Request (`POST /api/analyze`)
```json
{
  "target_id": "TIC 261108234"
}
```

### Response
```json
{
  "target_name": "TIC 261108234",
  "is_mock": false,
  "prediction": "exoplanet_transit",
  "confidence": 0.942,
  "probabilities": {
    "exoplanet_transit": 0.942,
    "eclipsing_binary": 0.048,
    "stellar_variability": 0.007,
    "instrumental_artifact": 0.003
  },
  "parameters": {
    "period": 3.5212,
    "epoch": 120.4502,
    "transit_depth": 0.00985,
    "transit_depth_percent": 0.985,
    "transit_duration_hours": 3.42,
    "planet_radius_earth": 10.82,
    "semi_major_axis_au": 0.0452,
    "snr": 24.52,
    "confidence_score": 0.957
  },
  "plots": {
    "raw": "iVBORw0KGgoAAAANS...",
    "detrended": "iVBORw0KGgoAAAANS...",
    "folded": "iVBORw0KGgoAAAANS...",
    "bls": "iVBORw0KGgoAAAANS..."
  }
}
```
