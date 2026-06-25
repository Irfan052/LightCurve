# AstroExo AI Pipeline

![AstroExo Banner](https://via.placeholder.com/1200x400/0f172a/38bdf8?text=AstroExo+AI+Pipeline)

AstroExo is an advanced, end-to-end scientific pipeline designed to detect and classify exoplanet transits from raw TESS (Transiting Exoplanet Survey Satellite) light curves. Built for speed, explainability, and accuracy, AstroExo transforms noisy photometric data into actionable scientific reports.

## Problem Statement
The TESS mission generates millions of light curves, but only a fraction contain genuine exoplanet transits. Traditional transit search algorithms often flag false positives such as eclipsing binaries, stellar variability, and instrumental artifacts. Human vetting of these candidates is incredibly time-consuming and creates a bottleneck in exoplanet discovery.

## Solution
AstroExo solves this by integrating a robust physics-based preprocessing pipeline with a Machine Learning classification layer. By engineering domain-specific features from the Box Least Squares (BLS) periodogram and the phase-folded light curve, AstroExo accurately differentiates between true planets and astrophysical false positives, completely automating the vetting process.

## Architecture

1. **Data Loading:** Automatically fetches light curves from the MAST archive using Lightkurve, or accepts direct CSV uploads.
2. **Quality Filtering & Detrending:** Removes low-quality flags, masks out NaNs, and flattens the light curve to remove low-frequency stellar activity.
3. **Transit Search (BLS):** Applies the Box Least Squares algorithm to identify the dominant periodic transit signal.
4. **Feature Engineering:** Extracts over a dozen physical metrics including transit depth, duration, odd/even depth differences, and phase-folded shape scores.
5. **AI Classification (Random Forest):** A trained machine learning model categorizes the signal into: `exoplanet_transit`, `eclipsing_binary`, `stellar_variability`, or `instrumental_artifact`.
6. **Parameter Estimation:** Derives the estimated planetary radius (in Earth radii) and orbital period.
7. **Scientific Reporting:** Generates a professional, interactive dashboard and exports a comprehensive PDF report capturing the charts and Explainable AI (XAI) feature importances.

## Quick Start (One-Command Startup)
AstroExo is designed for immediate deployment. You do not need to configure any environment variables or download pretrained weights (the model auto-generates on first boot).

**Windows:**
Double-click `start.bat` or run:
```cmd
start.bat
```

**Linux/Mac:**
```bash
chmod +x start.sh
./start.sh
```

These scripts will automatically install Python and Node.js dependencies, start the FastAPI backend on port `8000`, and launch the Vite React dashboard on port `5173`.

## ML Model
The core classifier is a Random Forest model. 
- **Explainable AI (XAI):** The model outputs Gini importance scores, allowing researchers to see exactly *why* a candidate was flagged (e.g., "High odd-even depth difference typical of background EBs").
- **Performance:** Validated against known targets yielding >90% confidence on verified exoplanets like TOI-700 and WASP-126.

## Results
The pipeline successfully identifies transits even in low Signal-to-Noise (SNR) environments, gracefully handling data gaps and instrumental noise.
See `docs/validation_results.md` for our performance matrix on real TESS targets.

## Screenshots
*(Insert hackathon screenshots here)*

## Future Work
- Integration with Kepler (K2) datasets.
- Deployment of deep learning models (1D CNNs) directly on the raw light curve.
- Mass batch-processing support for analyzing thousands of targets asynchronously.
