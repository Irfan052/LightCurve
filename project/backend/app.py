import os
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional

from backend.config import MODEL_PATH
from backend.utils import logger, safe_serialize
from backend.data_loader import load_tess_lightcurve, load_uploaded_file
from backend.quality_filter import clean_lightcurve
from backend.detrend import flatten_lightcurve
from backend.transit_search import search_transits
from backend.phase_fold import fold_lightcurve, bin_folded_lightcurve
from backend.feature_engineering import extract_features
from backend.classifier import predict_class, train_classifier, load_or_train_model
from backend.parameter_fit import estimate_parameters
from backend.visualize import (
    plot_raw_lightcurve, plot_cleaned_detrended, 
    plot_folded_transit, plot_bls_periodogram
)

app = FastAPI(
    title="AI-Enabled Exoplanet Detection Pipeline",
    description="A scientific pipeline using Box Least Squares (BLS) and Random Forest classifier to detect and classify planet transits in TESS light curves.",
    version="1.0.0"
)

# Enable CORS for frontend web requests (essential for local testing/file:// urls)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup event to ensure ML model is trained and ready
@app.on_event("startup")
def startup_event():
    logger.info("Starting up Backend Server...")
    # Load model (triggers training if pkl is missing)
    load_or_train_model()
    logger.info("Backend Server is ready and model is loaded.")

class AnalyzeRequest(BaseModel):
    target_id: str

@app.get("/api/health")
def health_check():
    model_ready = MODEL_PATH.exists()
    return {
        "status": "healthy",
        "model_loaded": model_ready,
        "model_path": str(MODEL_PATH)
    }

@app.post("/api/train")
def trigger_retrain(background_tasks: BackgroundTasks):
    """Triggers manual background retraining of the classifier."""
    background_tasks.add_task(train_classifier)
    return {"message": "Retraining task scheduled in background."}

def run_analysis_pipeline(
    time: np.ndarray, 
    flux: np.ndarray, 
    flux_err: np.ndarray, 
    target_name: str,
    is_mock: bool = False
) -> Dict[str, Any]:
    """Runs the full exoplanet detection science and ML pipeline."""
    try:
        # 1. Clean the raw light curve (NaNs, extreme outliers)
        time_clean, flux_clean, flux_err_clean = clean_lightcurve(time, flux, flux_err)
        
        if len(time_clean) < 10:
            raise HTTPException(status_code=400, detail="Insufficient clean data points in light curve.")
            
        # 2. Detrend / Flatten the light curve (remove low-frequency trends)
        flat_flux, trend_flux = flatten_lightcurve(time_clean, flux_clean)
        
        # 3. Search for periodic dips using Box Least Squares (BLS)
        bls_results = search_transits(time_clean, flat_flux, flux_err_clean)
        
        best_period = bls_results["period"]
        best_epoch = bls_results["epoch"]
        
        # 4. Fold the light curve at the best candidate period
        folded = fold_lightcurve(time_clean, flat_flux, flux_err_clean, best_period, best_epoch)
        
        # Bin the folded data (100 bins) for visualization and feature extraction
        bin_centers, bin_flux, bin_err = bin_folded_lightcurve(
            folded["phase"], folded["flux"], num_bins=100
        )
        
        # 5. Extract scientific features
        features = extract_features(time_clean, flat_flux, flux_err_clean, bls_results)
        
        # 6. Predict transit classification with Random Forest
        classification = predict_class(features)
        
        # 7. Estimate physical and orbital parameters
        parameters = estimate_parameters(bls_results, classification)
        
        # 8. Generate visualization plots (returned as base64 images)
        logger.info("Generating visualization plots...")
        raw_plot = plot_raw_lightcurve(time, flux, target_name)
        detrend_plot = plot_cleaned_detrended(time_clean, flux_clean, trend_flux, flat_flux, target_name)
        folded_plot = plot_folded_transit(folded["phase"], folded["flux"], bin_centers, bin_flux, target_name, best_period)
        
        # Check if periodogram variables exist and plot
        if len(bls_results["periods"]) > 0:
            bls_plot = plot_bls_periodogram(
                bls_results["periods"], 
                bls_results["powers"], 
                best_period, 
                target_name
            )
        else:
            bls_plot = ""
            
        logger.info("Pipeline executed successfully. Building response.")
        
        # Build JSON serializable output payload
        return safe_serialize({
            "target_name": target_name,
            "is_mock": is_mock,
            "prediction": classification["prediction_label"],
            "confidence": classification["confidence"],
            "probabilities": classification["probabilities"],
            "parameters": parameters,
            "features": features,
            "plots": {
                "raw": raw_plot,
                "detrended": detrend_plot,
                "folded": folded_plot,
                "bls": bls_plot
            }
        })
        
    except Exception as e:
        logger.exception("Error during pipeline execution")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

import numpy as np

@app.post("/api/analyze")
def analyze_target(request: AnalyzeRequest):
    """Downloads a TESS target ID or generates mock data and runs the analysis."""
    target_id = request.target_id.strip()
    if not target_id:
        raise HTTPException(status_code=400, detail="Target ID cannot be empty.")
        
    # Load light curve data
    data = load_tess_lightcurve(target_id)
    
    return run_analysis_pipeline(
        time=data["time"],
        flux=data["flux"],
        flux_err=data["flux_err"],
        target_name=data["target_name"],
        is_mock=data["is_mock"]
    )

@app.post("/api/upload")
async def upload_and_analyze(file: UploadFile = File(...)):
    """Accepts an uploaded CSV light curve file and runs the analysis."""
    # Ensure temporary upload directory exists
    temp_dir = Path("data/raw/uploads")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    temp_file_path = temp_dir / file.filename
    try:
        # Save file locally
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Parse CSV
        data = load_uploaded_file(str(temp_file_path))
        
        # Execute pipeline
        result = run_analysis_pipeline(
            time=data["time"],
            flux=data["flux"],
            flux_err=data["flux_err"],
            target_name=data["target_name"],
            is_mock=data["is_mock"]
        )
        
        return result
        
    except Exception as e:
        logger.exception("Error loading uploaded file")
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        # Clean up temporary upload file
        if temp_file_path.exists():
            os.remove(temp_file_path)
