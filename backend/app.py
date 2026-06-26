import os
import shutil
from pathlib import Path
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import io
import datetime
from datetime import timezone
import time
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from backend.config import MODEL_PATH
from backend.utils import logger, safe_serialize
from backend.data_loader import load_tess_lightcurve, load_uploaded_file
from backend.quality_filter import clean_lightcurve
from backend.detrend import flatten_lightcurve
from backend.transit_search import search_transits
from backend.phase_fold import fold_lightcurve, bin_folded_lightcurve
from backend.feature_engineering import extract_features
from backend.classifier import predict_class, train_classifier, load_or_train_model, REPORT_PATH
from backend.parameter_fit import estimate_parameters
import json

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
    start_time = time.time()
    
    retrained = False
    if not MODEL_PATH.exists():
        logger.info("Model not found. Initiating automatic training...")
        retrained = True
        
    # Load model (triggers training if pkl is missing)
    load_or_train_model()
    
    duration = time.time() - start_time
    
    report = {
        "startup_time_seconds": round(duration, 3),
        "model_retrained_on_startup": retrained,
        "timestamp": datetime.datetime.now(timezone.utc).isoformat()
    }
    
    with open("startup_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    logger.info("Model Ready")
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
        
        # 8. Load feature importance for Explainable AI
        feature_importance_summary = {}
        if REPORT_PATH.exists():
            try:
                with open(REPORT_PATH, 'r') as f:
                    report = json.load(f)
                    feature_importance_summary = report.get("feature_importance_summary", {})
            except Exception as exc:
                logger.warning(f"Could not load feature importance from report: {exc}")
        
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
            "feature_importance_summary": feature_importance_summary,
            "data": {
                "time": time.tolist(),
                "flux": flux.tolist(),
                "time_clean": time_clean.tolist(),
                "flux_clean": flux_clean.tolist(),
                "flat_flux": flat_flux.tolist(),
                "trend_flux": trend_flux.tolist(),
                "folded_phase": folded["phase"].tolist(),
                "folded_flux": folded["flux"].tolist(),
                "bin_centers": bin_centers.tolist(),
                "bin_flux": bin_flux.tolist(),
                "bls_periods": bls_results.get("periods", np.array([])).tolist(),
                "bls_powers": bls_results.get("powers", np.array([])).tolist(),
            }
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error during pipeline execution")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

@app.post("/api/analyze")
def analyze_target(request: AnalyzeRequest):
    """Downloads a TESS target ID or generates mock data and runs the analysis."""
    target_id = request.target_id.strip()
    if not target_id:
        raise HTTPException(status_code=400, detail="Target ID cannot be empty.")
        
    # Load light curve data
    try:
        data = load_tess_lightcurve(target_id)
    except ValueError as e:
        logger.error(f"Validation error for target {target_id}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except (ConnectionError, RuntimeError) as e:
        logger.error(f"External service error for target {target_id}: {str(e)}")
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error for {target_id}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while fetching the data.")
        
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
    import uuid
    temp_dir = Path("data/raw/uploads")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Verify upload size limits (Max 20MB)
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    if file_size > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Maximum upload size is 20MB.")
    
    # Use UUID to prevent race conditions during concurrent uploads
    safe_filename = f"{uuid.uuid4().hex}_{Path(file.filename).name}"
    temp_file_path = temp_dir / safe_filename
    try:
        # Save file locally
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Parse CSV
        data = load_uploaded_file(str(temp_file_path))
        
        # If it's a catalog, return immediately
        if data.get("type") == "catalog":
            return data
            
        # Execute pipeline
        result = run_analysis_pipeline(
            time=data["time"],
            flux=data["flux"],
            flux_err=data["flux_err"],
            target_name=data["target_name"],
            is_mock=data.get("is_mock", False)
        )
        
        return result
        
    except Exception as e:
        logger.exception("Error loading uploaded file")
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        # Clean up temporary upload file
        if temp_file_path.exists():
            os.remove(temp_file_path)

class ReportRequest(BaseModel):
    target_id: str
    prediction: str
    confidence: float
    period: float
    depth: float
    duration: float
    planet_radius: float
    snr: float
    probabilities: Dict[str, float]
    features: List[str]

@app.post("/api/report")
def generate_pdf_report(req: ReportRequest):
    buffer = io.BytesIO()
    
    # Setup document with dark theme
    doc = SimpleDocTemplate(
        buffer, pagesize=letter,
        rightMargin=40, leftMargin=40,
        topMargin=40, bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        textColor=colors.HexColor('#ffffff'),
        backColor=colors.HexColor('#0f172a'),
        alignment=1,
        spaceAfter=20,
        fontSize=24
    )
    
    heading_style = ParagraphStyle(
        'HeadingStyle',
        parent=styles['Heading2'],
        textColor=colors.HexColor('#38bdf8'),
        spaceAfter=10,
        spaceBefore=20
    )
    
    text_style = ParagraphStyle(
        'TextStyle',
        parent=styles['Normal'],
        textColor=colors.HexColor('#e2e8f0'),
        fontSize=12,
        spaceAfter=8
    )

    story = []
    
    # Add title
    story.append(Paragraph("AstroExo AI Pipeline Report", title_style))
    story.append(Spacer(1, 0.2 * 72))
    
    # Metadata
    story.append(Paragraph(f"<b>Target ID:</b> {req.target_id}", text_style))
    story.append(Paragraph(f"<b>Report Generated:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", text_style))
    story.append(Spacer(1, 0.1 * 72))
    
    # Classification Results
    story.append(Paragraph("Classification Results", heading_style))
    story.append(Paragraph(f"<b>Primary Prediction:</b> {req.prediction}", text_style))
    story.append(Paragraph(f"<b>Confidence:</b> {req.confidence*100:.1f}%", text_style))
    
    prob_data = [["Class", "Probability"]]
    for cls_name, prob in req.probabilities.items():
        prob_data.append([cls_name, f"{prob*100:.1f}%"])
        
    t = Table(prob_data, colWidths=[200, 100])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e293b')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor('#f8fafc')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 8),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#0f172a')),
        ('TEXTCOLOR', (0,1), (-1,-1), colors.HexColor('#cbd5e1')),
        ('GRID', (0,0), (-1,-1), 1, colors.HexColor('#334155'))
    ]))
    story.append(t)
    story.append(Spacer(1, 0.2 * 72))
    
    # Physical Parameters
    story.append(Paragraph("Derived Parameters", heading_style))
    params_data = [
        ["Parameter", "Value"],
        ["Orbital Period", f"{req.period:.3f} days"],
        ["Transit Depth", f"{req.depth:.3f}%"],
        ["Transit Duration", f"{req.duration:.2f} hrs"],
        ["Signal-to-Noise Ratio", f"{req.snr:.1f}"],
        ["Est. Planet Radius", f"{req.planet_radius:.2f} R⊕"]
    ]
    
    t_params = Table(params_data, colWidths=[200, 100])
    t_params.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e293b')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor('#f8fafc')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 8),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#0f172a')),
        ('TEXTCOLOR', (0,1), (-1,-1), colors.HexColor('#cbd5e1')),
        ('GRID', (0,0), (-1,-1), 1, colors.HexColor('#334155'))
    ]))
    story.append(t_params)
    story.append(Spacer(1, 0.2 * 72))
    
    # Top Features
    story.append(Paragraph("Top XAI Features (Explainability)", heading_style))
    for feat in req.features:
        story.append(Paragraph(f"• {feat}", text_style))
        
    def add_dark_background(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(colors.HexColor('#0f172a'))
        canvas.rect(0, 0, letter[0], letter[1], fill=1)
        canvas.restoreState()
        
    doc.build(story, onFirstPage=add_dark_background, onLaterPages=add_dark_background)
    
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="application/pdf", headers={
        "Content-Disposition": f"attachment; filename=astroexo-report-{req.target_id}.pdf"
    })

