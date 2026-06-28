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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import base64

from backend.config import MODEL_PATH
from backend.utils import logger, safe_serialize
from backend.data_loader import (
    load_tess_lightcurve,
    load_uploaded_file,
    detect_csv_type,
    CACHE_STATS
)
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
    logger.info("=== ASTROEXO BUILD ===")
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

@app.get("/api/system")
def get_system_stats():
    return {
        "status": "online",
        "cache_stats": CACHE_STATS
    }

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
        
        logger.info("Pipeline executed successfully. Downsampling arrays for frontend payload...")
        
        # Downsample arrays for frontend visualization if they are too large
        MAX_POINTS = 5000
        
        def downsample(arr: np.ndarray, max_pts: int) -> np.ndarray:
            if len(arr) > max_pts:
                stride = max(1, len(arr) // max_pts)
                return arr[::stride]
            return arr
            
        time_plot = downsample(time, MAX_POINTS)
        flux_plot = downsample(flux, MAX_POINTS)
        time_clean_plot = downsample(time_clean, MAX_POINTS)
        flux_clean_plot = downsample(flux_clean, MAX_POINTS)
        flat_flux_plot = downsample(flat_flux, MAX_POINTS)
        trend_flux_plot = downsample(trend_flux, MAX_POINTS)
        
        logger.info(f"Downsampled for frontend: {len(time)} -> {len(time_plot)} points.")
        
        # Build JSON serializable output payload
        return safe_serialize({
            "status": "SUCCESS",
            "target_name": target_name,
            "is_mock": is_mock,
            "prediction": classification["prediction_label"],
            "confidence": classification["confidence"],
            "probabilities": classification["probabilities"],
            "parameters": parameters,
            "features": features,
            "feature_importance_summary": feature_importance_summary,
            "data": {
                "time": time_plot.tolist(),
                "flux": flux_plot.tolist(),
                "time_clean": time_clean_plot.tolist(),
                "flux_clean": flux_clean_plot.tolist(),
                "flat_flux": flat_flux_plot.tolist(),
                "trend_flux": trend_flux_plot.tolist(),
                "folded_phase": folded["phase"].tolist(),
                "folded_flux": folded["flux"].tolist(),
                "bin_centers": bin_centers.tolist(),
                "bin_flux": bin_flux.tolist(),
                "bls_periods": bls_results.get("periods", np.array([])).tolist(),
                "bls_powers": bls_results.get("powers", np.array([])).tolist(),
            }
        })
        
    except Exception as e:
        logger.exception("Error during pipeline execution")
        return {"status": "UNKNOWN_BACKEND_ERROR", "message": f"Pipeline error: {str(e)}"}

@app.post("/api/analyze")
def analyze_target(request: AnalyzeRequest):
    """Downloads a TESS target ID or generates mock data and runs the analysis."""
    
    target_id = request.target_id.strip()
    if not target_id:
        return {"status": "INVALID_TIC", "target": target_id, "message": "Target ID cannot be empty."}
        
    # Load light curve data
    try:
        data = load_tess_lightcurve(target_id)
    except Exception as e:
        error_str = str(e)
        if "|" in error_str:
            status, msg = error_str.split("|", 1)
            logger.warning(f"Analysis Failed ({status}) for {target_id}: {msg}")
            return {"status": status.strip(), "target": target_id, "message": msg.strip()}
        else:
            logger.exception(f"Unexpected error for {target_id}")
            return {"status": "UNKNOWN_BACKEND_ERROR", "target": target_id, "message": str(e)}
        
    import time
    pipeline_start = time.time()
    
    result = run_analysis_pipeline(
        time=data["time"],
        flux=data["flux"],
        flux_err=data["flux_err"],
        target_name=data["target_name"],
        is_mock=data["is_mock"]
    )
    
    pipeline_time = time.time() - pipeline_start
    
    if result.get("status") == "SUCCESS" and "metadata" in data:
        result["metadata"] = data["metadata"]
    elif "status" not in result:
        result["status"] = "SUCCESS"
        if "metadata" in data:
            result["metadata"] = data["metadata"]
            
    if result.get("status") != "SUCCESS":
        result["target"] = target_id
        return result
    
    timings = data.get("timings", {"search": 0, "download": 0})
    total_time = timings["search"] + timings["download"] + pipeline_time
    
    logger.info(f"--- Timing Report for {target_id} ---")
    logger.info(f"MAST Search Time: {timings['search']:.2f}s")
    logger.info(f"Download Time: {timings['download']:.2f}s")
    logger.info(f"Pipeline Time: {pipeline_time:.2f}s")
    logger.info(f"Total Analysis Time: {total_time:.2f}s")
    
    return result

import math
import pandas as pd

def sanitize_for_json(obj: Any) -> Any:
    """Recursively sanitizes objects to be JSON compliant."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(sanitize_for_json(v) for v in obj)
    elif isinstance(obj, (pd.Series, pd.Index)):
        return sanitize_for_json(obj.tolist())
    elif isinstance(obj, pd.DataFrame):
        return sanitize_for_json(obj.to_dict(orient='records'))
    elif isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    elif isinstance(obj, (np.float32, np.float64)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    elif isinstance(obj, (np.int32, np.int64, np.int8, np.int16)):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj

@app.post("/api/upload")
async def upload_and_analyze(file: UploadFile = File(...)):
    """Accepts an uploaded CSV light curve file and runs the analysis."""
    # Ensure temporary upload directory exists
    import uuid
    temp_dir = Path("data/raw/uploads")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Verify upload size limits (Max 20MB)
    # Maximum upload size = 3 GB
    MAX_UPLOAD_SIZE = 3 * 1024 * 1024 * 1024  # 3GB
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    if file_size > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Maximum upload size is 3GB.")
    
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
            sanitized = sanitize_for_json(data)
            return sanitized
            
        # Execute pipeline
        result = run_analysis_pipeline(
            time=data["time"],
            flux=data["flux"],
            flux_err=data["flux_err"],
            target_name=data["target_name"],
            is_mock=data.get("is_mock", False)
        )
        
        return sanitize_for_json(result)
        
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
    semi_major_axis: float
    snr: float
    probabilities: Dict[str, float]
    is_mock: bool = False
    xai_evidence: List[str]
    xai_risks: List[str]
    xai_recommendation: str
    xai_interpretation: str
    feature_importance: List[Dict[str, Any]]
    plot_raw: str
    plot_detrended: str
    plot_bls: str
    plot_folded: str
    plot_model: str

@app.post("/api/report")
def generate_pdf_report(req: ReportRequest):
    buffer = io.BytesIO()
    
    # Setup document with dark theme
    doc = SimpleDocTemplate(
        buffer, pagesize=letter,
        rightMargin=36, leftMargin=36,
        topMargin=36, bottomMargin=36,
        title=f"AstroExo Report: {req.target_id}",
        author="AstroExo AI Platform"
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], textColor=colors.HexColor('#ffffff'), backColor=colors.HexColor('#0f172a'), alignment=1, spaceAfter=20, fontSize=24)
    heading_style = ParagraphStyle('HeadingStyle', parent=styles['Heading2'], textColor=colors.HexColor('#38bdf8'), spaceAfter=10, spaceBefore=20)
    text_style = ParagraphStyle('TextStyle', parent=styles['Normal'], textColor=colors.HexColor('#e2e8f0'), fontSize=11, spaceAfter=6, leading=14)
    bullet_style = ParagraphStyle('BulletStyle', parent=styles['Normal'], textColor=colors.HexColor('#e2e8f0'), fontSize=11, spaceAfter=4, leading=14, leftIndent=20)
    risk_style = ParagraphStyle('RiskStyle', parent=styles['Normal'], textColor=colors.HexColor('#ef4444'), fontSize=11, spaceAfter=4, leading=14, leftIndent=20)
    evidence_style = ParagraphStyle('EvidenceStyle', parent=styles['Normal'], textColor=colors.HexColor('#10b981'), fontSize=11, spaceAfter=4, leading=14, leftIndent=20)

    story = []
    
    # === Cover Page / Executive Summary ===
    story.append(Paragraph("AstroExo AI Powered Exoplanet Detection", title_style))
    story.append(Spacer(1, 0.2 * 72))
    
    story.append(Paragraph("Executive Summary", heading_style))
    story.append(Paragraph(f"<b>Target ID:</b> {req.target_id}", text_style))
    story.append(Paragraph(f"<b>Generation Date:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", text_style))
    story.append(Paragraph(f"<b>Primary Prediction:</b> {req.prediction.replace('_', ' ').upper()}", text_style))
    story.append(Paragraph(f"<b>AI Confidence:</b> {req.confidence*100:.1f}% ({req.xai_interpretation})", text_style))
    story.append(Paragraph(f"<b>Overall Assessment:</b> {req.xai_recommendation}", text_style))
    story.append(Spacer(1, 0.1 * 72))

    # === Observation Summary ===
    story.append(Paragraph("Observation Summary", heading_style))
    obs_data = [
        ["Mission", "TESS"],
        ["Data Source", "MAST Archive (Synthetic)" if req.is_mock else "MAST Archive"],
        ["Target", req.target_id],
        ["Status", "Processed Successfully"]
    ]
    t_obs = Table(obs_data, colWidths=[200, 200])
    t_obs.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#1e293b')),
        ('TEXTCOLOR', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('GRID', (0,0), (-1,-1), 1, colors.HexColor('#334155')),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8)
    ]))
    story.append(t_obs)

    # === Scientific Parameters ===
    story.append(Paragraph("Scientific Parameters", heading_style))
    params_data = [
        ["Parameter", "Value"],
        ["Orbital Period", f"{req.period:.4f} days"],
        ["Transit Depth", f"{req.depth:.3f}%"],
        ["Transit Duration", f"{req.duration:.2f} hrs"],
        ["Signal-to-Noise Ratio", f"{req.snr:.1f}"],
        ["Est. Planet Radius", f"{req.planet_radius:.2f} R⊕"],
        ["Semi-major Axis", f"{req.semi_major_axis:.4f} AU" if req.semi_major_axis > 0 else "Not Available"]
    ]
    t_params = Table(params_data, colWidths=[250, 150])
    t_params.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0f172a')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor('#f8fafc')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#1e293b')),
        ('TEXTCOLOR', (0,1), (-1,-1), colors.HexColor('#cbd5e1')),
        ('GRID', (0,0), (-1,-1), 1, colors.HexColor('#334155'))
    ]))
    story.append(t_params)

    # === Explainable AI ===
    story.append(PageBreak())
    story.append(Paragraph("Explainable AI (XAI) Analysis", heading_style))
    
    story.append(Paragraph("Scientific Evidence:", text_style))
    if req.xai_evidence:
        for ev in req.xai_evidence:
            story.append(Paragraph(f"✓ {ev}", evidence_style))
    else:
        story.append(Paragraph("No strong supporting evidence found.", text_style))
        
    story.append(Spacer(1, 0.05 * 72))
    story.append(Paragraph("Risk Indicators:", text_style))
    if req.xai_risks:
        for risk in req.xai_risks:
            story.append(Paragraph(f"⚠ {risk}", risk_style))
    else:
        story.append(Paragraph("No significant scientific risk indicators detected.", evidence_style))

    story.append(Spacer(1, 0.05 * 72))
    story.append(Paragraph("Feature Importance (Random Forest):", text_style))
    for feat in req.feature_importance:
        story.append(Paragraph(f"• {feat.get('feature', '').replace('_', ' ').title()}: {feat.get('importance', 0)*100:.1f}%", bullet_style))

    def add_plot(b64_str, title, aspect_ratio=0.5):
        if not b64_str: return
        try:
            if ',' in b64_str:
                b64_str = b64_str.split(',')[1]
            img_data = base64.b64decode(b64_str)
            img_buf = io.BytesIO(img_data)
            w = 540
            h = w * aspect_ratio
            img = RLImage(img_buf, width=w, height=h)
            story.append(Paragraph(title, heading_style))
            story.append(img)
            story.append(Spacer(1, 0.05 * 72))
        except Exception as e:
            logger.error(f"Failed to embed image {title}: {e}")

    # === Scientific Figures ===
    story.append(PageBreak())
    story.append(Paragraph("Scientific Visualizations", title_style))
    add_plot(req.plot_raw, "1. Raw Light Curve", 1.0)
    
    story.append(PageBreak())
    add_plot(req.plot_detrended, "2. Detrended Flux & Trend", 1.0)
    
    story.append(PageBreak())
    add_plot(req.plot_bls, "3. BLS Periodogram", 0.5)
    add_plot(req.plot_folded, "4. Phase Folded Observations", 0.5)
    
    story.append(PageBreak())
    add_plot(req.plot_model, "5. Transit Model Overlay", 0.5)

    # === Final Conclusion ===
    story.append(Paragraph("Final Conclusion", heading_style))
    story.append(Paragraph(
        f"Based on the analysis of the TESS light curve for {req.target_id}, the pipeline extracted a periodic transit signal "
        f"at {req.period:.4f} days with an SNR of {req.snr:.1f}. The Random Forest classifier predicted this target as "
        f"'{req.prediction.replace('_', ' ').upper()}' with a confidence of {req.confidence*100:.1f}%. "
        f"Given the derived parameters and morphological evidence, the recommendation is: {req.xai_recommendation}", text_style))
        
    def add_page_decorations(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(colors.HexColor('#0f172a'))
        canvas.rect(0, 0, letter[0], letter[1], fill=1)
        canvas.setFillColor(colors.HexColor('#64748b'))
        canvas.setFont('Helvetica', 9)
        canvas.drawString(36, 20, "AstroExo AI Scientific Analysis Report")
        canvas.drawRightString(letter[0] - 36, 20, f"Page {canvas.getPageNumber()}")
        canvas.restoreState()
        
    doc.build(story, onFirstPage=add_page_decorations, onLaterPages=add_page_decorations)
    
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="application/pdf", headers={
        "Content-Disposition": f"attachment; filename=astroexo-report-{req.target_id}.pdf"
    })
