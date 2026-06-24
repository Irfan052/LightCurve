import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from backend.utils import logger
from backend.data_loader import load_tess_lightcurve
from backend.quality_filter import clean_lightcurve
from backend.detrend import flatten_lightcurve
from backend.transit_search import search_transits
from backend.phase_fold import fold_lightcurve
from backend.feature_engineering import extract_features
from backend.classifier import train_classifier, predict_class
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def run_pipeline_for_target(tic_id: str) -> Dict[str, Any]:
    """Runs the full pipeline for a single target."""
    try:
        data = load_tess_lightcurve(tic_id)
        if "error" in data:
            return {"error": data["error"]}
        
        t, f, fe = data["time"], data["flux"], data["flux_err"]
        
        # Quality filter
        t_c, f_c, fe_c = clean_lightcurve(t, f, fe)
        
        # Detrend
        flat_flux, _ = flatten_lightcurve(t_c, f_c)
        
        # Transit search
        bls_res = search_transits(t_c, flat_flux, fe_c)
        if "error" in bls_res:
            return {"error": bls_res["error"]}
            
        if bls_res.get("period", 0) <= 0:
            return {"error": "Invalid BLS period"}
            
        # Feature extraction
        feats = extract_features(t_c, flat_flux, fe_c, bls_res)
        
        # Classification
        pred = predict_class(feats)
        
        return {
            "period": bls_res["period"],
            "depth": bls_res["depth"],
            "duration": bls_res["duration"],
            "prediction_label": pred["prediction_label"],
            "probabilities": pred["probabilities"],
            "confidence": pred["confidence"]
        }
        
    except Exception as e:
        logger.error(f"Pipeline failed for {tic_id}: {str(e)}")
        return {"error": str(e)}

def validate_pipeline(catalog_path: str, report_path: str = "validation_report.json", summary_path: str = "validation_summary.csv") -> Dict[str, Any]:
    """Runs pipeline on benchmark catalog and generates validation metrics."""
    logger.info(f"Loading benchmark catalog from {catalog_path}")
    
    if not os.path.exists(catalog_path):
        raise FileNotFoundError(f"Catalog not found: {catalog_path}")
        
    df = pd.read_csv(catalog_path)
    required_cols = ["tic_id", "true_period", "true_depth", "true_duration", "class_label"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in catalog: {col}")
            
    # Ensure classifier is trained
    train_classifier()
    
    results = []
    errors = []
    
    y_true = []
    y_pred = []
    
    period_errors = []
    depth_errors = []
    duration_errors = []
    
    summary_data = []
    
    for _, row in df.iterrows():
        tic_id = str(row["tic_id"])
        logger.info(f"Validating target: {tic_id}")
        
        res = run_pipeline_for_target(tic_id)
        
        if "error" in res:
            errors.append({"tic_id": tic_id, "error": res["error"]})
            summary_data.append({
                "tic_id": tic_id,
                "status": "failed",
                "error": res["error"]
            })
            continue
            
        pred_label = res["prediction_label"]
        true_label = row["class_label"]
        
        y_true.append(true_label)
        y_pred.append(pred_label)
        
        p_err = abs(res["period"] - row["true_period"])
        d_err = abs(res["depth"] - row["true_depth"])
        dur_err = abs(res["duration"] - row["true_duration"])
        
        period_errors.append(p_err)
        depth_errors.append(d_err)
        duration_errors.append(dur_err)
        
        res_dict = {
            "tic_id": tic_id,
            "status": "success",
            "true_class": true_label,
            "pred_class": pred_label,
            "true_period": float(row["true_period"]),
            "pred_period": float(res["period"]),
            "period_error": float(p_err),
            "true_depth": float(row["true_depth"]),
            "pred_depth": float(res["depth"]),
            "depth_error": float(d_err),
            "true_duration": float(row["true_duration"]),
            "pred_duration": float(res["duration"]),
            "duration_error": float(dur_err),
            "confidence": float(res["confidence"])
        }
        
        results.append(res_dict)
        summary_data.append(res_dict)
        
    metrics = {}
    if len(period_errors) > 0:
        metrics = {
            "mean_period_error": float(np.mean(period_errors)),
            "median_period_error": float(np.median(period_errors)),
            "mean_depth_error": float(np.mean(depth_errors)),
            "mean_duration_error": float(np.mean(duration_errors))
        }
    else:
        metrics = {
            "mean_period_error": 0.0,
            "median_period_error": 0.0,
            "mean_depth_error": 0.0,
            "mean_duration_error": 0.0
        }
        
    if len(y_true) > 0:
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["precision"] = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
        metrics["recall"] = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
        metrics["f1_score"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    else:
        metrics["accuracy"] = 0.0
        metrics["precision"] = 0.0
        metrics["recall"] = 0.0
        metrics["f1_score"] = 0.0
        metrics["confusion_matrix"] = []
        
    report = {
        "metrics": metrics,
        "results": results,
        "errors": errors,
        "total_targets": len(df),
        "successful_targets": len(results),
        "failed_targets": len(errors)
    }
    
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
        
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_path, index=False)
    
    logger.info(f"Validation complete. Report saved to {report_path}")
    
    return report

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=str, default="benchmark_catalog.csv")
    parser.add_argument("--create-dummy", action="store_true")
    args = parser.parse_args()
    
    if args.create_dummy:
        dummy_data = {
            "tic_id": ["9991", "9992", "9993", "9994"],
            "true_period": [3.5, 5.0, 1.2, 0.0],
            "true_depth": [0.01, 0.05, 0.02, 0.0],
            "true_duration": [0.15, 0.2, 0.0, 0.0],
            "class_label": ["exoplanet_transit", "eclipsing_binary", "stellar_variability", "instrumental_artifact"]
        }
        pd.DataFrame(dummy_data).to_csv(args.catalog, index=False)
        print(f"Created dummy catalog at {args.catalog}")
        
    if os.path.exists(args.catalog):
        validate_pipeline(args.catalog)
    else:
        print(f"Catalog {args.catalog} not found. Use --create-dummy to make one.")
