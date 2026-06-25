import json
from pathlib import Path
from backend.data_loader import load_tess_lightcurve
from backend.quality_filter import clean_lightcurve
from backend.detrend import flatten_lightcurve
from backend.transit_search import search_transits
from backend.feature_engineering import extract_features
from backend.classifier import predict_class
from backend.utils import logger

# 10 Known TESS Targets for Validation
# Includes exoplanets and non-exoplanets (some might be classified differently by the model)
VALIDATION_TARGETS = [
    "279741379", # TOI-700
    "28159019",  # WASP-126
    "261136679", # Pi Mensae
    "231663901", # WASP-18
    "147977348", # LHS 1140
    "38846515",  # TRAPPIST-1
    "31281820",  # HD 219666
    "167415946", # KELT-9
    "225297752", # WASP-121
    "23636576"   # K2-18
]

def run_real_tess_validation():
    logger.info("Starting Phase 6A: Real TESS Validation...")
    results = []
    failed = 0
    total_conf = 0.0
    
    for tic in VALIDATION_TARGETS:
        try:
            logger.info(f"Processing TIC {tic}...")
            # 1. Load Data
            raw_data = load_tess_lightcurve(tic)
            num_obs = len(raw_data['time'])
            
            # 2. Clean & Detrend
            time_c, flux_c, flux_err_c = clean_lightcurve(
                raw_data['time'], raw_data['flux'], raw_data['flux_err']
            )
            flat_flux, _ = flatten_lightcurve(time_c, flux_c)
            
            # 3. BLS Search
            bls_res = search_transits(time_c, flat_flux, flux_err_c)
            
            # 4. Feature Extraction
            feats = extract_features(time_c, flat_flux, flux_err_c, bls_res)
            
            # 5. Classify
            pred = predict_class(feats)
            
            # Calculate metrics
            period = bls_res.get('period', 0)
            depth = bls_res.get('transit_depth', 0) * 100
            radius = feats.get('planet_radius_earth', 0)
            
            res_dict = {
                "tic_id": tic,
                "observations": num_obs,
                "period_days": round(period, 2),
                "transit_depth_percent": round(depth, 3),
                "classification": pred['prediction_label'],
                "confidence": round(pred['confidence'], 3),
                "radius_earth": round(radius, 2),
                "features": feats # Saving features for Phase 6B
            }
            results.append(res_dict)
            total_conf += pred['confidence']
            
        except Exception as e:
            logger.error(f"Failed processing TIC {tic}: {e}")
            failed += 1

    # Generate Summary Stats
    success = len(results)
    avg_conf = total_conf / success if success > 0 else 0
    
    class_dist = {}
    for r in results:
        c = r['classification']
        class_dist[c] = class_dist.get(c, 0) + 1
        
    summary = {
        "total_targets": len(VALIDATION_TARGETS),
        "successful_downloads": success,
        "failed_downloads": failed,
        "success_rate": round(success / len(VALIDATION_TARGETS), 2),
        "average_confidence": round(avg_conf, 3),
        "classification_distribution": class_dist,
        "results": results
    }
    
    # Save JSON
    Path("real_tess_validation.json").write_text(json.dumps(summary, indent=2), encoding='utf-8')
    
    # Save Markdown
    md = ["# Real TESS Validation Results\n"]
    md.append("| TIC ID | Observations | Period (days) | Depth (%) | Radius (R⊕) | Classification | Confidence |")
    md.append("|--------|--------------|---------------|-----------|-------------|----------------|------------|")
    
    for r in results:
        md.append(
            f"| {r['tic_id']} | {r['observations']} | {r['period_days']} | "
            f"{r['transit_depth_percent']} | {r['radius_earth']} | "
            f"{r['classification']} | {r['confidence'] * 100:.1f}% |"
        )
        
    Path("docs/real_tess_results.md").parent.mkdir(exist_ok=True)
    Path("docs/real_tess_results.md").write_text("\n".join(md), encoding='utf-8')
    
    logger.info("Validation complete. Outputs saved.")

if __name__ == "__main__":
    run_real_tess_validation()
