import numpy as np
from typing import Dict, Any
from backend.utils import logger

def estimate_parameters(
    bls_results: Dict[str, Any],
    classification_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Fits and estimates physical and astronomical parameters of the system,
    including period, depth, duration, and calculated properties like 
    planetary radius and semi-major axis assuming a solar-type host star.
    """
    logger.info("Estimating physical system parameters...")
    
    period = bls_results.get("period", 0.0)
    depth = bls_results.get("depth", 0.0)
    duration = bls_results.get("duration", 0.0)
    snr = bls_results.get("snr", 0.0)
    
    pred_label = classification_results.get("prediction_label", "unknown")
    class_prob = classification_results.get("confidence", 0.0)
    
    # 1. Base estimates
    period_days = float(period)
    epoch_days = float(bls_results.get("epoch", 0.0))
    transit_depth_fraction = float(depth)
    transit_depth_percent = float(depth * 100.0)
    transit_duration_hours = float(duration * 24.0)
    
    # 2. Derive physical characteristics (assuming Solar-type host star: R_* = 1.0 R_solar, M_* = 1.0 M_solar)
    # R_earth = 0.00915 R_solar -> R_p/R_* = sqrt(depth) -> R_p = sqrt(depth) * R_* in Solar Radii
    # R_p (in Earth Radii) = sqrt(depth) / 0.00915 = sqrt(depth) * 109.2
    planet_radius_earth = 0.0
    semi_major_axis_au = 0.0
    
    if transit_depth_fraction > 0:
        planet_radius_earth = float(np.sqrt(transit_depth_fraction) * 109.2)
        
    if period_days > 0:
        # Kepler's Third Law: a^3 = P^2 (in years and AU, for solar mass host star)
        period_years = period_days / 365.25
        semi_major_axis_au = float(np.power(period_years**2, 1/3.0))
        
    # 3. Calculate Detection Confidence Score (0.0 to 1.0)
    # Combines classifier confidence and BLS detection SNR
    # SNR > 15 is considered highly significant. We normalize SNR by 15.0
    normalized_snr = min(1.0, snr / 15.0) if snr > 0 else 0.0
    
    # If it's a stellar variability or artifact, the confidence of it being an exoplanet is 0.
    # Rather, confidence represents the overall confidence of the pipeline's findings.
    # We will formulate finding_confidence = 0.4 * class_prob + 0.6 * normalized_snr
    finding_confidence = float(0.4 * class_prob + 0.6 * normalized_snr)
    
    # Bound it
    finding_confidence = max(0.0, min(1.0, finding_confidence))
    
    logger.info(
        f"Parameter Fit Result: Rp={planet_radius_earth:.2f} R_earth, "
        f"a={semi_major_axis_au:.4f} AU, Confidence={finding_confidence:.2f}"
    )
    
    return {
        "period": period_days,
        "epoch": epoch_days,
        "transit_depth": transit_depth_fraction,
        "transit_depth_percent": transit_depth_percent,
        "transit_duration_hours": transit_duration_hours,
        "planet_radius_earth": planet_radius_earth,
        "semi_major_axis_au": semi_major_axis_au,
        "snr": snr,
        "confidence_score": finding_confidence
    }
