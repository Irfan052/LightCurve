import numpy as np
from typing import Dict, Any
from backend.phase_fold import fold_lightcurve, bin_folded_lightcurve

def extract_features(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    bls_results: Dict[str, Any]
) -> Dict[str, float]:
    """
    Extracts numerical features from the folded light curve to feed into 
    the machine learning classifier.
    """
    features = {
        "period": 0.0,
        "depth": 0.0,
        "duration": 0.0,
        "snr": 0.0,
        "shape_score": 0.0,      # U-shape (closer to 1.0) vs V-shape (closer to 0)
        "odd_even_diff": 0.0,    # Depth difference between odd and even transits
        "secondary_depth": 0.0,  # Depth of secondary eclipse (phase 0.5)
        "secondary_ratio": 0.0,  # Ratio of secondary to primary depth
        "out_transit_var": 0.0,  # Out-of-transit variability
        "symmetry": 0.0          # Left-right symmetry of the transit profile
    }
    
    period = bls_results.get("period", 0.0)
    epoch = bls_results.get("epoch", 0.0)
    duration = bls_results.get("duration", 0.0)
    depth = bls_results.get("depth", 0.0)
    snr = bls_results.get("snr", 0.0)
    
    if period <= 0 or len(time) < 10 or depth <= 0:
        return features
        
    features["period"] = float(period)
    features["depth"] = float(depth)
    features["duration"] = float(duration)
    features["snr"] = float(snr)
    
    # 1. Fold the light curve
    folded = fold_lightcurve(time, flux, flux_err, period, epoch)
    phase = folded["phase"]
    fflux = folded["flux"]
    
    if len(phase) < 10:
        return features
        
    # Define transit window in phase units
    transit_half_width = (duration / period) / 2.0
    
    # Separate in-transit and out-of-transit points
    in_transit_mask = (phase >= -transit_half_width) & (phase <= transit_half_width)
    out_transit_mask = ~in_transit_mask
    
    # 2. Out-of-transit variability (standard deviation)
    if np.sum(out_transit_mask) > 5:
        features["out_transit_var"] = float(np.nanstd(fflux[out_transit_mask]))
    else:
        features["out_transit_var"] = 0.001
        
    # 3. Bin the folded light curve for shape and secondary eclipse analysis
    bin_centers, bin_flux, _ = bin_folded_lightcurve(phase, fflux, num_bins=100)
    
    # 4. Shape Score: U-shape vs V-shape
    # We measure width of the transit dip at 10% and 80% of the depth
    try:
        baseline = 1.0
        dip_flux = baseline - bin_flux
        
        # Thresholds
        thresh_10 = 0.1 * depth
        thresh_80 = 0.8 * depth
        
        # Find phases inside transit
        in_transit_bins = (bin_centers >= -transit_half_width) & (bin_centers <= transit_half_width)
        
        # Number of bins exceeding thresholds
        bins_10 = np.sum(in_transit_bins & (dip_flux >= thresh_10))
        bins_80 = np.sum(in_transit_bins & (dip_flux >= thresh_80))
        
        # Shape score is the ratio of bottom width to top width
        if bins_10 > 0:
            features["shape_score"] = float(bins_80 / bins_10)
        else:
            features["shape_score"] = 0.0
    except Exception:
        features["shape_score"] = 0.5
        
    # 5. Secondary Eclipse search (around phase 0.5)
    try:
        # Secondary eclipse occurs at phase ~0.5 or -0.5
        sec_mask = (bin_centers >= 0.4) | (bin_centers <= -0.4)
        if np.sum(sec_mask) > 0:
            sec_flux = bin_flux[sec_mask]
            # Deepest dip around phase 0.5
            sec_depth = float(1.0 - np.nanmin(sec_flux))
            features["secondary_depth"] = sec_depth
            features["secondary_ratio"] = float(sec_depth / (depth + 1e-6))
    except Exception:
        pass
        
    # 6. Odd-Even Depth Difference
    try:
        # Find which transit number each point belongs to
        transit_nums = np.round((time - epoch) / period)
        even_mask = (transit_nums % 2 == 0)
        odd_mask = ~even_mask
        
        # Calculate mean flux in-transit for odd and even transits
        time_phase = ((time - epoch) / period) % 1.0
        time_phase = np.where(time_phase >= 0.5, time_phase - 1.0, time_phase)
        in_transit_time_mask = (time_phase >= -transit_half_width) & (time_phase <= transit_half_width)
        
        odd_transit_flux = flux[in_transit_time_mask & odd_mask]
        even_transit_flux = flux[in_transit_time_mask & even_mask]
        
        if len(odd_transit_flux) > 0 and len(even_transit_flux) > 0:
            odd_depth = 1.0 - np.nanmedian(odd_transit_flux)
            even_depth = 1.0 - np.nanmedian(even_transit_flux)
            # Absolute normalized difference
            features["odd_even_diff"] = float(abs(odd_depth - even_depth) / (depth + 1e-6))
    except Exception:
        pass
        
    # 7. Symmetry
    try:
        # Compare left side (phase < 0) with right side (phase > 0)
        left_mask = (bin_centers >= -transit_half_width) & (bin_centers < 0)
        right_mask = (bin_centers > 0) & (bin_centers <= transit_half_width)
        
        left_profile = bin_flux[left_mask]
        right_profile = bin_flux[right_mask][::-1]  # Reverse to align
        
        min_len = min(len(left_profile), len(right_profile))
        if min_len > 3:
            diff = np.abs(left_profile[:min_len] - right_profile[:min_len])
            features["symmetry"] = float(1.0 - np.nanmean(diff))  # closer to 1 means more symmetric
    except Exception:
        features["symmetry"] = 1.0
        
    return features
