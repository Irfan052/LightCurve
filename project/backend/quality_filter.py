import numpy as np
from typing import Tuple, Dict, Any, Optional
from backend.config import SIGMA_CLIP_LIMIT
from backend.utils import logger

# Official TESS Spacecraft Quality Flags (from TESS Science Data Products Description)
TESS_QUALITY_FLAGS = {
    1: "Attitude tweak",
    2: "Safe mode",
    4: "Coarse point",
    8: "Earth point",
    16: "Reaction wheel desaturation (momentum dump)",
    32: "Cosmic ray in collateral data",
    64: "Cosmic ray in science data",
    128: "Sector boundary",
    256: "Stray light anomaly",
    512: "Direct physical event / Earthshine",
    1024: "Spacecraft anomaly / Fire",
    2048: "Stray light flare",
    4096: "Coarse pointing tweak",
    8192: "Orbit boundary"
}

# Standard TESS default quality bitmask (attitude tweaks, safe mode, coarse point, earth point, wheel desaturation, spacecraft fire)
DEFAULT_TESS_BITMASK = 1 + 2 + 4 + 8 + 16 + 1024  # 1055

class CleanedLightCurve(tuple):
    """
    A custom tuple class that behaves exactly like a 3-tuple (time, flux, flux_err)
    to maintain backward compatibility, but stores rich quality control metrics
    under the `.metrics` property.
    """
    def __new__(cls, time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray, metrics: Optional[Dict[str, Any]] = None):
        return super(CleanedLightCurve, cls).__new__(cls, (time, flux, flux_err))
        
    def __init__(self, time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray, metrics: Optional[Dict[str, Any]] = None):
        self.metrics = metrics if metrics is not None else {}

def remove_nans_and_inf(
    time: np.ndarray, 
    flux: np.ndarray, 
    flux_err: np.ndarray,
    quality: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], int]:
    """
    Removes NaN and Infinite values from the light curve arrays.
    Returns cleaned arrays and the count of removed elements.
    """
    if len(time) == 0:
        return time, flux, flux_err, quality, 0
        
    # Check for finite values only (removed the flux > 0 restriction)
    clean_mask = (
        ~np.isnan(time) & 
        ~np.isnan(flux) & 
        ~np.isnan(flux_err) &
        ~np.isinf(time) & 
        ~np.isinf(flux) & 
        ~np.isinf(flux_err)
    )
    
    removed_count = int(len(time) - np.sum(clean_mask))
    cleaned_quality = quality[clean_mask] if quality is not None else None
    return time[clean_mask], flux[clean_mask], flux_err[clean_mask], cleaned_quality, removed_count

def mask_tess_quality_flags(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    quality: np.ndarray,
    bitmask: int = DEFAULT_TESS_BITMASK,
    strict_quality_mode: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, Dict[str, int]]:
    """
    Applies TESS quality flag masking.
    - strict_quality_mode=True: removes any point where quality != 0
    - strict_quality_mode=False: removes points where (quality & bitmask) > 0
    Returns cleaned arrays, the total count of removed points, and flag breakdown statistics.
    """
    if len(time) == 0:
        return time, flux, flux_err, 0, {}
        
    if len(quality) != len(time):
        logger.warning("Quality array length mismatch. Skipping quality flag masking.")
        return time, flux, flux_err, 0, {}
        
    if strict_quality_mode:
        bad_quality_mask = (quality != 0)
    else:
        bad_quality_mask = (quality & bitmask) > 0
        
    keep_mask = ~bad_quality_mask
    removed_count = int(np.sum(bad_quality_mask))
    
    # Calculate detailed breakdown of which flags caused removals (only makes sense if not strictly any quality != 0,
    # but we can list active bits for both modes)
    flag_breakdown = {}
    if removed_count > 0:
        bad_quality_values = quality[bad_quality_mask]
        for flag_val, flag_name in TESS_QUALITY_FLAGS.items():
            if strict_quality_mode or ((flag_val & bitmask) > 0):
                occurrences = int(np.sum((bad_quality_values & flag_val) > 0))
                if occurrences > 0:
                    flag_breakdown[flag_name] = occurrences
                    
    return time[keep_mask], flux[keep_mask], flux_err[keep_mask], removed_count, flag_breakdown

def sigma_clip(
    time: np.ndarray, 
    flux: np.ndarray, 
    flux_err: np.ndarray, 
    sigma_upper: float = SIGMA_CLIP_LIMIT, 
    sigma_lower: float = 5.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Performs asymmetric sigma clipping to remove extreme outliers.
    Returns cleaned arrays and the count of removed elements.
    """
    if len(flux) < 10:
        return time, flux, flux_err, 0
        
    median = np.nanmedian(flux)
    std = np.nanstd(flux)
    
    if std == 0:
        std = 1e-6
        
    keep_mask = (flux >= (median - sigma_lower * std)) & (flux <= (median + sigma_upper * std))
    removed_count = int(len(flux) - np.sum(keep_mask))
    return time[keep_mask], flux[keep_mask], flux_err[keep_mask], removed_count

def detect_observation_gaps(
    time: np.ndarray, 
    gap_threshold_days: float = 0.5
) -> Dict[str, Any]:
    """
    Identifies observation gaps (e.g. TESS orbit downlinks) in the time series.
    Returns a dictionary of gap statistics (gaps found, maximum duration, duty cycle, and indices).
    """
    stats = {
        "gap_count": 0,
        "max_gap_days": 0.0,
        "total_gap_days": 0.0,
        "duty_cycle": 1.0,
        "gap_intervals": [],
        "gap_indices": []  # Added gap_indices
    }
    
    if len(time) < 2:
        return stats
        
    total_duration = time[-1] - time[0]
    if total_duration <= 0:
        return stats
        
    dt = np.diff(time)
    median_dt = np.nanmedian(dt)
    threshold = max(gap_threshold_days, 5.0 * median_dt)
    
    gap_indices = np.where(dt > threshold)[0]
    
    if len(gap_indices) > 0:
        stats["gap_indices"] = gap_indices.tolist()  # Convert numpy array to standard list of ints
        stats["gap_count"] = len(gap_indices)
        gap_durations = dt[gap_indices]
        stats["max_gap_days"] = float(np.max(gap_durations))
        stats["total_gap_days"] = float(np.sum(gap_durations))
        stats["duty_cycle"] = float((total_duration - stats["total_gap_days"]) / total_duration)
        
        for idx in gap_indices:
            stats["gap_intervals"].append({
                "start": float(time[idx]),
                "end": float(time[idx + 1]),
                "duration": float(dt[idx]),
                "pre_gap_index": int(idx)
            })
            
    return stats

def clean_lightcurve(
    time: np.ndarray, 
    flux: np.ndarray, 
    flux_err: np.ndarray,
    quality: Optional[np.ndarray] = None,
    quality_bitmask: int = DEFAULT_TESS_BITMASK,
    sigma_upper: float = SIGMA_CLIP_LIMIT,
    sigma_lower: float = 5.0,
    gap_threshold_days: float = 0.5,
    strict_quality_mode: bool = False
) -> CleanedLightCurve:
    """
    Applies the scientific cleaning pipeline:
    1. Removes NaNs, Infinities.
    2. Applies TESS spacecraft quality flag masking (if quality array is provided).
    3. Detects sampling gaps (e.g. orbit downlinks) and computes metrics.
    4. Performs asymmetric sigma-clipping.
    """
    total_raw = len(time)
    logger.info(f"Quality Filter: Processing light curve with {total_raw} raw points.")
    
    if len(flux) != total_raw or len(flux_err) != total_raw:
        raise ValueError(
            f"Array length mismatch: time ({total_raw}), flux ({len(flux)}), flux_err ({len(flux_err)})"
        )
        
    metrics = {
        "total_raw_points": total_raw,
        "nan_inf_removed": 0,
        "quality_array_available": quality is not None,  # Added metric
        "quality_flag_removed": 0,
        "quality_points_retained": total_raw,            # Added metric (initialize with all, update later)
        "sigma_clipped_removed": 0,
        "quality_flag_breakdown": {},
        "gap_count": 0,
        "max_gap_days": 0.0,
        "duty_cycle": 1.0,
        "gap_indices": []                                # Added metric
    }
    
    try:
        # Step 1: Remove NaNs and Infinities
        t_work, f_work, fe_work, q_work, nan_inf_count = remove_nans_and_inf(time, flux, flux_err, quality)
        metrics["nan_inf_removed"] = nan_inf_count
        
        # Step 2: Quality Flag Bitmasking
        if q_work is not None:
            t_work, f_work, fe_work, q_removed, breakdown = mask_tess_quality_flags(
                t_work, f_work, fe_work, q_work, quality_bitmask, strict_quality_mode
            )
            metrics["quality_flag_removed"] = q_removed
            metrics["quality_flag_breakdown"] = breakdown
            metrics["quality_points_retained"] = len(t_work)
        else:
            metrics["quality_points_retained"] = len(t_work)
            
        # Step 3: Gap Detection
        gap_stats = detect_observation_gaps(t_work, gap_threshold_days)
        metrics.update(gap_stats)
        
        # Step 4: Asymmetric Sigma-Clipping
        t_clean, f_clean, fe_clean, sigma_count = sigma_clip(
            t_work, f_work, fe_work, sigma_upper, sigma_lower
        )
        metrics["sigma_clipped_removed"] = sigma_count
        
        total_clean = len(t_clean)
        metrics["total_cleaned_points"] = total_clean
        metrics["total_removed_points"] = total_raw - total_clean
        
        logger.info(
            f"Quality Filter Complete: Cleaned={total_clean}/{total_raw} | "
            f"Removed={metrics['total_removed_points']} (NaNs={nan_inf_count}, "
            f"Flags={metrics['quality_flag_removed']}, SigmaClip={sigma_count}) | "
            f"Gaps={metrics['gap_count']} (Max={metrics['max_gap_days']:.2f}d, Duty={metrics['duty_cycle']*100:.1f}%)"
        )
        
        return CleanedLightCurve(t_clean, f_clean, fe_clean, metrics)
        
    except Exception as e:
        logger.error(f"Error during quality filtering pipeline: {str(e)}")
        metrics["pipeline_error"] = str(e)
        return CleanedLightCurve(time, flux, flux_err, metrics)
