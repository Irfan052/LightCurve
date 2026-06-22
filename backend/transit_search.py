import numpy as np
from astropy.timeseries import BoxLeastSquares
from typing import Dict, Any
from backend.config import (
    BLS_MIN_PERIOD, BLS_MAX_PERIOD, BLS_OVERSAMPLE, 
    BLS_DURATION_MIN, BLS_DURATION_MAX
)
from backend.utils import logger

def search_transits(
    time: np.ndarray, 
    flux: np.ndarray, 
    flux_err: np.ndarray
) -> Dict[str, Any]:
    """
    Performs a Box Least Squares (BLS) periodogram search to find the most 
    dominant periodic transit-like signal.
    
    Returns:
        A dictionary containing:
            - period: Best-fit orbital period in days
            - epoch: Transit epoch (T0) in days
            - duration: Transit duration in days
            - depth: Transit depth
            - snr: Signal-to-noise ratio of the detection
            - bls_power: Peak power in the periodogram
            - periods: Array of search periods (for plotting/diagnostics)
            - powers: Array of BLS power values
            - bls_model: The astropy BLS object (optional)
    """
    logger.info("Initializing Box Least Squares (BLS) periodogram search...")
    
    # Check if there are enough points
    if len(time) < 20:
        logger.warning("Not enough points to run transit search.")
        return {
            "period": 0.0, "epoch": 0.0, "duration": 0.0, "depth": 0.0, 
            "snr": 0.0, "bls_power": 0.0, "periods": np.array([]), "powers": np.array([])
        }
        
    try:
        # Initialize astropy BLS
        bls = BoxLeastSquares(time, flux, flux_err)
        
        # Estimate duration grid based on config fractions
        # Duration is checked as a fraction of the period
        durations = np.linspace(BLS_DURATION_MIN, BLS_DURATION_MAX, 5) # 5 duration test values
        
        # Run auto-power search
        # We need to specify the durations as absolute values, so we pass a range
        # that is reasonable for typical transits (e.g. 0.05 to 0.5 days)
        # Or, we can use the durations parameter directly.
        # Astropy's autopower takes durations in the same units as time (days)
        duration_grid = np.linspace(0.05, 0.4, 5) # in days (1.2 to 9.6 hours)
        
        # Calculate period grid
        periodogram = bls.autopower(
            duration_grid, 
            minimum_period=BLS_MIN_PERIOD, 
            maximum_period=min(BLS_MAX_PERIOD, (time[-1] - time[0]) / 2.0),
            oversample=BLS_OVERSAMPLE
        )
        
        # Find the peak power index
        peak_idx = np.argmax(periodogram.power)
        
        best_period = float(periodogram.period[peak_idx])
        best_epoch = float(periodogram.transit_time[peak_idx])
        best_duration = float(periodogram.duration[peak_idx])
        best_depth = float(periodogram.depth[peak_idx])
        best_snr = float(periodogram.depth_snr[peak_idx])
        peak_power = float(periodogram.power[peak_idx])
        
        logger.info(
            f"BLS search complete. Best Period: {best_period:.4f} d, "
            f"Epoch (T0): {best_epoch:.4f} d, Depth: {best_depth*100:.3f}%, SNR: {best_snr:.2f}"
        )
        
        return {
            "period": best_period,
            "epoch": best_epoch,
            "duration": best_duration,
            "depth": best_depth,
            "snr": best_snr,
            "bls_power": peak_power,
            "periods": periodogram.period,
            "powers": periodogram.power
        }
        
    except Exception as e:
        logger.error(f"BLS search failed: {str(e)}. Returning empty results.")
        return {
            "period": 0.0, "epoch": 0.0, "duration": 0.0, "depth": 0.0, 
            "snr": 0.0, "bls_power": 0.0, "periods": np.array([]), "powers": np.array([])
        }
