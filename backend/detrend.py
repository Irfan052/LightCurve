import numpy as np
from scipy.signal import savgol_filter
from typing import Tuple
from backend.config import DETREND_WINDOW_DAYS, DETREND_POLYORDER
from backend.utils import logger

def flatten_lightcurve(
    time: np.ndarray, 
    flux: np.ndarray, 
    window_days: float = DETREND_WINDOW_DAYS, 
    polyorder: int = DETREND_POLYORDER
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Flattens the light curve by fitting a Savitzky-Golay filter to capture 
    the low-frequency stellar trend and dividing the flux by this trend.
    
    Returns:
        flat_flux: Flattened relative flux (centered at 1.0)
        trend_flux: Fitted low-frequency stellar variability trend
    """
    if len(time) < 10:
        return flux.copy(), np.ones_like(flux)
        
    # Calculate median cadence in days
    time_diffs = np.diff(time)
    median_dt = np.nanmedian(time_diffs)
    
    if median_dt == 0:
        median_dt = 1e-4
        
    # Convert window_days to number of points
    window_length = int(window_days / median_dt)
    
    # Savitzky-Golay window size must be odd and greater than polyorder
    if window_length % 2 == 0:
        window_length += 1
        
    # Limit window length to be smaller than the array size and at least polyorder+2
    if window_length >= len(flux):
        window_length = len(flux) - 1
        if window_length % 2 == 0:
            window_length -= 1
            
    if window_length <= polyorder:
        window_length = polyorder + 1
        if window_length % 2 == 0:
            window_length += 1
            
    logger.info(f"Applying Savitzky-Golay filter with window length {window_length} points ({window_days} days).")
    
    try:
        # Fit trend
        trend_flux = savgol_filter(flux, window_length=window_length, polyorder=polyorder)
        
        # Flatten by division
        flat_flux = flux / trend_flux
        
    except Exception as e:
        logger.error(f"Savgol filter failed: {str(e)}. Falling back to dividing by median.")
        trend_flux = np.full_like(flux, np.nanmedian(flux))
        flat_flux = flux / trend_flux
        
    return flat_flux, trend_flux
