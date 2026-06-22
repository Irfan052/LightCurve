import numpy as np
from typing import Tuple, Dict, Any
from backend.utils import logger

def fold_lightcurve(
    time: np.ndarray, 
    flux: np.ndarray, 
    flux_err: np.ndarray,
    period: float, 
    epoch: float
) -> Dict[str, Any]:
    """
    Folds the light curve at a specified period and epoch, centering the 
    transit at phase 0.0.
    
    Returns a dictionary of folded phase, flux, and flux error, sorted by phase.
    """
    if period <= 0:
        logger.warning("Invalid period for phase folding. Returning empty arrays.")
        return {"phase": np.array([]), "flux": np.array([]), "flux_err": np.array([])}
        
    # Calculate phase in range [0, 1)
    phase = ((time - epoch) / period) % 1.0
    
    # Re-center phase to [-0.5, 0.5) so transit (at epoch) is at 0.0
    phase = np.where(phase >= 0.5, phase - 1.0, phase)
    
    # Sort by phase
    sort_idx = np.argsort(phase)
    folded_phase = phase[sort_idx]
    folded_flux = flux[sort_idx]
    folded_flux_err = flux_err[sort_idx]
    
    return {
        "phase": folded_phase,
        "flux": folded_flux,
        "flux_err": folded_flux_err
    }

def bin_folded_lightcurve(
    phase: np.ndarray, 
    flux: np.ndarray, 
    num_bins: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Groups the phase-folded light curve into uniform bins and calculates 
    the mean flux and standard error for each bin.
    """
    if len(phase) == 0:
        return np.array([]), np.array([]), np.array([])
        
    bin_edges = np.linspace(-0.5, 0.5, num_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    binned_flux = np.zeros(num_bins)
    binned_err = np.zeros(num_bins)
    
    for i in range(num_bins):
        mask = (phase >= bin_edges[i]) & (phase < bin_edges[i+1])
        if np.sum(mask) > 0:
            binned_flux[i] = np.nanmean(flux[mask])
            binned_err[i] = np.nanstd(flux[mask]) / np.sqrt(np.sum(mask) + 1e-9)
        else:
            # If a bin is empty, interpolate from neighbors or fill with 1.0
            binned_flux[i] = 1.0
            binned_err[i] = 0.0
            
    # Clean any remaining NaNs in empty bins
    nan_mask = np.isnan(binned_flux)
    binned_flux[nan_mask] = 1.0
    binned_err[nan_mask] = 0.0
    
    return bin_centers, binned_flux, binned_err
