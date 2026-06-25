import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, Union
import lightkurve as lk
from backend.utils import logger

def generate_synthetic_lightcurve(
    target_type: str, 
    length_days: float = 27.0, 
    cadence_minutes: float = 10.0,
    return_params: bool = False
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]]:
    """
    Generates a realistic synthetic light curve for hackathon testing and demo.
    
    Parameters:
        target_type: 'exoplanet_transit', 'eclipsing_binary', 'stellar_variability', 'instrumental_artifact'
        length_days: total length in days
        cadence_minutes: observational cadence in minutes
        
    Returns:
        time: 1D array of days
        flux: 1D array of relative fluxes (normalized near 1.0)
        flux_err: 1D array of flux uncertainties
    """
    num_points = int((length_days * 24 * 60) / cadence_minutes)
    time = np.linspace(0, length_days, num_points)
    
    # Base flux is 1.0
    flux = np.ones_like(time)
    
    # 1. Add low-frequency stellar variability (rotation, spots)
    # Exclude instrumental artifact from this or keep it low
    if target_type != "instrumental_artifact":
        var_period = np.random.uniform(3.0, 8.0)
        var_amp = np.random.uniform(0.002, 0.008)
        # Combine a couple of harmonics
        flux += var_amp * np.sin(2 * np.pi * time / var_period)
        flux += (var_amp * 0.3) * np.sin(4 * np.pi * time / var_period + 1.2)
    else:
        # Instrumental artifacts have sudden steps/offsets
        flux += 0.01 * np.sin(2 * np.pi * time / 15.0)
        
    # 2. Add White Noise
    noise_level = 0.001  # 1000 ppm
    if target_type == "instrumental_artifact":
        noise_level = 0.005  # much noisier
    elif target_type == "stellar_variability":
        noise_level = 0.0015
    
    flux_err = np.full_like(time, noise_level)
    flux += np.random.normal(0, noise_level, size=num_points)
    
    ground_truth = {
        "period": 0.0,
        "epoch": 0.0,
        "depth": 0.0,
        "duration": 0.0,
        "snr": 5.0
    }

    # 3. Inject signals based on class
    if target_type == "exoplanet_transit":
        # Period, Epoch, Depth, Duration
        period = np.random.uniform(2.5, 5.5)
        epoch = np.random.uniform(0.2, 1.2)
        depth = np.random.uniform(0.006, 0.015)  # 0.6% to 1.5%
        duration_hours = np.random.uniform(2.5, 4.5)
        duration_days = duration_hours / 24.0
        
        ground_truth.update({
            "period": period,
            "epoch": epoch,
            "depth": depth,
            "duration": duration_days,
            "snr": 20.0
        })

        # Calculate phases and inject U-shaped transits
        phases = ((time - epoch) % period) / period
        # Map phases to range [-0.5, 0.5]
        phases = np.where(phases > 0.5, phases - 1.0, phases)
        
        # Super-gaussian for U-shape flat-bottom transit
        width_phase = (duration_days / period) / 2.0
        # Transit model (1 - depth * exp(-(phase/width)^8))
        transit_signal = -depth * np.exp(-((phases / width_phase) ** 8))
        flux += transit_signal
        
    elif target_type == "eclipsing_binary":
        # Period, Epoch, Primary Depth, Secondary Depth
        period = np.random.uniform(3.0, 7.0)
        epoch = np.random.uniform(0.2, 1.2)
        primary_depth = np.random.uniform(0.04, 0.08)  # Deep eclipse
        secondary_depth = np.random.uniform(0.008, 0.02)  # Secondary eclipse
        duration_hours = np.random.uniform(3.0, 5.0)
        duration_days = duration_hours / 24.0
        
        ground_truth.update({
            "period": period,
            "epoch": epoch,
            "depth": primary_depth,
            "duration": duration_days,
            "snr": 50.0
        })

        # Primary eclipse (at phase 0.0)
        phases_pri = ((time - epoch) % period) / period
        phases_pri = np.where(phases_pri > 0.5, phases_pri - 1.0, phases_pri)
        width_phase = (duration_days / period) / 2.0
        # V-shape model (standard Gaussian exponent 2 or trapezoidal)
        primary_signal = -primary_depth * np.exp(-((phases_pri / width_phase) ** 2))
        
        # Secondary eclipse (at phase 0.5)
        phases_sec = ((time - epoch - (period / 2.0)) % period) / period
        phases_sec = np.where(phases_sec > 0.5, phases_sec - 1.0, phases_sec)
        secondary_signal = -secondary_depth * np.exp(-((phases_sec / width_phase) ** 2))
        
        flux += primary_signal + secondary_signal
        
    elif target_type == "stellar_variability":
        # High amplitude stellar oscillations (spots/pulsator)
        osc_period = np.random.uniform(0.5, 2.0)
        osc_amp = np.random.uniform(0.015, 0.035)  # 1.5% to 3.5%
        flux += osc_amp * np.sin(2 * np.pi * time / osc_period)
        
    elif target_type == "instrumental_artifact":
        # Random cosmic ray spikes
        num_spikes = np.random.randint(3, 8)
        spike_indices = np.random.choice(num_points, num_spikes, replace=False)
        flux[spike_indices] += np.random.choice([-0.05, 0.05], num_spikes)
        
        # A sudden baseline offset shift (e.g. thruster firing / momentum dump)
        shift_idx = int(num_points * np.random.uniform(0.3, 0.7))
        flux[shift_idx:] += np.random.uniform(-0.015, 0.015)
        
    if return_params:
        return time, flux, flux_err, ground_truth
    return time, flux, flux_err

def load_tess_lightcurve(tic_id: str) -> Dict[str, Any]:
    """
    Downloads a TESS light curve for the given TIC ID. 
    If the TIC ID is mock (e.g., TIC 9991, TIC 9992, TIC 9993, TIC 9994) or if the MAST API fails,
    it automatically falls back to generating high-fidelity synthetic data.
    """
    tic_str = tic_id.upper()
    if tic_str.startswith("TIC"):
        tic_str = tic_str[3:].strip()
    else:
        tic_str = tic_str.strip()
    
    # Check for synthetic/mock overrides
    # TIC 9991: Exoplanet, TIC 9992: EB, TIC 9993: Stellar Var, TIC 9994: Artifact
    if tic_str == "9991":
        logger.info("Generating mock Exoplanet Transit light curve...")
        time, flux, flux_err = generate_synthetic_lightcurve("exoplanet_transit")
        return {"time": time, "flux": flux, "flux_err": flux_err, "target_name": "TIC 9991 (Mock Exoplanet)", "is_mock": True}
    elif tic_str == "9992":
        logger.info("Generating mock Eclipsing Binary light curve...")
        time, flux, flux_err = generate_synthetic_lightcurve("eclipsing_binary")
        return {"time": time, "flux": flux, "flux_err": flux_err, "target_name": "TIC 9992 (Mock Eclipsing Binary)", "is_mock": True}
    elif tic_str == "9993":
        logger.info("Generating mock Stellar Variability light curve...")
        time, flux, flux_err = generate_synthetic_lightcurve("stellar_variability")
        return {"time": time, "flux": flux, "flux_err": flux_err, "target_name": "TIC 9993 (Mock Stellar Var)", "is_mock": True}
    elif tic_str == "9994":
        logger.info("Generating mock Instrumental Artifact light curve...")
        time, flux, flux_err = generate_synthetic_lightcurve("instrumental_artifact")
        return {"time": time, "flux": flux, "flux_err": flux_err, "target_name": "TIC 9994 (Mock Artifact)", "is_mock": True}
        
    logger.info(f"Searching MAST for real TESS data for TIC {tic_str}...")
    try:
        # Search for TESS light curves
        search_result = lk.search_lightcurve(f"TIC {tic_str}", mission="TESS")
        if len(search_result) == 0:
            raise ValueError(f"No TESS data found for TIC {tic_str}. It may not have been observed by TESS or the ID is invalid.")
            
        # Download the first available sector
        logger.info(f"Downloading light curve for TIC {tic_str}...")
        lc = search_result[0].download()
        
        if lc is None:
            raise ValueError(f"Failed to download light curve for TIC {tic_str}. The download might be corrupted or network failed.")
        
        # Extract quality-filtered values
        # Keep only good quality flags (quality == 0)
        mask = (lc.quality == 0)
        time = lc.time[mask].value
        flux = lc.flux[mask].value
        flux_err = lc.flux_err[mask].value
        
        if len(time) == 0:
            raise ValueError(f"No high-quality data points available for TIC {tic_str} after filtering.")
        
        # Normalize the flux if not already normalized
        median_flux = np.nanmedian(flux)
        if median_flux > 0:
            flux = flux / median_flux
            flux_err = flux_err / median_flux
            
        logger.info(f"Successfully downloaded TIC {tic_str} with {len(time)} data points.")
        return {
            "time": time,
            "flux": flux,
            "flux_err": flux_err,
            "target_name": f"TIC {tic_str}",
            "is_mock": False
        }
        
    except lk.search.SearchError as e:
        logger.error(f"Network or MAST API failure for TIC {tic_str}: {str(e)}")
        raise ConnectionError(f"Network failure while communicating with MAST API: {str(e)}")
    except ValueError as e:
        logger.error(str(e))
        raise
    except Exception as e:
        logger.error(f"Unexpected error fetching data for TIC {tic_str}: {str(e)}")
        raise RuntimeError(f"Corrupted download or unexpected error: {str(e)}")

def load_uploaded_file(file_path: str) -> Dict[str, Any]:
    """Reads a CSV file containing columns: time, flux, and optionally flux_err."""
    logger.info(f"Loading uploaded file: {file_path}")
    ext = Path(file_path).suffix.lower()
    
    if ext != ".csv":
        raise ValueError(f"Unsupported file type: {ext}. Only CSV files are supported.")
        
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Failed to parse CSV file: {str(e)}. Please ensure it is a valid text CSV.")
        
    if df.empty:
        raise ValueError("The uploaded CSV file is empty.")
        
    # Standardize column headers (case insensitive)
    df.columns = [str(c).lower().strip() for c in df.columns]
    
    if "time" not in df.columns or "flux" not in df.columns:
        raise ValueError("CSV file must contain 'time' and 'flux' columns.")
        
    # Coerce to numeric, turning non-numeric into NaN
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["flux"] = pd.to_numeric(df["flux"], errors="coerce")
    
    # Check for excessive NaNs or entirely non-numeric columns
    if df["time"].isna().all() or df["flux"].isna().all():
        raise ValueError("The 'time' and 'flux' columns must contain numeric values.")
        
    time = df["time"].values
    flux = df["flux"].values
    
    if "flux_err" in df.columns:
        df["flux_err"] = pd.to_numeric(df["flux_err"], errors="coerce")
        flux_err = df["flux_err"].values
    else:
        flux_err = np.full_like(flux, 0.001)
        
    # Sort by time
    sort_idx = np.argsort(time)
    time = time[sort_idx]
    flux = flux[sort_idx]
    flux_err = flux_err[sort_idx]
    
    # Clean NaNs
    nan_mask = ~(np.isnan(time) | np.isnan(flux) | np.isnan(flux_err))
    time = time[nan_mask]
    flux = flux[nan_mask]
    flux_err = flux_err[nan_mask]
    
    if len(time) == 0:
        raise ValueError("CSV file contained no valid numeric data points after cleaning.")
        
    # Normalize
    median_flux = np.nanmedian(flux)
    if median_flux > 0:
        flux = flux / median_flux
        flux_err = flux_err / median_flux
        
    return {
        "time": time,
        "flux": flux,
        "flux_err": flux_err,
        "target_name": Path(file_path).stem,
        "is_mock": False
    }
