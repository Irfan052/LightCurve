import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, Union, List
import lightkurve as lk
import requests
from requests.adapters import HTTPAdapter
from backend.utils import logger

# [MINIMAL FIX]: Monkey-patch requests.adapters.HTTPAdapter.send to ALWAYS enforce a timeout.
# This prevents lightkurve/astroquery from hanging indefinitely if they fail to pass a timeout
# or if they override the global socket timeout with None.
_original_send = HTTPAdapter.send

def _patched_send(self, request, stream=False, timeout=None, verify=True, cert=None, proxies=None):
    if timeout is None:
        timeout = 30.0
    return _original_send(self, request, stream, timeout, verify, cert, proxies)

HTTPAdapter.send = _patched_send

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

UNAVAILABLE_TARGETS_CACHE = set()

CACHE_STATS = {
    "hits": 0,
    "misses": 0,
    "recovered_downloads": 0,
    "corrupted_deleted": 0,
    "retries": 0,
    "timeouts": 0,
    "network_errors": 0,
    "success_downloads": 0,
    "failed_downloads": 0
}

try:
    from astroquery.mast import conf as mast_conf
    mast_conf.timeout = 15
    from astropy.utils.data import conf as astropy_conf
    astropy_conf.remote_timeout = 30
except ImportError:
    pass

def load_tess_lightcurve(target_id: str) -> Dict[str, Any]:
    import time
    from datetime import datetime
    """
    Downloads a TESS light curve for the given TIC ID. 
    If the TIC ID is mock (e.g., TIC 9991, TIC 9992, TIC 9993, TIC 9994) or if the MAST API fails,
    it automatically falls back to generating high-fidelity synthetic data.
    """
    tic_str = target_id.upper()
    if tic_str.startswith("TIC"):
        tic_str = tic_str[3:].strip()
    else:
        tic_str = tic_str.strip()
    
    # Check for synthetic/mock overrides
    # TIC 9991: Exoplanet, TIC 9992: EB, TIC 9993: Stellar Var, TIC 9994: Artifact
    if tic_str == "9991":
        logger.info("Generating mock Exoplanet Transit light curve...")
        time_data, flux, flux_err = generate_synthetic_lightcurve("exoplanet_transit")
        return {"time": time_data, "flux": flux, "flux_err": flux_err, "target_name": "TIC 9991 (Mock Exoplanet)", "is_mock": True, "timings": {"search": 0, "download": 0}}
    elif tic_str == "9992":
        logger.info("Generating mock Eclipsing Binary light curve...")
        time_data, flux, flux_err = generate_synthetic_lightcurve("eclipsing_binary")
        return {"time": time_data, "flux": flux, "flux_err": flux_err, "target_name": "TIC 9992 (Mock Eclipsing Binary)", "is_mock": True, "timings": {"search": 0, "download": 0}}
    elif tic_str == "9993":
        logger.info("Generating mock Stellar Variability light curve...")
        time_data, flux, flux_err = generate_synthetic_lightcurve("stellar_variability")
        return {"time": time_data, "flux": flux, "flux_err": flux_err, "target_name": "TIC 9993 (Mock Stellar Var)", "is_mock": True, "timings": {"search": 0, "download": 0}}
    elif tic_str == "9994":
        logger.info("Generating mock Instrumental Artifact light curve...")
        time_data, flux, flux_err = generate_synthetic_lightcurve("instrumental_artifact")
        return {"time": time_data, "flux": flux, "flux_err": flux_err, "target_name": "TIC 9994 (Mock Artifact)", "is_mock": True, "timings": {"search": 0, "download": 0}}
        
    if tic_str in UNAVAILABLE_TARGETS_CACHE:
        raise ValueError("NO_PUBLIC_DATA|Target previously confirmed to have no data.")
        
    logger.info(f"Searching MAST for real TESS data for TIC {tic_str}...")
    
    max_retries = 3
    search_result = None
    search_time = 0
    search_start = time.time()
    
    for attempt in range(max_retries):
        try:
            search_result = lk.search_lightcurve(f"TIC {tic_str}", mission="TESS")
            search_time = time.time() - search_start
            break
        except lk.search.SearchError as e:
            CACHE_STATS["retries"] += 1
            if attempt == max_retries - 1:
                CACHE_STATS["timeouts"] += 1
                CACHE_STATS["network_errors"] += 1
                logger.error(f"Network or MAST API failure for TIC {tic_str}: {str(e)}")
                raise ConnectionError(f"DOWNLOAD_TIMEOUT|MAST server did not respond after {max_retries} attempts.")
            time.sleep(2 ** (attempt + 1))
        except Exception as e:
            CACHE_STATS["retries"] += 1
            if attempt == max_retries - 1:
                CACHE_STATS["network_errors"] += 1
                raise ConnectionError(f"NETWORK_ERROR|Unexpected network error: {str(e)}")
            time.sleep(2 ** (attempt + 1))

    if search_result is None or len(search_result) == 0:
        UNAVAILABLE_TARGETS_CACHE.add(tic_str)
        raise ValueError(f"NO_PUBLIC_DATA|Products found: 0")
        
    logger.info(f"MAST Search completed in {search_time:.2f}s. Products found: {len(search_result)}")
    
    download_start = time.time()
    authors_priority = ["SPOC", "QLP", "TESS-SPOC", "CDIPS", "TASOC"]
    available_authors = set(search_result.author)
    logger.info(f"Authors available for TIC {tic_str}: {available_authors}")
    
    authors_to_try = [a for a in authors_priority if a in available_authors]
    authors_to_try.extend([a for a in available_authors if a not in authors_priority])
    
    lc = None
    selected_author = None
    author_result = None
    
    for author in authors_to_try:
        logger.info(f"--- Search Attempt: TIC {tic_str} | Author: {author} | Mission: TESS ---")
        current_author_result = search_result[search_result.author == author]
        
        for attempt in range(max_retries):
            try:
                lc = current_author_result[0].download()
                if lc is not None:
                    CACHE_STATS["success_downloads"] += 1
                    selected_author = author
                    author_result = current_author_result
                    logger.info(f"Successfully downloaded light curve from {author}.")
                    break
            except Exception as e:
                error_msg = str(e).lower()
                if "corrupt" in error_msg or "error in reading" in error_msg or "empty" in error_msg:
                    logger.warning(f"Cache corrupted for {author}. Attempting recovery...")
                    try:
                        from astropy.utils.data import clear_download_cache
                        url = current_author_result.table['dataURI'][0]
                        clear_download_cache(url)
                        CACHE_STATS["corrupted_deleted"] += 1
                        logger.info("Deleted corrupted cached FITS file. Re-downloading...")
                    except Exception as clear_e:
                        logger.warning(f"Recovery failed to clear cache: {str(clear_e)}")
                
                CACHE_STATS["retries"] += 1
                if attempt == max_retries - 1:
                    CACHE_STATS["failed_downloads"] += 1
                    logger.warning(f"Failed to download from {author} after {max_retries} attempts: {str(e)}")
                    lc = None
                else:
                    time.sleep(2 ** (attempt + 1))
        
        if lc is not None:
            break
            
    download_time = time.time() - download_start
    
    if lc is None:
        UNAVAILABLE_TARGETS_CACHE.add(tic_str)
        if any(a not in authors_priority for a in available_authors):
            # E.g. TARS only
            unsupported = [a for a in available_authors if a not in authors_priority]
            logger.warning(f"Failure: UNSUPPORTED_PRODUCT | Authors: {unsupported}")
            raise ValueError(f"UNSUPPORTED_PRODUCT|Author : {unsupported[0] if unsupported else 'Unknown'} | Missing required columns")
        else:
            raise ValueError(f"DOWNLOAD_FAILED|All download attempts failed or products corrupted.")

    # Quality filtering
    if not hasattr(lc, 'quality'):
        raise ValueError("NO_QUALITY_FLAGS|Light curve lacks QUALITY column.")
    if not hasattr(lc, 'flux_err'):
        raise ValueError("NO_FLUX_ERROR|Light curve lacks FLUX_ERR column.")

    mask = (lc.quality == 0)
    time_val = lc.time[mask].value
    flux_val = lc.flux[mask].value
    flux_err_val = lc.flux_err[mask].value
    
    if len(time_val) == 0:
        UNAVAILABLE_TARGETS_CACHE.add(tic_str)
        raise ValueError("EMPTY_LIGHTCURVE|No high-quality data points available after filtering.")
        
    median_flux = np.nanmedian(flux_val)
    if median_flux > 0:
        flux_val = flux_val / median_flux
        flux_err_val = flux_err_val / median_flux
        
    # Extract Metadata safely
    def safe_get_meta(col_name):
        try:
            return author_result[col_name][0] if col_name in author_result.columns else "Unavailable"
        except:
            return "Unavailable"

    metadata = {
        "mission": safe_get_meta('mission'),
        "author": safe_get_meta('author'),
        "sector": safe_get_meta('sequence_number'),
        "cadence": safe_get_meta('t_exptime'),
        "collection": safe_get_meta('obs_collection'),
        "observation_id": safe_get_meta('obs_id'),
        "product_type": safe_get_meta('dataproduct_type'),
        "provenance": safe_get_meta('provenance_name'),
        "filename": safe_get_meta('productFilename')
    }
    
    # Structured Logging
    logger.info("=" * 48)
    logger.info(f"Target          : {tic_str}")
    logger.info(f"Products Found  : {len(search_result)}")
    logger.info(f"Mission         : {metadata['mission']}")
    logger.info(f"Authors         : {', '.join(available_authors)}")
    logger.info(f"Selected Author : {selected_author}")
    logger.info(f"Sector          : {metadata['sector']}")
    logger.info(f"Cadence         : {metadata['cadence']} sec")
    logger.info(f"Download        : Success")
    logger.info(f"Elapsed         : {(search_time + download_time):.1f} sec")
    logger.info("=" * 48)

    return {
        "time": time_val,
        "flux": flux_val,
        "flux_err": flux_err_val,
        "target_name": f"TIC {tic_str}",
        "is_mock": False,
        "metadata": metadata,
        "timings": {
            "search": search_time,
            "download": download_time
        }
    }

def detect_csv_type(df: pd.DataFrame) -> Tuple[str, str]:
    """Detects if the dataframe is a light curve, tic_catalog, gaia_catalog, or mixed_catalog."""
    cols = [str(c).strip().lower() for c in df.columns]
    
    time_cols = {"time", "time_bjd", "time_btjd", "jd", "bjd", "mjd", "hjd", "date"}
    flux_cols = {"flux", "sap_flux", "pdcsap_flux", "normalized_flux", "brightness", "intensity"}
    mag_cols = {"mag", "magnitude", "vmag", "kmag", "tmag"}
    
    has_time_col = any(c in time_cols for c in cols)
    has_flux_col = any(c in flux_cols for c in cols)
    has_mag_col = any(c in mag_cols for c in cols)
    
    tic_cols = {"tic", "tic_id", "ticid", "toi", "target_id"}
    gaia_cols = {"gaia", "gaia_id", "gaia_source_id", "source_id", "gaia2"}
    
    has_tic = any(c in tic_cols for c in cols)
    has_gaia = any(c in gaia_cols for c in cols)
    
    # Check for MAST TIC bulk download columns (often ID is TIC ID)
    if "id" in cols and any(c in cols for c in ["tmag", "ra", "dec", "objtype"]):
        has_tic = True

    # 1. Light Curve Explicit
    if has_time_col and (has_flux_col or has_mag_col):
        id_col = next((c for c in cols if c in ["id", "tic", "tic_id", "toi", "target_id", "gaia", "gaia_id"]), None)
        if id_col and len(df[id_col].unique()) > 1:
            return "mixed_catalog", "Contains time series but has multiple object identifiers."
        return "lightcurve", "Detected as lightcurve because explicit time and flux/magnitude columns exist."
        
    # 2. Catalogs
    if has_tic and has_gaia:
        return "mixed_catalog", "Detected as mixed_catalog because both TIC and Gaia identifiers exist."
    if has_tic:
        return "tic_catalog", "Detected as tic_catalog because TIC identifiers exist."
    if has_gaia:
        return "gaia_catalog", "Detected as gaia_catalog because Gaia identifiers exist."
        
    # 3. Handle `id` specially
    if "id" in cols:
        id_series = pd.to_numeric(df["id"], errors='coerce').dropna()
        if len(id_series) > 0:
            diffs = id_series.diff().dropna()
            is_sequential = len(diffs) > 0 and (diffs == 1).all()
            if is_sequential:
                return "lightcurve", "Detected as lightcurve because 'id' column contains sequential row numbers."
            else:
                if id_series.mean() > 10000:
                    return "tic_catalog", "Detected as tic_catalog because 'id' column contains large integer identifiers."
                    
    # 4. No recognized headers fallback
    if len(df.columns) > 0:
        first_col = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna()
        if len(first_col) > 1:
            diffs = first_col.diff().dropna()
            is_increasing = len(diffs) > 0 and (diffs >= 0).all()
            is_decimal = (first_col % 1 != 0).any()
            
            if is_increasing and is_decimal:
                for i in range(1, len(df.columns)):
                    other_col = pd.to_numeric(df.iloc[:, i], errors='coerce').dropna()
                    if len(other_col) > 0:
                        mean_val = other_col.mean()
                        std_val = other_col.std()
                        if 0.5 < mean_val < 1.5 and std_val < 0.5:
                            return "lightcurve", "Detected as lightcurve based on first column and flux-like column."
                        elif 5 < mean_val < 30:
                            return "lightcurve", "Detected as lightcurve based on first column and magnitude-like column."
                            
            if not is_decimal and first_col.mean() > 10000:
                return "tic_catalog", "Detected as tic_catalog because first column contains large integer identifiers."
                
    return "unsupported", "Could not classify as lightcurve or catalog based on semantic rules."

def parse_lightcurve(df: pd.DataFrame, file_path: str) -> Dict[str, Any]:
    """Parses a light curve DataFrame with automatic dataset transformation."""
    
    time_cols = {"time", "time_bjd", "time_btjd", "jd", "bjd", "mjd", "hjd", "date"}
    flux_cols = {"flux", "sap_flux", "pdcsap_flux", "normalized_flux", "brightness", "intensity"}
    mag_cols = {"mag", "magnitude", "vmag", "kmag", "tmag"}
    err_cols = {"flux_err", "flux_error", "err", "error", "mag_err", "mag_error"}
    id_cols = {"id", "tic", "toi", "gaia", "object_id", "target_id", "star_id"}
    
    # Identify columns
    id_col = next((c for c in df.columns if str(c).strip().lower() in id_cols), None)
    time_col = next((c for c in df.columns if str(c).strip().lower() in time_cols), None)
    flux_col = next((c for c in df.columns if str(c).strip().lower() in flux_cols), None)
    mag_col = next((c for c in df.columns if str(c).strip().lower() in mag_cols), None)
    err_col = next((c for c in df.columns if str(c).strip().lower() in err_cols), None)
    
    # Case E: Group by ID if multiple objects exist
    if id_col is not None and len(df[id_col].unique()) > 1:
        most_common_id = df[id_col].value_counts().idxmax()
        df = df[df[id_col] == most_common_id].copy()
        logger.info(f"Mixed catalog detected. Filtered to most common object ID: {most_common_id}")
        
    # If standard columns not found, assume positions
    numeric_cols = [c for c in df.columns if pd.to_numeric(df[c], errors='coerce').notna().sum() > 10]
    
    if time_col is None:
        if len(numeric_cols) > 0: time_col = numeric_cols[0]
        else: raise ValueError("No numeric columns found for time.")
        
    if flux_col is None and mag_col is None:
        if len(numeric_cols) > 1:
            flux_col = numeric_cols[1] if numeric_cols[0] == time_col else numeric_cols[0]
        else: raise ValueError("No numeric columns found for flux.")
        
    # Coerce to numeric, replace infs with NaN
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    
    if flux_col: df[flux_col] = pd.to_numeric(df[flux_col], errors="coerce")
    if mag_col: df[mag_col] = pd.to_numeric(df[mag_col], errors="coerce")
    if err_col: df[err_col] = pd.to_numeric(df[err_col], errors="coerce")
    
    import numpy as np
    df = df.replace([np.inf, -np.inf], np.nan)
    
    time = df[time_col].values
    
    # Magnitude to Flux conversion
    if flux_col is None and mag_col is not None:
        mag = df[mag_col].values
        flux = 10 ** (-mag / 2.5)
        if err_col:
            mag_err = df[err_col].values
            flux_err = 0.92103 * flux * mag_err
        else:
            flux_err = np.full_like(flux, np.nan)
    else:
        flux = df[flux_col].values
        if err_col:
            flux_err = df[err_col].values
        else:
            flux_err = np.full_like(flux, np.nan)

    # Remove NaNs
    valid = ~(np.isnan(time) | np.isnan(flux))
    time, flux, flux_err = time[valid], flux[valid], flux_err[valid]
    
    if len(time) < 10:
        raise ValueError("Fewer than 10 usable observations remain after NaN removal.")
        
    # Sort strictly by time
    sort_idx = np.argsort(time)
    time, flux, flux_err = time[sort_idx], flux[sort_idx], flux_err[sort_idx]
    
    if len(time) < 10:
        raise ValueError("Fewer than 10 usable observations remain after sorting.")
    
    # Remove duplicate timestamps
    time, unique_idx = np.unique(time, return_index=True)
    flux = flux[unique_idx]
    flux_err = flux_err[unique_idx]
    
    if len(time) < 10:
        raise ValueError("Fewer than 10 usable observations remain after duplicate timestamp removal.")
    
    # Interpolate tiny gaps
    if len(time) > 10:
        s_flux = pd.Series(flux)
        s_flux = s_flux.interpolate(method='linear', limit=3)
        flux = s_flux.values
    
    if len(time) < 10:
        raise ValueError("Fewer than 10 usable observations remain after interpolation.")
        
    # Estimate missing flux_err
    if np.isnan(flux_err).all():
        flux_err = np.full_like(flux, np.nanstd(flux) * 0.05 if np.nanstd(flux) > 0 else 0.001)
    else:
        nan_err_mask = np.isnan(flux_err)
        if nan_err_mask.any():
            flux_err[nan_err_mask] = np.nanmedian(flux_err) if not np.isnan(np.nanmedian(flux_err)) else 0.001
            
    # Normalize flux
    med_flux = np.nanmedian(flux)
    if med_flux > 0:
        flux = flux / med_flux
        flux_err = flux_err / med_flux
    
    if len(time) < 10:
        raise ValueError("Fewer than 10 usable observations remain after median normalization.")
        
    # 5-sigma clipping
    mean_f = np.nanmean(flux)
    std_f = np.nanstd(flux)
    if std_f > 0:
        clip_mask = np.abs(flux - mean_f) < (5 * std_f)
        time, flux, flux_err = time[clip_mask], flux[clip_mask], flux_err[clip_mask]
        
    if len(time) < 10:
        raise ValueError("Fewer than 10 usable observations remain after preprocessing and clipping.")
        
    return {
        "type": "lightcurve",
        "time": time,
        "flux": flux,
        "flux_err": flux_err,
        "target_name": Path(file_path).stem,
        "is_mock": False
    }

def parse_catalog(df: pd.DataFrame, original_cols: List[str], catalog_type: str = "tic_catalog", headerless: bool = False, seen_tics: set = None) -> Dict[str, Any]:
    """Parses a catalog DataFrame, extracting valid TIC/Gaia IDs and preserving all rows."""
    if seen_tics is None:
        seen_tics = set()
        
    total_rows = len(df)
    valid_targets = 0
    invalid_rows = 0
    skipped_rows = 0
    
    cols = df.columns.tolist()
    
    tic_col = None
    gaia_col = None
    
    if headerless:
        # Infer from data values rather than column names
        for col in cols:
            sample = df[col].dropna().head(20)
            if sample.empty:
                continue
                
            try:
                numeric_sample = pd.to_numeric(sample, errors='coerce')
                if numeric_sample.notna().all():
                    mean_val = numeric_sample.mean()
                    # Gaia IDs are typically 18-19 digits long (> 1e17)
                    if mean_val > 1e17 and not gaia_col:
                        gaia_col = col
                    # TIC IDs are typically 8-10 digits long (between 1e6 and 1e12)
                    elif mean_val > 1e6 and mean_val < 1e12 and not tic_col:
                        # Ensure it is an integer-like identifier, not a floating timestamp
                        if (numeric_sample % 1 == 0).all():
                            tic_col = col
            except Exception:
                pass
                
        # Fallback for TIC if not found (usually the first column)
        if not tic_col and len(cols) > 0:
            col0 = pd.to_numeric(df.iloc[:, 0], errors='coerce')
            if col0.notna().sum() / len(col0) > 0.9:
                tic_col = cols[0]
    else:
        priority = ["tic", "ticid", "tic_id", "id", "objectid", "target_id"]
        for p in priority:
            if p in cols:
                tic_col = p
                break
                
        if not tic_col and catalog_type in ["tic_catalog", "mixed_catalog"]:
            if len(cols) > 0:
                first_col = df.iloc[:, 0]
                numeric_count = pd.to_numeric(first_col, errors='coerce').notna().sum()
                if len(first_col) > 0 and numeric_count / len(first_col) > 0.9:
                    tic_col = cols[0]
                    
        # Check for Gaia only if we didn't firmly identify a TIC column, or if it's explicitly a Gaia catalog
        if not tic_col or catalog_type == "gaia_catalog":
            for p in ["gaia_id", "gaia_source_id", "source_id", "gaia", "gaia2"]:
                if p in cols:
                    gaia_col = p
                    break
                    
        if not tic_col and not gaia_col:
            # Final fallback for purely headerless unlabeled gaia catalogs
            if len(cols) > 0:
                first_col = df.iloc[:, 0]
                numeric_count = pd.to_numeric(first_col, errors='coerce').notna().sum()
                if len(first_col) > 0 and numeric_count / len(first_col) > 0.9:
                    if catalog_type == "gaia_catalog":
                        gaia_col = cols[0]
                    else:
                        tic_col = cols[0]
                        
    primary_col = tic_col if tic_col else gaia_col
    is_gaia = not tic_col and gaia_col
    
    # 1. Clean primary_col
    s = df[primary_col].astype(str).str.strip()
    
    # Remove .0 if it's purely digits before it
    mask_dot_zero = s.str.endswith(".0") & s.str.slice(0, -2).str.isdigit()
    s.loc[mask_dot_zero] = s.loc[mask_dot_zero].str.slice(0, -2)
    
    # 2. Find invalid rows (empty, 'nan')
    is_empty_or_nan = (s == "") | (s.str.lower() == "nan") | (s.str.lower() == "<na>")
    
    # 3. Numeric conversion to find valid positive values
    s_numeric = pd.to_numeric(s, errors='coerce')
    is_valid_num = s_numeric.notna() & (s_numeric >= 0)
    
    # 4. If Gaia, string starting with "GAIA" is also valid
    is_valid_str = pd.Series(False, index=s.index)
    if is_gaia:
        is_valid_str = s.str.upper().str.startswith("GAIA")
        
    # Final valid mask
    valid_mask = (~is_empty_or_nan) & (is_valid_num | is_valid_str)
    
    invalid_rows = (~valid_mask).sum()
    
    # 5. Handle duplicates
    valid_df = df[valid_mask].copy()
    valid_s = s[valid_mask]
    
    is_duplicate_internal = valid_s.duplicated(keep='first')
    is_duplicate_external = valid_s.isin(seen_tics)
    is_duplicate = is_duplicate_internal | is_duplicate_external
    
    skipped_rows = is_duplicate.sum()
    
    valid_df = valid_df[~is_duplicate]
    valid_s = valid_s[~is_duplicate]
    valid_targets = len(valid_df)
    
    # Add new valid identifiers to the seen set
    seen_tics.update(valid_s)
    
    # 6. Extract top 500 for preview
    preview_limit = 500
    preview_df = valid_df.head(preview_limit).copy()
    preview_s = valid_s.head(preview_limit)
    
    if is_gaia:
        preview_df["id"] = preview_s.apply(lambda x: x if str(x).upper().startswith("GAIA") else f"GAIA {x}")
        preview_df["gaia_id"] = preview_df["id"]
        preview_targets = [{"tic": val} for val in preview_df["id"]]
    else:
        preview_df["id"] = preview_s
        preview_df["tic"] = preview_s
        preview_targets = [{"tic": str(val)} for val in preview_s]
        
    preview_objects = preview_df.to_dict(orient="records")
    
    return {
        "type": "catalog",
        "catalog_type": catalog_type,
        "targets": preview_targets,
        "objects": preview_objects,
        "columns": df.columns.tolist(),
        "total_rows": total_rows,
        "valid_targets": valid_targets,
        "invalid_rows": invalid_rows,
        "skipped_rows": skipped_rows,
        "preview_rows": len(preview_objects)
    }

def load_uploaded_file(file_path: str) -> Dict[str, Any]:
    """Reads a CSV file and processes it as either a light curve or a catalog."""
    try:
        # Intercept headerless light curves and catalogs
        try:
            df_test = pd.read_csv(file_path, nrows=50, comment='#', skipinitialspace=True, header=None)
            is_headerless_lc = False
            headerless_catalog = False
            if len(df_test.columns) >= 2:
                col0 = pd.to_numeric(df_test[0], errors='coerce')
                col1 = pd.to_numeric(df_test[1], errors='coerce')
                
                # Check if first col is numeric, second col is numeric
                if col0.notna().all() and col1.notna().all():
                    # Check if first col is monotonically increasing
                    diffs = col0.diff().dropna()
                    if len(diffs) > 0 and (diffs >= 0).all():
                        # Exclude obvious catalogs (TIC IDs are large integers)
                        if (col0 % 1 != 0).any() or col0.mean() < 10000:
                            is_headerless_lc = True
                            
                # Detect headerless catalogs
                if not is_headerless_lc:
                    # If the first value in the first row is a number, it's very likely data, not a column name string
                    first_val = pd.to_numeric(df_test.iloc[0, 0], errors='coerce')
                    if not pd.isna(first_val):
                        headerless_catalog = True
        except Exception:
            is_headerless_lc = False
            headerless_catalog = False
            
        file_size = os.path.getsize(file_path)
        is_large = file_size > 50 * 1024 * 1024  # 50 MB
        
        headerless = False
        def normalize_col(c):
            c = str(c).strip().lower()
            import re
            c = re.sub(r'[^a-z0-9_]', '_', c)
            c = re.sub(r'_+', '_', c)
            return c.strip('_')
            
        if is_large and not is_headerless_lc:
            # For massive files, stream chunk by chunk
            if headerless_catalog:
                chunk_iter = pd.read_csv(file_path, header=None, comment='#', skipinitialspace=True, chunksize=100000, low_memory=False)
                headerless = True
            else:
                chunk_iter = pd.read_csv(file_path, comment='#', skipinitialspace=True, chunksize=100000, low_memory=False)
                headerless = False
                
            total_rows = 0
            valid_targets = 0
            invalid_rows = 0
            skipped_rows = 0
            preview_objects = []
            preview_targets = []
            seen_tics = set()
            
            first_chunk = True
            csv_type = "catalog"
            original_cols = []
            
            for chunk in chunk_iter:
                if headerless:
                    chunk.columns = [f"col_{i}" for i in range(len(chunk.columns))]
                else:
                    chunk.columns = [normalize_col(c) for c in chunk.columns]
                    
                if first_chunk:
                    original_cols = chunk.columns.tolist()
                    csv_type, debug_reason = detect_csv_type(chunk)
                    
                    if csv_type == "lightcurve":
                        raise ValueError("Large files >50MB cannot be processed as lightcurves.")
                    
                    logger.info(f"Detected {csv_type.replace('_', ' ').title()} CSV (Chunked Mode)")
                    first_chunk = False
                    
                if all(c.startswith("column_") or c.isdigit() for c in chunk.columns):
                    if len(chunk.columns) >= 3:
                        chunk.rename(columns={chunk.columns[0]: 'tic', chunk.columns[1]: 'ra', chunk.columns[2]: 'dec'}, inplace=True)
                        
                res = parse_catalog(chunk, original_cols, catalog_type=csv_type, headerless=headerless, seen_tics=seen_tics)
                
                total_rows += res["total_rows"]
                valid_targets += res["valid_targets"]
                invalid_rows += res["invalid_rows"]
                skipped_rows += res["skipped_rows"]
                
                if len(preview_objects) < 500:
                    needed = 500 - len(preview_objects)
                    preview_targets.extend(res["targets"][:needed])
                    preview_objects.extend(res["objects"][:needed])
                    
            logger.info(f"Finished processing large {csv_type} in chunks. Total rows: {total_rows}")
            return {
                "type": "catalog",
                "catalog_type": csv_type,
                "targets": preview_targets,
                "objects": preview_objects,
                "columns": original_cols,
                "total_rows": total_rows,
                "valid_targets": valid_targets,
                "invalid_rows": invalid_rows,
                "skipped_rows": skipped_rows,
                "preview_rows": len(preview_objects)
            }
            
        if is_headerless_lc:
            df = pd.read_csv(file_path, comment='#', skipinitialspace=True, header=None)
            original_cols = df.columns.tolist()
            new_cols = []
            for i in range(len(df.columns)):
                if i == 0: new_cols.append("time")
                elif i == 1: new_cols.append("flux")
                elif i == 2: new_cols.append("flux_err")
                else: new_cols.append(f"col_{i}")
            df.columns = new_cols
        elif headerless_catalog:
            df = pd.read_csv(file_path, header=None, comment='#', skipinitialspace=True, low_memory=False)
            original_cols = df.columns.tolist()
            df.columns = [f"col_{i}" for i in range(len(df.columns))]
            headerless = True
        else:
            df = pd.read_csv(file_path, comment='#', skipinitialspace=True)
            original_cols = df.columns.tolist()
            
            df.columns = [normalize_col(c) for c in df.columns]
        
    except pd.errors.EmptyDataError:
        raise ValueError("Uploaded file is empty.")
    except Exception as e:
        # Fallback to manual parsing if pandas fails (e.g., bad headers or comments)
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            
        header_idx = 0
        known_keywords = {"tic", "toi", "gaia", "ra", "dec", "time", "flux", "target", "object", "sector", "magnitude"}
        
        for i, line in enumerate(lines[:100]):
            line_lower = line.lower()
            if not line.strip():
                continue
                
            # If the line contains typical data values (like lots of numbers), it's probably data
            import re
            numbers = len(re.findall(r'\b\d+\.?\d*\b', line))
            words = len(re.findall(r'[a-zA-Z]+', line))
            if numbers > words + 2 and not line.startswith('#'):
                # Data started, meaning header was not found or we passed it.
                break
                
            # Check for header keywords
            matched_kws = [kw for kw in known_keywords if kw in line_lower]
            if line_lower.strip().startswith('#'):
                if len(matched_kws) >= 2:
                    header_idx = i
                    break
            else:
                if len(matched_kws) >= 1:
                    header_idx = i
                    break
                    
        try:
            df = pd.read_csv(file_path, skiprows=header_idx, comment='#', skipinitialspace=True)
            original_cols = df.columns.tolist()
            headerless = False
        except Exception as inner_e:
            raise ValueError(f"Failed to parse CSV file: {str(inner_e)}")

    if df.empty or len(df.columns) == 0:
         raise ValueError("No columns to parse from file. Please ensure it is a valid text CSV.")

    is_tic = "tic_dec88" in file_path

    # --- TRANSPOSED CSV DETECTION ---
    if df.shape[0] <= 10 and df.shape[1] >= 100:
        first_col_vals = [str(df.columns[0]).strip().lower()] + df.iloc[:, 0].astype(str).str.strip().str.lower().tolist()
        valid_labels = {"time", "jd", "bjd", "btjd", "flux", "sap_flux", "pdcsap_flux", "magnitude", "mag", "flux_err", "error", "sigma"}
        
        if any(label in first_col_vals for label in valid_labels):
            orig_shape = df.shape
            
            # Reconstruct and transpose
            header_df = pd.DataFrame([df.columns])
            df.columns = range(df.shape[1])
            df_full = pd.concat([header_df, df], ignore_index=True)
            
            df = df_full.T
            
            # Assign new headers and remove the header row
            df.columns = df.iloc[0].astype(str).str.strip().str.lower()
            df = df[1:].reset_index(drop=True)
            
            time_col_name = next((c for c in df.columns if str(c) in {"time", "time_bjd", "time_btjd", "jd", "bjd", "mjd", "hjd", "date"}), None)
            
            # Convert numeric columns safely
            df = df.apply(pd.to_numeric, errors='ignore')
            
            # Now safely normalize columns AFTER reconstruction
            df.columns = [normalize_col(c) for c in df.columns]
            
            logger.info("[Ingestion Engine]")
            logger.info("Transposed astronomical dataset detected.")
            logger.info(f"Original shape: {orig_shape}")
            logger.info(f"New shape: {df.shape}")
        else:
            df.columns = [normalize_col(c) for c in df.columns]
    else:
        df.columns = [normalize_col(c) for c in df.columns]
        
    csv_type, debug_reason = detect_csv_type(df)
        
    logger.info(f"Detector reasoning: {debug_reason}")
    
    if csv_type == "lightcurve":
        logger.info("Detected Light Curve CSV")
        return parse_lightcurve(df, file_path)
    elif csv_type in ["tic_catalog", "gaia_catalog", "mixed_catalog", "catalog"]:
        logger.info(f"Detected {csv_type.replace('_', ' ').title()} CSV")
        if all(c.startswith("column_") or c.isdigit() for c in df.columns):
            logger.info("Detected headerless MAST catalog format, injecting explicit column names.")
            # Map known MAST columns if possible
            if len(df.columns) >= 3:
                df.rename(columns={df.columns[0]: 'tic', df.columns[1]: 'ra', df.columns[2]: 'dec'}, inplace=True)
        return parse_catalog(df, original_cols, catalog_type=csv_type, headerless=headerless)
    else:
        raise ValueError(f"Unsupported CSV format. {debug_reason}")
