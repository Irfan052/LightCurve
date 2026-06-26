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
        return {"time": time_data, "flux": flux, "flux_err": flux_err, "target_name": "TIC 9991 (Mock Exoplanet)", "is_mock": True}
    elif tic_str == "9992":
        logger.info("Generating mock Eclipsing Binary light curve...")
        time_data, flux, flux_err = generate_synthetic_lightcurve("eclipsing_binary")
        return {"time": time_data, "flux": flux, "flux_err": flux_err, "target_name": "TIC 9992 (Mock Eclipsing Binary)", "is_mock": True}
    elif tic_str == "9993":
        logger.info("Generating mock Stellar Variability light curve...")
        time_data, flux, flux_err = generate_synthetic_lightcurve("stellar_variability")
        return {"time": time_data, "flux": flux, "flux_err": flux_err, "target_name": "TIC 9993 (Mock Stellar Var)", "is_mock": True}
    elif tic_str == "9994":
        logger.info("Generating mock Instrumental Artifact light curve...")
        time_data, flux, flux_err = generate_synthetic_lightcurve("instrumental_artifact")
        return {"time": time_data, "flux": flux, "flux_err": flux_err, "target_name": "TIC 9994 (Mock Artifact)", "is_mock": True}
        
    logger.info(f"Searching MAST for real TESS data for TIC {tic_str}...")
    try:
        authors = ["SPOC", "QLP", "TESS-SPOC", "CDIPS", "TASOC"]
        lc = None
        
        for author in authors:
            logger.info(f"--- Search Attempt: TIC {tic_str} | Author: {author} | Mission: TESS ---")
            search_result = lk.search_lightcurve(f"TIC {tic_str}", mission="TESS", author=author)
            logger.info(f"SearchResult: {search_result}")
            logger.info(f"Products found: {len(search_result)}")
            
            if len(search_result) > 0:
                logger.info(f"Downloading light curve for TIC {tic_str} from author {author}...")
                try:
                    lc = search_result[0].download()
                    if lc is not None:
                        logger.info(f"Successfully downloaded light curve from {author}.")
                        break
                except Exception as e:
                    logger.warning(f"Failed to download from {author}: {str(e)}")
                    lc = None
                    
        if lc is None:
            raise ValueError(f"Light curve not available for TIC {tic_str}.")
        
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

def detect_csv_type(df: pd.DataFrame) -> Tuple[str, str]:
    """Detects if the dataframe is a light curve or a catalog based on semantic rules."""
    cols = set([str(c).strip().lower() for c in df.columns])
    
    time_cols = {"time", "time_bjd", "time_btjd", "jd", "bjd", "mjd", "hjd", "date"}
    flux_cols = {"flux", "sap_flux", "pdcsap_flux", "normalized_flux", "brightness", "intensity"}
    mag_cols = {"mag", "magnitude", "vmag", "kmag", "tmag"}
    
    has_time_col = next((c for c in df.columns if str(c).strip().lower() in time_cols), None)
    has_flux_col = next((c for c in df.columns if str(c).strip().lower() in flux_cols), None)
    has_mag_col = next((c for c in df.columns if str(c).strip().lower() in mag_cols), None)
    
    # 1. LIGHT CURVE explicit
    if has_time_col and (has_flux_col or has_mag_col):
        return "lightcurve", "Detected as lightcurve because explicit time and flux/magnitude columns exist."
                
    # 2. CATALOG explicit
    catalog_cols = {"tic", "tic_id", "ticid", "toi", "target_id", "gaia_id", "gaia_source_id", "source_id", "objectid", "gaia", "gaia2"}
    matched_catalog = next((c for c in df.columns if str(c).strip().lower() in catalog_cols), None)
    
    # Handle id mixed with light curve (Case E)
    id_col = next((c for c in df.columns if str(c).strip().lower() == "id"), None)
    if id_col and has_time_col and (has_flux_col or has_mag_col):
        return "lightcurve", "Detected as lightcurve despite ID column due to presence of time and flux."
        
    if matched_catalog:
        return "catalog", f"Detected as catalog because column '{matched_catalog}' exists."
        
    # 3. Handle `id` specially
    if id_col:
        id_series = pd.to_numeric(df[id_col], errors='coerce').dropna()
        if len(id_series) > 0:
            diffs = id_series.diff().dropna()
            is_sequential = len(diffs) > 0 and (diffs == 1).all()
            if is_sequential:
                return "lightcurve", "Detected as lightcurve because 'id' column contains sequential row numbers."
            else:
                if id_series.mean() > 10000:
                    return "catalog", "Detected as catalog because 'id' column contains large integer identifiers."
                    
    # 4. No recognized headers
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
                            return "lightcurve", "Detected as lightcurve because first column is mostly increasing decimal values and another numeric column behaves like normalized flux (~1.0)."
                        elif 5 < mean_val < 30:
                            return "lightcurve", "Detected as lightcurve because another numeric column behaves like magnitude."
                            
            if not is_decimal and first_col.mean() > 10000:
                return "catalog", "Detected as catalog because first column contains large integer identifiers."
                
    # Fallback to lightcurve if at least two numeric columns exist
    numeric_cols = [c for c in df.columns if pd.to_numeric(df[c], errors='coerce').notna().sum() > 10]
    if len(numeric_cols) >= 2:
        return "lightcurve", "Fallback to lightcurve salvage attempt."
        
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
    
    # Remove duplicate timestamps
    time, unique_idx = np.unique(time, return_index=True)
    flux = flux[unique_idx]
    flux_err = flux_err[unique_idx]
    
    # Interpolate tiny gaps
    if len(time) > 10:
        s_flux = pd.Series(flux)
        s_flux = s_flux.interpolate(method='linear', limit=3)
        flux = s_flux.values
        
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

def parse_catalog(df: pd.DataFrame, original_cols: List[str]) -> Dict[str, Any]:
    """Parses a catalog DataFrame, extracting valid TIC IDs."""
    total_rows = len(df)
    valid_targets = 0
    invalid_rows = 0
    skipped_rows = 0
    
    cols = df.columns.tolist()
    
    # Check for Gaia first to preserve Gaia-only catalog detection
    gaia_col = None
    for p in ["gaia_id", "gaia_source_id", "source_id", "gaia", "gaia2"]:
        if p in cols:
            gaia_col = p
            break
            
    tic_col = None
    priority = ["tic", "ticid", "tic_id", "id", "objectid", "target_id"]
    for p in priority:
        if p in cols:
            tic_col = p
            break
            
    if not tic_col and not gaia_col:
        if len(cols) > 0:
            first_col = df.iloc[:, 0]
            numeric_count = pd.to_numeric(first_col, errors='coerce').notna().sum()
            if len(first_col) > 0 and numeric_count / len(first_col) > 0.9:
                tic_col = cols[0]
                
    if not tic_col and not gaia_col:
        raise ValueError("Catalog detected but no astronomical identifier could be found.")
        
    targets = []
    objects = []
    seen_tics = set()
    
    primary_col = tic_col if tic_col else gaia_col
    is_gaia = not tic_col and gaia_col
    
    for _, row in df.iterrows():
        val = row[primary_col]
        
        if pd.isna(val):
            invalid_rows += 1
            continue
            
        val_str = str(val).strip()
        if not val_str or val_str.lower() == 'nan':
            invalid_rows += 1
            continue
            
        if val_str.endswith(".0") and val_str[:-2].isdigit():
            val_str = val_str[:-2]
            
        try:
            val_num = float(val_str)
            if val_num < 0:
                invalid_rows += 1
                continue
        except ValueError:
            if is_gaia and val_str.upper().startswith("GAIA"):
                pass
            else:
                invalid_rows += 1
                continue
            
        if val_str in seen_tics:
            skipped_rows += 1
            continue
            
        seen_tics.add(val_str)
        valid_targets += 1
        
        if is_gaia:
            gaia_id = val_str if val_str.upper().startswith("GAIA") else f"GAIA {val_str}"
            objects.append({"id": gaia_id, "gaia_id": gaia_id})
        else:
            targets.append({"tic": val_str})
            objects.append({"id": val_str, "tic": val_str})
            
    return {
        "type": "catalog",
        "targets": targets,
        "objects": objects,
        "columns": ["gaia_id"] if is_gaia else ["tic"],
        "total_rows": total_rows,
        "valid_targets": valid_targets,
        "invalid_rows": invalid_rows,
        "skipped_rows": skipped_rows
    }

def load_uploaded_file(file_path: str) -> Dict[str, Any]:
    """Reads a CSV file and processes it as either a light curve or a catalog."""
    try:
        # Intercept headerless light curves
        try:
            df_test = pd.read_csv(file_path, nrows=50, comment='#', skipinitialspace=True, header=None)
            is_headerless_lc = False
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
        except Exception:
            is_headerless_lc = False
            
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
        else:
            df = pd.read_csv(file_path, comment='#', skipinitialspace=True)
            original_cols = df.columns.tolist()
            
            def normalize_col(c):
                c = str(c).strip().lower()
                import re
                c = re.sub(r'[^a-z0-9_]', '_', c)
                c = re.sub(r'_+', '_', c)
                return c.strip('_')
                
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
            df.columns = [normalize_col(c) for c in df.columns]
        except Exception as inner_e:
            raise ValueError(f"Failed to parse CSV file: {str(inner_e)}")

    if df.empty or len(df.columns) == 0:
         raise ValueError("No columns to parse from file. Please ensure it is a valid text CSV.")

    csv_type, debug_reason = detect_csv_type(df)
    logger.info(f"Detector reasoning: {debug_reason}")
    
    if csv_type == "lightcurve":
        logger.info("Detected Light Curve CSV")
        return parse_lightcurve(df, file_path)
    elif csv_type == "catalog":
        logger.info("Detected TESS/Gaia Catalog CSV")
        if all(c.startswith("column_") or c.isdigit() for c in df.columns):
            logger.info("Detected headerless MAST catalog format, injecting explicit column names.")
            # Map known MAST columns if possible
            if len(df.columns) >= 3:
                df.rename(columns={df.columns[0]: 'tic', df.columns[1]: 'ra', df.columns[2]: 'dec'}, inplace=True)
        return parse_catalog(df, original_cols)
    else:
        raise ValueError(f"Unsupported CSV format. {debug_reason}")
