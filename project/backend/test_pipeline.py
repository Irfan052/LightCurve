import unittest
import numpy as np
import os
import sys
from pathlib import Path

# Add project root to sys.path so we can import backend packages
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.data_loader import generate_synthetic_lightcurve, load_tess_lightcurve
from backend.quality_filter import clean_lightcurve
from backend.detrend import flatten_lightcurve
from backend.transit_search import search_transits
from backend.phase_fold import fold_lightcurve, bin_folded_lightcurve
from backend.feature_engineering import extract_features
from backend.classifier import train_classifier, predict_class, FEATURE_KEYS
from backend.parameter_fit import estimate_parameters

class TestExoplanetPipeline(unittest.TestCase):
    
    def test_synthetic_data_generation(self):
        """Tests that the mock data generator returns correctly shaped and typed arrays."""
        for target in ["exoplanet_transit", "eclipsing_binary", "stellar_variability", "instrumental_artifact"]:
            time, flux, flux_err = generate_synthetic_lightcurve(target, length_days=5.0, cadence_minutes=30.0)
            self.assertEqual(len(time), len(flux))
            self.assertEqual(len(flux), len(flux_err))
            self.assertTrue(len(time) > 0)
            self.assertTrue(np.all(flux > 0))
            
    def test_cleaning_and_detrending(self):
        """Tests quality filtering (sigma clipping) and flattening."""
        time, flux, flux_err = generate_synthetic_lightcurve("exoplanet_transit", length_days=5.0, cadence_minutes=30.0)
        
        # Inject NaNs to test quality cleaning
        flux[5] = np.nan
        flux_err[10] = np.inf
        
        t_clean, f_clean, fe_clean = clean_lightcurve(time, flux, flux_err)
        # Ensure NaNs and infs are removed
        self.assertFalse(np.any(np.isnan(t_clean)))
        self.assertFalse(np.any(np.isnan(f_clean)))
        self.assertFalse(np.any(np.isinf(fe_clean)))
        
        # Test flattening
        flat_flux, trend_flux = flatten_lightcurve(t_clean, f_clean, window_days=0.5)
        self.assertEqual(len(flat_flux), len(t_clean))
        self.assertEqual(len(trend_flux), len(t_clean))
        # Flattened flux should be closely normalized to 1.0
        self.assertAlmostEqual(np.nanmedian(flat_flux), 1.0, places=2)
        
    def test_transit_search_and_folding(self):
        """Tests Box Least Squares (BLS) search and phase folding."""
        # Generate clean exoplanet transit
        time, flux, flux_err = generate_synthetic_lightcurve("exoplanet_transit", length_days=10.0, cadence_minutes=15.0)
        t_clean, f_clean, fe_clean = clean_lightcurve(time, flux, flux_err)
        flat_flux, _ = flatten_lightcurve(t_clean, f_clean)
        
        # Search transits
        bls_res = search_transits(t_clean, flat_flux, fe_clean)
        self.assertIn("period", bls_res)
        self.assertGreater(bls_res["period"], 0)
        self.assertGreater(bls_res["snr"], 0)
        
        # Fold lightcurve
        folded = fold_lightcurve(t_clean, flat_flux, fe_clean, bls_res["period"], bls_res["epoch"])
        self.assertEqual(len(folded["phase"]), len(t_clean))
        self.assertTrue(np.all(folded["phase"] >= -0.5) and np.all(folded["phase"] <= 0.5))
        
        # Bin folded lightcurve
        bin_centers, bin_flux, bin_err = bin_folded_lightcurve(folded["phase"], folded["flux"], num_bins=50)
        self.assertEqual(len(bin_centers), 50)
        self.assertEqual(len(bin_flux), 50)
        self.assertEqual(len(bin_err), 50)
        
    def test_feature_engineering_and_classification(self):
        """Tests feature extraction and Random Forest classifier training/inference."""
        # Ensure model is trained (or retrain for testing)
        rf = train_classifier()
        self.assertIsNotNone(rf)
        
        # Run pipeline on a mock target to get features
        time, flux, flux_err = generate_synthetic_lightcurve("exoplanet_transit", length_days=10.0, cadence_minutes=15.0)
        t_c, f_c, fe_c = clean_lightcurve(time, flux, flux_err)
        flat_flux, _ = flatten_lightcurve(t_c, f_c)
        bls_res = search_transits(t_c, flat_flux, fe_c)
        
        feats = extract_features(t_c, flat_flux, fe_c, bls_res)
        
        # Ensure all feature keys exist
        for key in FEATURE_KEYS:
            self.assertIn(key, feats)
            self.assertFalse(np.isnan(feats[key]))
            
        # Classify
        pred = predict_class(feats)
        self.assertIn("prediction_label", pred)
        self.assertIn("confidence", pred)
        self.assertIn("probabilities", pred)
        self.assertTrue(0.0 <= pred["confidence"] <= 1.0)
        self.assertIn(pred["prediction_label"], pred["probabilities"])
        
        # Parameter estimation
        params = estimate_parameters(bls_res, pred)
        self.assertIn("planet_radius_earth", params)
        self.assertIn("semi_major_axis_au", params)
        self.assertIn("confidence_score", params)
        self.assertTrue(0.0 <= params["confidence_score"] <= 1.0)

if __name__ == "__main__":
    unittest.main()
