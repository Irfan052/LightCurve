import unittest
import numpy as np
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.quality_filter import (
    clean_lightcurve,
    remove_nans_and_inf,
    mask_tess_quality_flags,
    sigma_clip,
    detect_observation_gaps,
    CleanedLightCurve
)

class TestQualityFilterUpgrades(unittest.TestCase):
    
    def setUp(self):
        # Create standard test data
        self.time = np.linspace(0, 10, 1000)
        self.flux = np.ones_like(self.time) + np.random.normal(0, 0.001, len(self.time))
        self.flux_err = np.full_like(self.time, 0.001)
        self.quality = np.zeros(len(self.time), dtype=int)
        
    def test_cleaned_light_curve_unpacking(self):
        """Verify CleanedLightCurve acts like a 3-tuple and supports attribute access."""
        metrics = {"test_metric": 42}
        cl = CleanedLightCurve(self.time, self.flux, self.flux_err, metrics)
        
        # Test unpacking
        t, f, fe = cl
        np.testing.assert_array_equal(t, self.time)
        np.testing.assert_array_equal(f, self.flux)
        np.testing.assert_array_equal(fe, self.flux_err)
        
        # Test length and indexing
        self.assertEqual(len(cl), 3)
        np.testing.assert_array_equal(cl[0], self.time)
        
        # Test metrics access
        self.assertEqual(cl.metrics["test_metric"], 42)

    def test_remove_nans_and_inf_negative_flux(self):
        """Test safe removal of NaNs, Infs, but preservation of negative fluxes (no flux > 0 requirement)."""
        time_bad = self.time.copy()
        flux_bad = self.flux.copy()
        flux_err_bad = self.flux_err.copy()
        quality_bad = self.quality.copy()
        
        # Inject bad values
        flux_bad[10] = np.nan
        time_bad[20] = np.inf
        flux_err_bad[30] = -np.inf
        flux_bad[40] = -0.5  # Non-positive flux (should be retained now!)
        
        t, f, fe, q, removed = remove_nans_and_inf(time_bad, flux_bad, flux_err_bad, quality_bad)
        
        self.assertEqual(removed, 3)  # Only 3 removed (nan, inf, inf), NOT 4
        self.assertEqual(len(t), len(self.time) - 3)
        self.assertIn(-0.5, f)  # -0.5 is preserved
        self.assertFalse(np.any(np.isnan(f)))
        self.assertFalse(np.any(np.isinf(t)))
        self.assertEqual(len(q), len(t))

    def test_mask_tess_quality_flags_strict(self):
        """Test quality flag masking in both strict and bitmask modes."""
        quality_flags = self.quality.copy()
        quality_flags[50] = 1   # Attitude tweak (removed in both)
        quality_flags[100] = 256 # Stray light (retained in bitmask 17, removed in strict)
        
        # Test bitmask 17 (strict=False)
        t_bm, _, _, removed_bm, breakdown_bm = mask_tess_quality_flags(
            self.time, self.flux, self.flux_err, quality_flags, bitmask=17, strict_quality_mode=False
        )
        self.assertEqual(removed_bm, 1)
        self.assertEqual(len(t_bm), len(self.time) - 1)
        self.assertIn("Attitude tweak", breakdown_bm)
        self.assertNotIn("Stray light anomaly", breakdown_bm)
        
        # Test strict mode (strict=True)
        t_str, _, _, removed_str, breakdown_str = mask_tess_quality_flags(
            self.time, self.flux, self.flux_err, quality_flags, bitmask=17, strict_quality_mode=True
        )
        self.assertEqual(removed_str, 2)
        self.assertEqual(len(t_str), len(self.time) - 2)
        self.assertIn("Attitude tweak", breakdown_str)
        self.assertIn("Stray light anomaly", breakdown_str)

    def test_sigma_clip(self):
        """Test asymmetric outlier removal."""
        flux_outliers = self.flux.copy()
        
        # Inject one positive cosmic ray (large outlier) and one negative transit-like dip
        flux_outliers[100] = 2.0  # Big upper outlier
        flux_outliers[200] = 0.99 # Lower dip (1% dip, standard std is 0.001, so this is 10-sigma)
        
        # Clip with standard settings (upper=3.0, lower=15.0)
        t, f, fe, removed = sigma_clip(self.time, flux_outliers, self.flux_err, sigma_upper=3.0, sigma_lower=15.0)
        
        self.assertEqual(removed, 1)
        self.assertNotIn(2.0, f)
        self.assertIn(0.99, f)

    def test_detect_observation_gaps(self):
        """Test detection of gaps, duty cycle calculations, and gap indices."""
        time_gaps = self.time.copy()
        # Shift all timestamps after index 500 by 1.2 days
        time_gaps[500:] += 1.2
        
        stats = detect_observation_gaps(time_gaps, gap_threshold_days=0.5)
        
        self.assertEqual(stats["gap_count"], 1)
        self.assertAlmostEqual(stats["max_gap_days"], 1.2, delta=0.02)
        self.assertAlmostEqual(stats["total_gap_days"], 1.2, delta=0.02)
        
        # Gap indices check
        self.assertEqual(stats["gap_indices"], [499])
        
        # Duty cycle check
        expected_duty = (time_gaps[-1] - time_gaps[0] - stats["total_gap_days"]) / (time_gaps[-1] - time_gaps[0])
        self.assertAlmostEqual(stats["duty_cycle"], expected_duty, places=5)
        self.assertEqual(len(stats["gap_intervals"]), 1)
        self.assertAlmostEqual(stats["gap_intervals"][0]["duration"], 1.2, delta=0.02)
        self.assertEqual(stats["gap_intervals"][0]["pre_gap_index"], 499)

    def test_clean_lightcurve_end_to_end(self):
        """Verify the full clean_lightcurve pipeline returns metrics and maintains unpacking."""
        # Make test data with NaNs, quality flags, and gaps
        time_raw = self.time.copy()
        time_raw[500:] += 0.8  # Gap
        
        flux_raw = self.flux.copy()
        flux_raw[10] = np.nan  # NaN
        flux_raw[100] = 1.5    # Cosmic ray
        
        flux_err_raw = self.flux_err.copy()
        
        quality_raw = self.quality.copy()
        quality_raw[200] = 16  # Momentum dump
        
        # Run end-to-end cleaning
        cl = clean_lightcurve(
            time_raw, flux_raw, flux_err_raw, 
            quality=quality_raw, 
            quality_bitmask=16,
            sigma_upper=3.0,
            sigma_lower=5.0,
            strict_quality_mode=False
        )
        
        # Ensure we can unpack the output
        t, f, fe = cl
        self.assertEqual(len(t), len(f))
        
        # Verify new metrics
        self.assertTrue(cl.metrics["quality_array_available"])
        self.assertEqual(cl.metrics["quality_points_retained"], len(self.time) - 2) # 1 nan, 1 flag
        self.assertEqual(cl.metrics["gap_indices"], [498]) # index shifted by nan removal
        
        # Verify standard metrics
        self.assertEqual(cl.metrics["nan_inf_removed"], 1)
        self.assertEqual(cl.metrics["quality_flag_removed"], 1)
        self.assertEqual(cl.metrics["sigma_clipped_removed"], 1)
        self.assertEqual(cl.metrics["gap_count"], 1)
        self.assertAlmostEqual(cl.metrics["max_gap_days"], 0.8, delta=0.02)

    def test_error_handling(self):
        """Test array length validation and boundary conditions."""
        t, f, fe = np.array([]), np.array([]), np.array([])
        cl = clean_lightcurve(t, f, fe)
        self.assertEqual(len(cl[0]), 0)
        self.assertEqual(cl.metrics["total_raw_points"], 0)
        
        # Mismatched length
        with self.assertRaises(ValueError):
            clean_lightcurve(self.time, self.flux[:100], self.flux_err)

if __name__ == "__main__":
    unittest.main()
