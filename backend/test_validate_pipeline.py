import unittest
import os
import json
import pandas as pd
from unittest.mock import patch, MagicMock

from backend.validate_pipeline import run_pipeline_for_target, validate_pipeline

class TestValidatePipeline(unittest.TestCase):
    
    @patch("backend.validate_pipeline.load_tess_lightcurve")
    @patch("backend.validate_pipeline.clean_lightcurve")
    @patch("backend.validate_pipeline.flatten_lightcurve")
    @patch("backend.validate_pipeline.search_transits")
    @patch("backend.validate_pipeline.extract_features")
    @patch("backend.validate_pipeline.predict_class")
    def test_run_pipeline_for_target_success(self, mock_predict, mock_extract, mock_search, mock_flatten, mock_clean, mock_load):
        # Setup mocks
        mock_load.return_value = {"time": [1, 2, 3], "flux": [1, 1, 1], "flux_err": [0.01, 0.01, 0.01]}
        mock_clean.return_value = ([1, 2, 3], [1, 1, 1], [0.01, 0.01, 0.01])
        mock_flatten.return_value = ([1, 1, 1], [1, 1, 1])
        mock_search.return_value = {"period": 2.0, "depth": 0.05, "duration": 0.1}
        mock_extract.return_value = {"feat1": 1.0}
        mock_predict.return_value = {"prediction_label": "exoplanet_transit", "probabilities": {"exoplanet_transit": 0.9}, "confidence": 0.9}
        
        result = run_pipeline_for_target("TIC 123")
        
        self.assertNotIn("error", result)
        self.assertEqual(result["period"], 2.0)
        self.assertEqual(result["depth"], 0.05)
        self.assertEqual(result["duration"], 0.1)
        self.assertEqual(result["prediction_label"], "exoplanet_transit")
        self.assertEqual(result["confidence"], 0.9)
        
    @patch("backend.validate_pipeline.load_tess_lightcurve")
    def test_run_pipeline_for_target_load_error(self, mock_load):
        mock_load.return_value = {"error": "Failed to load"}
        
        result = run_pipeline_for_target("TIC 123")
        
        self.assertIn("error", result)
        self.assertEqual(result["error"], "Failed to load")
        
    @patch("backend.validate_pipeline.load_tess_lightcurve")
    @patch("backend.validate_pipeline.clean_lightcurve")
    @patch("backend.validate_pipeline.flatten_lightcurve")
    @patch("backend.validate_pipeline.search_transits")
    def test_run_pipeline_for_target_search_error(self, mock_search, mock_flatten, mock_clean, mock_load):
        mock_load.return_value = {"time": [1, 2, 3], "flux": [1, 1, 1], "flux_err": [0.01, 0.01, 0.01]}
        mock_clean.return_value = ([1, 2, 3], [1, 1, 1], [0.01, 0.01, 0.01])
        mock_flatten.return_value = ([1, 1, 1], [1, 1, 1])
        mock_search.return_value = {"error": "BLS failed"}
        
        result = run_pipeline_for_target("TIC 123")
        
        self.assertIn("error", result)
        self.assertEqual(result["error"], "BLS failed")
        
    @patch("backend.validate_pipeline.train_classifier")
    @patch("backend.validate_pipeline.run_pipeline_for_target")
    def test_validate_pipeline(self, mock_run, mock_train):
        # Create a dummy catalog
        catalog_path = "test_catalog.csv"
        report_path = "test_report.json"
        summary_path = "test_summary.csv"
        
        df = pd.DataFrame({
            "tic_id": ["1", "2"],
            "true_period": [2.0, 3.0],
            "true_depth": [0.05, 0.01],
            "true_duration": [0.1, 0.2],
            "class_label": ["exoplanet_transit", "eclipsing_binary"]
        })
        df.to_csv(catalog_path, index=False)
        
        # Setup mocks
        mock_run.side_effect = [
            {
                "period": 2.1,
                "depth": 0.04,
                "duration": 0.11,
                "prediction_label": "exoplanet_transit",
                "probabilities": {},
                "confidence": 0.95
            },
            {
                "period": 3.0,
                "depth": 0.01,
                "duration": 0.2,
                "prediction_label": "eclipsing_binary",
                "probabilities": {},
                "confidence": 0.85
            }
        ]
        
        report = validate_pipeline(catalog_path, report_path, summary_path)
        
        self.assertTrue(os.path.exists(report_path))
        self.assertTrue(os.path.exists(summary_path))
        
        self.assertEqual(report["total_targets"], 2)
        self.assertEqual(report["successful_targets"], 2)
        self.assertEqual(report["failed_targets"], 0)
        
        metrics = report["metrics"]
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1_score"], 1.0)
        
        # errors for target 1
        self.assertAlmostEqual(report["results"][0]["period_error"], 0.1, places=5)
        self.assertAlmostEqual(report["results"][0]["depth_error"], 0.01, places=5)
        self.assertAlmostEqual(report["results"][0]["duration_error"], 0.01, places=5)
        
        # Cleanup
        os.remove(catalog_path)
        os.remove(report_path)
        os.remove(summary_path)

if __name__ == "__main__":
    unittest.main()
