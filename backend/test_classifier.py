import json
import unittest
from pathlib import Path
import numpy as np
from unittest.mock import patch
import joblib

from backend.config import RANDOM_STATE
from backend.classifier import (
    FEATURE_KEYS,
    LEGACY_FEATURE_KEYS,
    MORPHOLOGY_FEATURE_KEYS,
    REPORT_SCHEMA_VERSION,
    SELECTION_CRITERION,
    _features_to_vector,
    _validate_feature_vector,
    morphology_features_included,
    _create_random_forest,
    _create_xgboost_classifier,
    evaluate_classifier,
    select_best_model,
    extract_feature_importance,
    build_classifier_report,
    save_classifier_report,
    build_training_dataset,
    train_classifier,
    predict_class,
    load_or_train_model,
    _expected_feature_count
)

class TestClassifierModule(unittest.TestCase):
    """Comprehensive unit tests for Phase 2E classifier enhancements."""

    def test_legacy_feature_keys_preserved_at_front(self):
        self.assertEqual(FEATURE_KEYS[: len(LEGACY_FEATURE_KEYS)], LEGACY_FEATURE_KEYS)

    def test_morphology_features_included_in_feature_keys(self):
        for key in MORPHOLOGY_FEATURE_KEYS:
            self.assertIn(key, FEATURE_KEYS)
        self.assertTrue(morphology_features_included())

    def test_feature_keys_are_unique(self):
        self.assertEqual(len(FEATURE_KEYS), len(set(FEATURE_KEYS)))

    def test_features_to_vector_length_and_defaults(self):
        vector = _features_to_vector({"period": 1.5, "depth": 0.01})
        self.assertEqual(len(vector), len(FEATURE_KEYS))
        self.assertAlmostEqual(vector[0], 1.5)
        self.assertAlmostEqual(vector[1], 0.01)
        self.assertEqual(vector[2], 0.0)

    def test_validate_feature_vector_rejects_nan(self):
        self.assertFalse(_validate_feature_vector([1.0, float("nan")]))
        self.assertTrue(_validate_feature_vector([1.0, 2.0, 3.0]))

    def test_evaluate_classifier_returns_required_metrics(self):
        rng = np.random.default_rng(RANDOM_STATE)
        X = rng.normal(size=(80, len(FEATURE_KEYS)))
        y = rng.integers(0, 4, size=80)
        model = _create_random_forest()
        model.fit(X, y)
        metrics = evaluate_classifier(model, X, y)
        for key in ("accuracy", "precision", "recall", "f1_score"):
            self.assertIn(key, metrics)
            self.assertGreaterEqual(metrics[key], 0.0)
            self.assertLessEqual(metrics[key], 1.0)

    def test_select_best_model_prefers_higher_f1(self):
        metrics = {
            "random_forest": {
                "accuracy": 0.80,
                "precision": 0.79,
                "recall": 0.78,
                "f1_score": 0.77,
            },
            "xgboost": {
                "accuracy": 0.75,
                "precision": 0.74,
                "recall": 0.73,
                "f1_score": 0.85,
            },
        }
        self.assertEqual(select_best_model(metrics), "xgboost")

    def test_select_best_model_tiebreaks_on_accuracy(self):
        metrics = {
            "random_forest": {
                "accuracy": 0.90,
                "precision": 0.80,
                "recall": 0.80,
                "f1_score": 0.80,
            },
            "xgboost": {
                "accuracy": 0.85,
                "precision": 0.80,
                "recall": 0.80,
                "f1_score": 0.80,
            },
        }
        self.assertEqual(select_best_model(metrics), "random_forest")

    def test_extract_feature_importance_random_forest(self):
        rng = np.random.default_rng(RANDOM_STATE)
        X = rng.normal(size=(60, len(FEATURE_KEYS)))
        y = rng.integers(0, 4, size=60)
        model = _create_random_forest()
        model.fit(X, y)
        importance = extract_feature_importance(model, "random_forest")
        self.assertEqual(set(importance.keys()), set(FEATURE_KEYS))
        self.assertAlmostEqual(sum(importance.values()), 1.0, places=5)

    def test_extract_feature_importance_xgboost(self):
        rng = np.random.default_rng(RANDOM_STATE)
        X = rng.normal(size=(60, len(FEATURE_KEYS)))
        y = rng.integers(0, 4, size=60)
        model = _create_xgboost_classifier()
        model.fit(X, y)
        importance = extract_feature_importance(model, "xgboost")
        self.assertEqual(set(importance.keys()), set(FEATURE_KEYS))
        self.assertGreater(sum(importance.values()), 0.0)

    def test_build_classifier_report_schema(self):
        report = build_classifier_report(
            comparison_metrics={
                "random_forest": {
                    "accuracy": 0.9,
                    "precision": 0.88,
                    "recall": 0.87,
                    "f1_score": 0.875,
                },
                "xgboost": {
                    "accuracy": 0.91,
                    "precision": 0.89,
                    "recall": 0.88,
                    "f1_score": 0.885,
                },
            },
            selected_model="xgboost",
            feature_importance={
                "random_forest": {key: 1.0 / len(FEATURE_KEYS) for key in FEATURE_KEYS},
                "xgboost": {key: 1.0 / len(FEATURE_KEYS) for key in FEATURE_KEYS},
            },
            training_samples=160,
            test_samples=40,
        )

        self.assertEqual(report["schema_version"], REPORT_SCHEMA_VERSION)
        self.assertEqual(report["selection_criterion"], SELECTION_CRITERION)
        self.assertTrue(report["morphology_features_included"])
        self.assertEqual(report["legacy_feature_count"], len(LEGACY_FEATURE_KEYS))
        self.assertEqual(report["morphology_feature_count"], len(MORPHOLOGY_FEATURE_KEYS))
        self.assertIn("model_comparison", report)
        self.assertIn("feature_importance", report)
        self.assertIn("feature_importance_summary", report)
        self.assertIn("top_random_forest", report["feature_importance_summary"])
        self.assertIn("top_xgboost", report["feature_importance_summary"])

    def test_save_classifier_report_writes_json(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            report_path = Path(tmp_dir) / "classifier_report.json"
            payload = {"schema_version": REPORT_SCHEMA_VERSION, "selected_model": "random_forest"}
            save_classifier_report(payload, report_path=report_path)
            self.assertTrue(report_path.exists())
            loaded = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["selected_model"], "random_forest")

    def test_build_training_dataset_includes_morphology_columns(self):
        with patch("backend.classifier.TRAINING_SET_SIZE", 40):
            X, y = build_training_dataset(training_set_size=40)
        self.assertEqual(X.shape[1], len(FEATURE_KEYS))
        self.assertGreater(len(y), 0)
        self.assertTrue(morphology_features_included())

    def test_train_classifier_dual_pipeline_and_report(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "classifier.pkl"
            report_path = Path(tmp_dir) / "classifier_report.json"
            with patch("backend.classifier.MODEL_PATH", model_path):
                with patch("backend.classifier.REPORT_PATH", report_path):
                    with patch("backend.classifier.TRAINING_SET_SIZE", 40):
                        model = train_classifier(training_set_size=40, test_size=0.25)

            self.assertTrue(model_path.exists())
            self.assertTrue(report_path.exists())

            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertIn(report["selected_model"], ("random_forest", "xgboost"))
            for model_name in ("random_forest", "xgboost"):
                for metric in ("accuracy", "precision", "recall", "f1_score"):
                    self.assertIn(metric, report["model_comparison"][model_name])
                self.assertEqual(
                    set(report["feature_importance"][model_name].keys()),
                    set(FEATURE_KEYS),
                )

            loaded = joblib.load(model_path)
            self.assertEqual(_expected_feature_count(loaded), len(FEATURE_KEYS))

    def test_predict_class_backward_compatible_with_legacy_keys(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "classifier.pkl"
            report_path = Path(tmp_dir) / "classifier_report.json"
            with patch("backend.classifier.MODEL_PATH", model_path):
                with patch("backend.classifier.REPORT_PATH", report_path):
                    with patch("backend.classifier.TRAINING_SET_SIZE", 40):
                        train_classifier(training_set_size=40, test_size=0.25)

            legacy_features = {key: 0.0 for key in LEGACY_FEATURE_KEYS}
            legacy_features.update({"period": 3.0, "depth": 0.01, "snr": 10.0})

            with patch("backend.classifier.MODEL_PATH", model_path):
                prediction = predict_class(legacy_features)

            self.assertIn("prediction_label", prediction)
            self.assertIn("confidence", prediction)
            self.assertIn("probabilities", prediction)
            self.assertTrue(0.0 <= prediction["confidence"] <= 1.0)
            self.assertIn(prediction["prediction_label"], prediction["probabilities"])

    def test_load_or_train_model_retrains_on_feature_mismatch(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "classifier.pkl"
            report_path = Path(tmp_dir) / "classifier_report.json"
            stale_model = _create_random_forest()
            rng = np.random.default_rng(RANDOM_STATE)
            stale_X = rng.normal(size=(40, len(LEGACY_FEATURE_KEYS)))
            stale_y = rng.integers(0, 4, size=40)
            stale_model.fit(stale_X, stale_y)
            joblib.dump(stale_model, model_path)

            with patch("backend.classifier.MODEL_PATH", model_path):
                with patch("backend.classifier.REPORT_PATH", report_path):
                    with patch("backend.classifier.TRAINING_SET_SIZE", 40):
                        model = load_or_train_model()

            self.assertEqual(_expected_feature_count(model), len(FEATURE_KEYS))

if __name__ == "__main__":
    unittest.main()
