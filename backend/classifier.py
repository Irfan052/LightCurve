import json
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import patch

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from backend.config import MODEL_PATH, TRAINING_SET_SIZE, CLASSES, RANDOM_STATE
from backend.data_loader import generate_synthetic_lightcurve
from backend.quality_filter import clean_lightcurve
from backend.detrend import flatten_lightcurve
from backend.transit_search import search_transits
from backend.feature_engineering import extract_features
from backend.utils import logger

# Legacy ML feature keys preserved for backward compatibility.
LEGACY_FEATURE_KEYS: List[str] = [
    "period", "depth", "duration", "snr", "shape_score",
    "odd_even_diff", "secondary_depth", "secondary_ratio",
    "out_transit_var", "symmetry",
]

# Phase 2D folded-light-curve morphology features.
MORPHOLOGY_FEATURE_KEYS: List[str] = [
    "odd_depth", "even_depth", "secondary_phase_flux",
    "ingress_slope", "egress_slope", "transit_symmetry",
    "u_shape_score", "v_shape_score", "out_of_transit_rms",
    "depth_to_rms", "transit_bottom_width", "transit_full_width",
    "in_transit_points", "out_of_transit_points", "folded_point_count",
    "binned_transit_min_flux", "binned_transit_depth", "transit_centroid_offset",
]

# Full training feature set: legacy keys first, then morphology extensions.
FEATURE_KEYS: List[str] = LEGACY_FEATURE_KEYS + MORPHOLOGY_FEATURE_KEYS

REPORT_PATH = MODEL_PATH.parent / "classifier_report.json"
REPORT_SCHEMA_VERSION = "phase_2e"
SELECTION_CRITERION = "f1_score"

ClassifierModel = Union[RandomForestClassifier, XGBClassifier]


def _features_to_vector(features: Dict[str, float]) -> List[float]:
    """Convert a feature dictionary to a fixed-order numeric vector."""
    return [float(features.get(k, 0.0)) for k in FEATURE_KEYS]


def _validate_feature_vector(vector: List[float]) -> bool:
    """Return True when the vector contains only finite values."""
    arr = np.asarray(vector, dtype=float)
    return not np.any(np.isnan(arr)) and not np.any(np.isinf(arr))


def morphology_features_included() -> bool:
    """Verify Phase 2D morphology keys are part of the training feature set."""
    return all(key in FEATURE_KEYS for key in MORPHOLOGY_FEATURE_KEYS)


def build_training_dataset(
    training_set_size: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic light curves, extract features, and return X, y arrays.

    Both Random Forest and XGBoost are trained on the identical feature matrix.
    """
    if not morphology_features_included():
        raise ValueError("Phase 2D morphology features are missing from FEATURE_KEYS.")

    size = training_set_size if training_set_size is not None else TRAINING_SET_SIZE
    classes_to_generate = [
        ("instrumental_artifact", 0),
        ("stellar_variability", 1),
        ("eclipsing_binary", 2),
        ("exoplanet_transit", 3),
    ]

    X_list: List[List[float]] = []
    y_list: List[int] = []
    samples_per_class = max(10, size // 4)

    for class_name, label in classes_to_generate:
        logger.info(
            f"Generating training data for class: {class_name} ({samples_per_class} samples)..."
        )
        success_count = 0
        attempts = 0

        while success_count < samples_per_class and attempts < samples_per_class * 2:
            attempts += 1
            try:
                time, flux, flux_err = generate_synthetic_lightcurve(class_name)
                time_c, flux_c, flux_err_c = clean_lightcurve(time, flux, flux_err)
                flat_flux, _ = flatten_lightcurve(time_c, flux_c)
                bls_res = search_transits(time_c, flat_flux, flux_err_c)
                feats = extract_features(time_c, flat_flux, flux_err_c, bls_res)
                vector = _features_to_vector(feats)

                if _validate_feature_vector(vector):
                    X_list.append(vector)
                    y_list.append(label)
                    success_count += 1
            except Exception as exc:
                logger.error(
                    f"Error generating sample {success_count} for {class_name}: {exc}"
                )

    if not X_list:
        raise RuntimeError("Failed to build a non-empty training dataset.")

    return np.asarray(X_list, dtype=float), np.asarray(y_list, dtype=int)


def _create_random_forest() -> RandomForestClassifier:
    """Instantiate the legacy Random Forest classifier with project defaults."""
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )


def _create_xgboost_classifier() -> XGBClassifier:
    """Instantiate an XGBoost multi-class classifier aligned with RF depth."""
    return XGBClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=RANDOM_STATE,
        objective="multi:softprob",
        eval_metric="mlogloss",
        verbosity=0,
    )


def evaluate_classifier(
    model: ClassifierModel,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """Compute accuracy, precision, recall, and F1 score on held-out data."""
    y_pred = model.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(
            precision_score(y_test, y_pred, average="macro", zero_division=0)
        ),
        "recall": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
    }


def _model_ranking_key(metrics: Dict[str, float]) -> Tuple[float, float, float, float]:
    """Rank models by F1, then accuracy, precision, and recall."""
    return (
        metrics["f1_score"],
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
    )


def select_best_model(
    comparison_metrics: Dict[str, Dict[str, float]],
) -> str:
    """Select the best-performing model based on macro F1 score."""
    candidates = ["random_forest", "xgboost"]
    return max(candidates, key=lambda name: _model_ranking_key(comparison_metrics[name]))


def extract_feature_importance(
    model: ClassifierModel,
    model_name: str,
) -> Dict[str, float]:
    """Extract per-feature importance for Random Forest or XGBoost."""
    if model_name == "random_forest":
        importances = getattr(model, "feature_importances_", None)
    elif model_name == "xgboost":
        importances = getattr(model, "feature_importances_", None)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    if importances is None:
        return {key: 0.0 for key in FEATURE_KEYS}

    return {
        feature: float(importance)
        for feature, importance in zip(FEATURE_KEYS, importances)
    }


def summarize_feature_importance(
    importance_map: Dict[str, float],
    top_n: int = 5,
) -> List[Dict[str, Union[str, float]]]:
    """Return the top-N features ranked by importance."""
    ranked = sorted(importance_map.items(), key=lambda item: item[1], reverse=True)
    return [
        {"feature": feature, "importance": float(importance)}
        for feature, importance in ranked[:top_n]
    ]


def build_classifier_report(
    comparison_metrics: Dict[str, Dict[str, float]],
    selected_model: str,
    feature_importance: Dict[str, Dict[str, float]],
    training_samples: int,
    test_samples: int,
) -> Dict[str, Any]:
    """
    Build the classifier_report.json payload.

    Schema (classifier_report.json):
        schema_version (str): Report format identifier ("phase_2e").
        generated_at (str): UTC ISO-8601 timestamp.
        feature_keys (list[str]): Ordered training feature names.
        legacy_feature_count (int): Count of legacy ML features.
        morphology_feature_count (int): Count of Phase 2D morphology features.
        morphology_features_included (bool): Whether morphology keys are in training.
        morphology_feature_keys (list[str]): Phase 2D morphology feature names.
        training_samples (int): Number of training rows.
        test_samples (int): Number of held-out evaluation rows.
        selection_criterion (str): Metric used to pick the best model ("f1_score").
        selected_model (str): "random_forest" or "xgboost".
        model_comparison (object):
            random_forest (object): accuracy, precision, recall, f1_score.
            xgboost (object): accuracy, precision, recall, f1_score.
        feature_importance (object):
            random_forest (object): feature name -> importance score.
            xgboost (object): feature name -> importance score.
        feature_importance_summary (object):
            top_random_forest (list): Top features for Random Forest.
            top_xgboost (list): Top features for XGBoost.
    """
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "feature_keys": FEATURE_KEYS,
        "legacy_feature_count": len(LEGACY_FEATURE_KEYS),
        "morphology_feature_count": len(MORPHOLOGY_FEATURE_KEYS),
        "morphology_features_included": morphology_features_included(),
        "morphology_feature_keys": MORPHOLOGY_FEATURE_KEYS,
        "training_samples": training_samples,
        "test_samples": test_samples,
        "selection_criterion": SELECTION_CRITERION,
        "selected_model": selected_model,
        "model_comparison": comparison_metrics,
        "feature_importance": feature_importance,
        "feature_importance_summary": {
            "top_random_forest": summarize_feature_importance(
                feature_importance["random_forest"]
            ),
            "top_xgboost": summarize_feature_importance(
                feature_importance["xgboost"]
            ),
        },
    }


def save_classifier_report(
    report: Dict[str, Any],
    report_path: Optional[Path] = None,
) -> Path:
    """Persist comparison metrics and feature importance to classifier_report.json."""
    destination = report_path if report_path is not None else REPORT_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    logger.info(f"Classifier comparison report saved to {destination}")
    return destination


def _expected_feature_count(model: ClassifierModel) -> Optional[int]:
    """Return the number of features the fitted model expects."""
    return getattr(model, "n_features_in_", None)


def load_or_train_model() -> ClassifierModel:
    """Load the saved classifier or auto-train when missing or incompatible."""
    if MODEL_PATH.exists():
        try:
            logger.info("Loading pre-trained classifier model...")
            model = joblib.load(MODEL_PATH)
            expected = _expected_feature_count(model)
            if expected is not None and expected != len(FEATURE_KEYS):
                logger.warning(
                    "Saved model feature count (%s) differs from current feature set (%s). Retraining...",
                    expected,
                    len(FEATURE_KEYS),
                )
            else:
                return model
        except Exception as exc:
            logger.warning(f"Failed to load model: {exc}. Retraining...")

    return train_classifier()


def train_classifier(
    training_set_size: Optional[int] = None,
    test_size: float = 0.2,
) -> ClassifierModel:
    """
    Train Random Forest and XGBoost on the same feature set, compare metrics,
    persist the best model, and write classifier_report.json.
    """
    logger.info("Starting dual-model classifier training on synthetic datasets...")

    X, y = build_training_dataset(training_set_size=training_set_size)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    logger.info(f"Training set ready. Feature matrix shape: {X_train.shape}")

    rf_model = _create_random_forest()
    rf_model.fit(X_train, y_train)

    xgb_model = _create_xgboost_classifier()
    xgb_model.fit(X_train, y_train)

    comparison_metrics = {
        "random_forest": evaluate_classifier(rf_model, X_test, y_test),
        "xgboost": evaluate_classifier(xgb_model, X_test, y_test),
    }
    selected_model = select_best_model(comparison_metrics)
    best_model = rf_model if selected_model == "random_forest" else xgb_model

    feature_importance = {
        "random_forest": extract_feature_importance(rf_model, "random_forest"),
        "xgboost": extract_feature_importance(xgb_model, "xgboost"),
    }

    report = build_classifier_report(
        comparison_metrics=comparison_metrics,
        selected_model=selected_model,
        feature_importance=feature_importance,
        training_samples=int(X_train.shape[0]),
        test_samples=int(X_test.shape[0]),
    )
    save_classifier_report(report)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    logger.info(
        "Classifier training complete. Selected model: %s (F1=%.4f). Saved to %s",
        selected_model,
        comparison_metrics[selected_model]["f1_score"],
        MODEL_PATH,
    )
    logger.info(
        "Model comparison — RF F1=%.4f, XGB F1=%.4f",
        comparison_metrics["random_forest"]["f1_score"],
        comparison_metrics["xgboost"]["f1_score"],
    )

    return best_model


def predict_class(features: Dict[str, float]) -> Dict[str, Any]:
    """
    Predict the classification label and confidence score from extracted features.

    Returns:
        prediction_label: Name of the predicted class
        confidence: Probability score of the predicted class (0.0 to 1.0)
        probabilities: Dictionary of probabilities for all classes
    """
    model = load_or_train_model()

    vector = np.array([_features_to_vector(features)], dtype=float)
    vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)

    pred_id = int(model.predict(vector)[0])
    pred_prob = model.predict_proba(vector)[0]

    prediction_label = CLASSES[pred_id]
    confidence = float(pred_prob[pred_id])
    probabilities = {CLASSES[i]: float(prob) for i, prob in enumerate(pred_prob)}

    return {
        "prediction_label": prediction_label,
        "confidence": confidence,
        "probabilities": probabilities,
    }


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
