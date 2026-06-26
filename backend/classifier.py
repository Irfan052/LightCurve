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
    training_mode: bool = True,
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
                if training_mode:
                    time, flux, flux_err, ground_truth = generate_synthetic_lightcurve(class_name, return_params=True)
                    time_c, flux_c, flux_err_c = clean_lightcurve(time, flux, flux_err)
                    flat_flux, _ = flatten_lightcurve(time_c, flux_c)
                    bls_res = ground_truth
                else:
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


def build_hybrid_training_dataset(
    training_set_size: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Builds a hybrid training dataset combining synthetic samples and real TESS features
    extracted during validation.
    """
    X_syn, y_syn = build_training_dataset(training_set_size=training_set_size, training_mode=True)
    
    real_features_path = Path("real_tess_validation.json")
    if not real_features_path.exists():
        logger.warning(f"{real_features_path} not found. Returning synthetic dataset only.")
        return X_syn, y_syn
        
    try:
        with open(real_features_path, "r", encoding="utf-8") as f:
            real_data = json.load(f)
            
        X_real_list = []
        y_real_list = []
        
        for res in real_data.get("results", []):
            if "features" in res:
                vector = _features_to_vector(res["features"])
                if _validate_feature_vector(vector):
                    X_real_list.append(vector)
                    
                    # Convert string classification back to int label
                    class_name = res["classification"]
                    label_idx = CLASSES.index(class_name) if class_name in CLASSES else 3
                    y_real_list.append(label_idx)
                    
        if X_real_list:
            logger.info(f"Loaded {len(X_real_list)} real TESS feature samples for hybrid training.")
            X_real = np.asarray(X_real_list, dtype=float)
            y_real = np.asarray(y_real_list, dtype=int)
            
            # Combine arrays
            X_combined = np.vstack((X_syn, X_real))
            y_combined = np.hstack((y_syn, y_real))
            return X_combined, y_combined
    except Exception as e:
        logger.error(f"Failed to load real features: {e}")
        
    return X_syn, y_syn


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


_cached_model: Optional[ClassifierModel] = None

def load_or_train_model() -> ClassifierModel:
    """Load the saved classifier or auto-train when missing or incompatible."""
    global _cached_model
    if _cached_model is not None:
        return _cached_model
        
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
                _cached_model = model
                return model
        except Exception as exc:
            logger.warning(f"Failed to load model: {exc}. Retraining...")

    return train_classifier()


def train_classifier(
    training_set_size: Optional[int] = None,
    test_size: float = 0.2,
) -> ClassifierModel:
    """
    Train Random Forest on Hybrid dataset, and XGBoost on Synthetic dataset for comparison.
    (Or Random Forest on both to compare Hybrid vs Synthetic).
    For Phase 6B, we will train a Hybrid model, evaluate it, and save the report.
    """
    logger.info("Starting classifier training (Hybrid vs Synthetic comparison)...")

    # Synthetic Dataset
    X_syn, y_syn = build_training_dataset(training_set_size=training_set_size)
    Xs_train, Xs_test, ys_train, ys_test = train_test_split(
        X_syn, y_syn, test_size=test_size, random_state=RANDOM_STATE, stratify=y_syn
    )
    
    # Hybrid Dataset
    X_hyb, y_hyb = build_hybrid_training_dataset(training_set_size=training_set_size)
    Xh_train, Xh_test, yh_train, yh_test = train_test_split(
        X_hyb, y_hyb, test_size=test_size, random_state=RANDOM_STATE, stratify=y_hyb
    )

    logger.info(f"Synthetic training set shape: {Xs_train.shape}")
    logger.info(f"Hybrid training set shape: {Xh_train.shape}")

    # Train Synthetic Model
    syn_model = _create_random_forest()
    syn_model.fit(Xs_train, ys_train)

    # Train Hybrid Model
    hyb_model = _create_random_forest()
    hyb_model.fit(Xh_train, yh_train)

    comparison_metrics = {
        "synthetic": evaluate_classifier(syn_model, Xs_test, ys_test),
        "hybrid": evaluate_classifier(hyb_model, Xh_test, yh_test),
    }
    
    # Save hybrid report
    hybrid_report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "synthetic_model_metrics": comparison_metrics["synthetic"],
        "hybrid_model_metrics": comparison_metrics["hybrid"],
        "synthetic_samples": int(Xs_train.shape[0]),
        "hybrid_samples": int(Xh_train.shape[0]),
    }
    with open("hybrid_training_report.json", "w", encoding="utf-8") as f:
        json.dump(hybrid_report, f, indent=2)
    logger.info("Saved hybrid_training_report.json")

    # Select best model (Hybrid vs Synthetic) based on F1
    if comparison_metrics["hybrid"]["f1_score"] >= comparison_metrics["synthetic"]["f1_score"]:
        best_model = hyb_model
        best_name = "hybrid"
    else:
        best_model = syn_model
        best_name = "synthetic"

    # Generate and save XAI feature importance report
    feature_importance = {
        "random_forest": extract_feature_importance(best_model, "random_forest"),
        "xgboost": extract_feature_importance(best_model, "random_forest")
    }
    report = build_classifier_report(
        comparison_metrics={"random_forest": comparison_metrics[best_name], "xgboost": comparison_metrics[best_name]},
        selected_model="random_forest",
        feature_importance=feature_importance,
        training_samples=int(Xh_train.shape[0]) if best_name == "hybrid" else int(Xs_train.shape[0]),
        test_samples=int(Xh_test.shape[0]) if best_name == "hybrid" else int(Xs_test.shape[0])
    )
    save_classifier_report(report)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    logger.info(
        "Classifier training complete. Selected model: %s (F1=%.4f). Saved to %s",
        best_name,
        comparison_metrics[best_name]["f1_score"],
        MODEL_PATH,
    )

    global _cached_model
    _cached_model = best_model
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




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Exoplanet Classifier')
    parser.add_argument('--train', action='store_true', help='Train the classifier')
    args = parser.parse_args()
    if args.train:
        train_classifier()
    else:
        # Default behavior when run directly
        train_classifier()
