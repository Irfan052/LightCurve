import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.ensemble import RandomForestClassifier
from backend.config import MODEL_PATH, TRAINING_SET_SIZE, CLASSES, RANDOM_STATE
from backend.data_loader import generate_synthetic_lightcurve
from backend.quality_filter import clean_lightcurve
from backend.detrend import flatten_lightcurve
from backend.transit_search import search_transits
from backend.feature_engineering import extract_features
from backend.utils import logger

FEATURE_KEYS = [
    "period", "depth", "duration", "snr", "shape_score", 
    "odd_even_diff", "secondary_depth", "secondary_ratio", 
    "out_transit_var", "symmetry"
]

def load_or_train_model() -> RandomForestClassifier:
    """Loads the classifier model. If not found, triggers auto-training."""
    if MODEL_PATH.exists():
        try:
            logger.info("Loading pre-trained classifier model...")
            return joblib.load(MODEL_PATH)
        except Exception as e:
            logger.warning(f"Failed to load model: {str(e)}. Retraining...")
            
    # Model not found or load failed, train a new one
    return train_classifier()

def train_classifier() -> RandomForestClassifier:
    """
    Generates a synthetic training set of light curves, processes them 
    through the pipeline, extracts features, and trains a Random Forest classifier.
    """
    logger.info("Starting model auto-training on synthetic datasets...")
    
    # Class allocations: 200 samples total, 50 per class
    classes_to_generate = [
        ("instrumental_artifact", 0),
        ("stellar_variability", 1),
        ("eclipsing_binary", 2),
        ("exoplanet_transit", 3)
    ]
    
    X_list = []
    y_list = []
    
    samples_per_class = max(10, TRAINING_SET_SIZE // 4)
    
    for class_name, label in classes_to_generate:
        logger.info(f"Generating training data for class: {class_name} ({samples_per_class} samples)...")
        success_count = 0
        attempts = 0
        
        while success_count < samples_per_class and attempts < samples_per_class * 2:
            attempts += 1
            try:
                # Generate
                time, flux, flux_err = generate_synthetic_lightcurve(class_name)
                
                # Preprocess
                time_c, flux_c, flux_err_c = clean_lightcurve(time, flux, flux_err)
                flat_flux, _ = flatten_lightcurve(time_c, flux_c)
                
                # Search
                bls_res = search_transits(time_c, flat_flux, flux_err_c)
                
                # Feature extract
                feats = extract_features(time_c, flat_flux, flux_err_c, bls_res)
                
                # Convert feature dict to vector
                vector = [feats.get(k, 0.0) for k in FEATURE_KEYS]
                
                # Ensure no NaNs or infs in feature vector
                if not np.any(np.isnan(vector)) and not np.any(np.isinf(vector)):
                    X_list.append(vector)
                    y_list.append(label)
                    success_count += 1
            except Exception as e:
                logger.error(f"Error generating sample {success_count} for {class_name}: {str(e)}")
                
    X = np.array(X_list)
    y = np.array(y_list)
    
    logger.info(f"Training set ready. Total features matrix shape: {X.shape}")
    
    # Fit Random Forest
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=RANDOM_STATE,
        class_weight="balanced"
    )
    rf.fit(X, y)
    
    # Save the model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, MODEL_PATH)
    logger.info(f"Classifier model trained and saved to {MODEL_PATH}")
    
    return rf

def predict_class(features: Dict[str, float]) -> Dict[str, Any]:
    """
    Predicts the classification label and confidence score from extracted features.
    
    Returns:
        prediction_label: Name of the predicted class
        confidence: Probability score of the predicted class (0.0 to 1.0)
        probabilities: Dictionary of probabilities for all classes
    """
    model = load_or_train_model()
    
    # Format feature vector
    vector = np.array([[features.get(k, 0.0) for k in FEATURE_KEYS]])
    
    # Replace NaNs/Infs just in case
    vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Run prediction
    pred_id = int(model.predict(vector)[0])
    pred_prob = model.predict_proba(vector)[0]
    
    prediction_label = CLASSES[pred_id]
    confidence = float(pred_prob[pred_id])
    
    probabilities = {CLASSES[i]: float(prob) for i, prob in enumerate(pred_prob)}
    
    return {
        "prediction_label": prediction_label,
        "confidence": confidence,
        "probabilities": probabilities
    }
