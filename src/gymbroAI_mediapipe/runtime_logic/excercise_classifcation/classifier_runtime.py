"""
XGBoost classifier loading and inference for exercise prediction.

Loads the trained model, label encoder, and feature config from disk,
then classifies each frame's landmarks into an exercise name.
"""

import os

import joblib
import numpy as np


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_classifier(models_dir):
    """Load XGBoost model, label encoder, and feature config from disk."""
    model = joblib.load(os.path.join(models_dir, "model_updated.pkl"))
    encoder = joblib.load(os.path.join(models_dir, "encoder_updated.pkl"))
    feature_config = joblib.load(os.path.join(models_dir, "feature_config_updated.pkl"))
    return model, encoder, feature_config


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def predict_exercise(curr_lm, model, encoder, feature_config):
    """Return the predicted exercise name string for the current frame."""
    features = _build_feature_vector(curr_lm, feature_config)
    prediction = model.predict(features.reshape(1, -1))[0]

    try:
        return encoder.inverse_transform([prediction])[0]
    except Exception:
        return str(prediction)


def _build_feature_vector(curr_lm, feature_config):
    """Build a numpy feature vector from landmarks and the saved config."""
    positions = _extract_relative_positions(curr_lm, feature_config["body_landmarks"])
    return np.array(positions, dtype=np.float32)


def _extract_relative_positions(lm, body_landmarks):
    """Compute body-center-relative, torso-normalized positions for each landmark."""
    hip_cx = (lm["left_hip_x"] + lm["right_hip_x"]) / 2.0
    hip_cy = (lm["left_hip_y"] + lm["right_hip_y"]) / 2.0
    hip_cz = (lm["left_hip_z"] + lm["right_hip_z"]) / 2.0

    sho_cx = (lm["left_shoulder_x"] + lm["right_shoulder_x"]) / 2.0
    sho_cy = (lm["left_shoulder_y"] + lm["right_shoulder_y"]) / 2.0
    sho_cz = (lm["left_shoulder_z"] + lm["right_shoulder_z"]) / 2.0

    body_cx = (hip_cx + sho_cx) / 2.0
    body_cy = (hip_cy + sho_cy) / 2.0
    body_cz = (hip_cz + sho_cz) / 2.0

    torso = max(
        np.sqrt(
            (sho_cx - hip_cx) ** 2 + (sho_cy - hip_cy) ** 2 + (sho_cz - hip_cz) ** 2
        ),
        1e-6,
    )

    feats = []
    for name in body_landmarks:
        feats.extend(
            [
                (lm[f"{name}_x"] - body_cx) / torso,
                (lm[f"{name}_y"] - body_cy) / torso,
                (lm[f"{name}_z"] - body_cz) / torso,
            ]
        )
    return feats
