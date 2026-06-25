"""
ats_model.py
Trains (or loads) an XGBoost regression model to predict an "ATS Score" (0-100)
based on engineered features:
  - tfidf_similarity
  - bert_similarity
  - skill_match_ratio
  - resume_length (word count, normalized)
  - num_matched_skills
  - num_missing_skills

Since real labeled ATS data is rarely available, this module generates
synthetic but realistic training data based on heuristic rules, then
fits an XGBoost model on it. The trained model is cached to disk.
"""

import os
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib

MODEL_PATH = os.path.join(os.path.dirname(__file__), "ats_xgb_model.joblib")


def _generate_synthetic_data(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic feature/label pairs.
    Label (ATS score) is computed via a weighted heuristic + noise,
    which XGBoost then learns to approximate (and can generalize on
    real feature distributions at inference time).
    """
    rng = np.random.default_rng(seed)

    tfidf_sim = rng.beta(2, 2, n_samples)            # 0-1
    bert_sim = rng.beta(2.5, 2, n_samples)           # 0-1
    skill_match_ratio = rng.beta(2, 2, n_samples)    # 0-1
    resume_len_norm = rng.uniform(0.2, 1.0, n_samples)  # normalized length
    num_matched = (skill_match_ratio * 20).round()
    num_missing = ((1 - skill_match_ratio) * 15).round()

    # Heuristic "true" ATS score formula
    score = (
        tfidf_sim * 20 +
        bert_sim * 35 +
        skill_match_ratio * 35 +
        resume_len_norm * 10
    )
    # Add noise and clip to 0-100
    noise = rng.normal(0, 3, n_samples)
    score = np.clip(score + noise, 0, 100)

    df = pd.DataFrame({
        "tfidf_similarity": tfidf_sim,
        "bert_similarity": bert_sim,
        "skill_match_ratio": skill_match_ratio,
        "resume_length_norm": resume_len_norm,
        "num_matched_skills": num_matched,
        "num_missing_skills": num_missing,
        "ats_score": score,
    })
    return df


def train_model() -> xgb.XGBRegressor:
    """Train XGBoost regressor on synthetic data and save it."""
    df = _generate_synthetic_data()
    feature_cols = [
        "tfidf_similarity", "bert_similarity", "skill_match_ratio",
        "resume_length_norm", "num_matched_skills", "num_missing_skills"
    ]
    X = df[feature_cols]
    y = df["ats_score"]

    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror",
    )
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    return model
