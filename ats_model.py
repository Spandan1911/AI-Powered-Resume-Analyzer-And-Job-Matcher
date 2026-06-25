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
