"""
app.py
AI-Powered Resume Analyzer & Job Matcher
Run with: streamlit run app.py
"""

import streamlit as st
from extract_utils import extract_text, clean_text
from skill_extractor import extract_skills, get_matched_skills, get_missing_skills
from similarity import combined_similarity
from ats_model import predict_ats_score
from rag_suggestions import generate_suggestions
