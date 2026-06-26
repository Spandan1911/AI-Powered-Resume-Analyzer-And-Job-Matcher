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

st.set_page_config(page_title="AI Resume Analyzer & Job Matcher", layout="wide")

st.title("📄 AI-Powered Resume Analyzer & Job Matcher")
st.caption("NLP (TF-IDF, BERT) • XGBoost ATS Scoring • FAISS RAG Suggestions")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1️⃣ Upload Resume")
    resume_file = st.file_uploader("Upload your resume (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"])

with col2:
    st.subheader("2️⃣ Job Description")
    jd_text_input = st.text_area("Paste the job description here", height=220)

analyze_clicked = st.button("🔍 Analyze Resume", type="primary", use_container_width=True)
