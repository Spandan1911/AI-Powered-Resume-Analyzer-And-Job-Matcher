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

if analyze_clicked:
    if resume_file is None:
        st.error("Please upload a resume file.")
    elif not jd_text_input.strip():
        st.error("Please paste a job description.")
    else:
        with st.spinner("Extracting text..."):
            raw_resume_text = extract_text(resume_file.read(), resume_file.name)
            resume_clean = clean_text(raw_resume_text)
            jd_clean = clean_text(jd_text_input)

        # --- Skill Extraction ---
        with st.spinner("Extracting skills..."):
            resume_skills = extract_skills(resume_clean)
            jd_skills = extract_skills(jd_clean)
            matched_skills = get_matched_skills(resume_skills, jd_skills)
            missing_skills = get_missing_skills(resume_skills, jd_skills)
            skill_match_ratio = (
                len(matched_skills) / len(jd_skills) if jd_skills else 0.0
            )

        # --- Similarity (TF-IDF + BERT) ---
        with st.spinner("Computing similarity (TF-IDF + BERT)..."):
            sim_scores = combined_similarity(resume_clean, jd_clean)

        # --- ATS Score (XGBoost) ---
        with st.spinner("Predicting ATS score (XGBoost)..."):
            word_count = len(raw_resume_text.split())
            ats_score = predict_ats_score(
                tfidf_sim=sim_scores["tfidf_similarity"],
                bert_sim=sim_scores["bert_similarity"],
                skill_match_ratio=skill_match_ratio,
                resume_word_count=word_count,
                num_matched=len(matched_skills),
                num_missing=len(missing_skills),
            )

        # --- RAG Suggestions (FAISS) ---
        with st.spinner("Generating AI suggestions (RAG + FAISS)..."):
            suggestions = generate_suggestions(missing_skills, jd_skills, resume_skills, ats_score)

        st.success("Analysis complete!")
        st.divider()

        # --- Results Display ---
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("ATS Score", f"{ats_score:.1f} / 100")
        m2.metric("TF-IDF Similarity", f"{sim_scores['tfidf_similarity']*100:.1f}%")
        m3.metric("BERT Similarity", f"{sim_scores['bert_similarity']*100:.1f}%")
        m4.metric("Skill Match", f"{skill_match_ratio*100:.1f}%")

        st.divider()

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("✅ Matched Skills")
            if matched_skills:
                st.write(", ".join(s.title() for s in matched_skills))
            else:
                st.write("No matched skills found.")

        with c2:
            st.subheader("⚠️ Missing Skills")
            if missing_skills:
                st.write(", ".join(s.title() for s in missing_skills))
            else:
                st.write("None! Your resume covers all required skills.")

        st.divider()
        st.subheader("💡 AI Suggestions (RAG-powered)")
        for tip in suggestions["ai_suggestions"]:
            st.markdown(f"- {tip}")

        st.divider()
        with st.expander("🔧 Debug: Extracted Resume Skills"):
            st.write(sorted(resume_skills))
        with st.expander("🔧 Debug: Extracted JD Skills"):
            st.write(sorted(jd_skills))

st.sidebar.header("ℹ️ About")
st.sidebar.markdown("""
**Pipeline:**
1. Extract text from resume (PDF/DOCX/TXT)
2. Extract skills via keyword/phrase matching
3. Compute TF-IDF + BERT cosine similarity
4. Predict ATS score using XGBoost
5. Detect missing skills vs job description
6. Generate AI suggestions via RAG + FAISS retrieval
""")
