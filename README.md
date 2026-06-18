# AI-Powered-Resume-Analyzer-And-Job-Matcher
This project analyzes a candidate's resume against a job description  using AI and ML techniques. It extracts skills, computes keyword and  semantic similarity using TF-IDF and BERT, predicts ATS score using  XGBoost, detects missing skills, and generates AI-powered suggestions  using RAG and FAISS — all running locally with no external API.


This project analyzes a candidate's resume against a job description using AI and Machine Learning techniques. The objective is to help job seekers understand how well their resume matches a role, identify missing skills, predict ATS compatibility, and receive AI-powered suggestions to improve their resume before applying.

🎯 Objectives

- Automatically extract text from resume files (PDF, DOCX, TXT)
- Extract and identify technical and soft skills from resume and job description
- Compute keyword-based similarity between resume and job description
- Compute semantic/meaning-based similarity using BERT
- Predict ATS (Applicant Tracking System) score using XGBoost
- Detect skills present in job description but missing from resume
- Generate personalized AI suggestions using RAG and FAISS
- Display all results on an interactive web dashboard

🛠️ Technologies Used

- Python
- Streamlit
- Scikit-learn
- Sentence-Transformers (BERT)
- XGBoost
- FAISS (Facebook AI Similarity Search)
- PyPDF2
- Python-docx
- NumPy
- Pandas
- Regex (re)
- Joblib

⚙️ Algorithms Used

- TF-IDF (Term Frequency - Inverse Document Frequency) — Keyword similarity
- BERT (all-MiniLM-L6-v2) — Semantic meaning similarity
- Cosine Similarity — Vector comparison between resume and JD
- XGBoost Regressor — ATS score prediction (0 to 100)
- FAISS (IndexFlatIP) — Fast vector search for relevant suggestions
- RAG (Retrieval Augmented Generation) — AI suggestion generation
- Regex Pattern Matching — Skill extraction from text

✨ Features

- Upload Resume in PDF, DOCX, or TXT format
- Extract Skills automatically from both resume and job description
- Keyword Match % — how many important JD keywords appear in resume (TF-IDF)
- Meaning Match % — how semantically similar resume is to job description (BERT)
- ATS Score — predicted ATS compatibility score out of 100 (XGBoost)
- Matched Skills — skills present in both resume and job description
- Missing Skills — skills required in JD but absent from resume
- AI Suggestions — personalized tips to improve resume (RAG + FAISS)
- Interactive Sidebar — user-friendly guidance for non-technical users
