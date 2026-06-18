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

🗂️ Project Structure

resume_analyzer/

 - app.py                  → Main Streamlit UI and dashboard
 - extract_utils.py        → PDF / DOCX / TXT text extraction and cleaning
 - skill_extractor.py      → Skill detection using regex and skills database
 - similarity.py           → TF-IDF and BERT similarity computation
 - ats_model.py            → XGBoost ATS score predictor
 - rag_suggestions.py      → FAISS + RAG AI suggestion engine
 - requirements.txt        → All required Python libraries

⚙️ How It Works

- Step 1 — User uploads resume (PDF / DOCX / TXT) and pastes job description
- Step 2 — Text is extracted from resume file using PyPDF2 or python-docx
- Step 3 — Text is cleaned (lowercase, remove special characters)
- Step 4 — Skills are extracted from both resume and job description using regex
- Step 5 — Matched skills and missing skills are identified
- Step 6 — TF-IDF computes keyword overlap between resume and JD
- Step 7 — BERT computes semantic/meaning similarity between resume and JD
- Step 8 — XGBoost predicts ATS score (0-100) using all computed features
- Step 9 — FAISS retrieves most relevant tips from knowledge base
- Step 10 — AI suggestions are generated and displayed on dashboard

🔍 Key Findings / Output

- ATS Score shows overall resume compatibility with the job (0 to 100)
- Keyword Match % reveals if resume uses same terminology as job description
- Meaning Match % shows if resume context aligns with job requirements
- Skill Match % shows percentage of required skills present in resume
- Missing skills list tells exactly what to add to improve the resume
- AI suggestions give specific, actionable tips based on the user's situation

🔌 API Usage

- No external API is used in this project
- Everything runs completely on local machine
- BERT model (~80MB) is downloaded once from HuggingFace on first run
- After first run the app works fully offline with no internet required
- Resume data never leaves the user's machine — complete privacy

🔎 Conclusion

This Resume Analyzer project demonstrates how NLP, Machine Learning, and Generative AI techniques can be combined to solve a real-world problem faced by millions of job seekers. The insights obtained from TF-IDF keyword matching, BERT semantic analysis, and XGBoost ATS prediction help candidates understand exactly where their resume falls short and what to fix before applying. The RAG + FAISS suggestion engine adds a layer of intelligent, personalized guidance — all running locally with no API cost and full privacy.

🚀 How to Run

- Go into the project folder
    cd resume-analyzer

- Install all dependencies
    pip install -r requirements.txt

- Run the application
   streamlit run app.py

- Open browser at
   http://localhost:8501
