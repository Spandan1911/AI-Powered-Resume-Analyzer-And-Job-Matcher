"""
rag_suggestions.py
Implements a lightweight RAG (Retrieval-Augmented Generation) pipeline using
FAISS for vector storage/retrieval of resume-writing best-practice snippets,
combined with a template-based generator to produce AI suggestions.

This avoids requiring an external LLM API key while still demonstrating the
RAG + FAISS architecture. If an LLM API key (Anthropic) is available, it can
optionally be used to generate richer suggestions (see `generate_with_llm`).
"""

import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Knowledge base of resume-improvement tips (the "documents" for RAG)
KNOWLEDGE_BASE = [
    "Use strong action verbs like 'led', 'built', 'optimized', and 'designed' instead of passive phrases like 'responsible for'.",
    "Quantify achievements with numbers and metrics, such as 'reduced latency by 30%' or 'increased revenue by $50K'.",
    "Tailor your resume keywords to match the exact terminology used in the job description for ATS optimization.",
    "Include a dedicated 'Skills' section listing both technical tools (e.g., Python, XGBoost, FAISS) and soft skills.",
    "Highlight experience with machine learning frameworks like TensorFlow, PyTorch, or scikit-learn when applying for ML roles.",
    "For NLP roles, mention specific techniques used such as TF-IDF, BERT embeddings, or transformer fine-tuning.",
    "If the job requires cloud experience, explicitly list AWS, Azure, or GCP services you've used (e.g., SageMaker, EC2, Lambda).",
    "Keep resume length concise: 1 page for under 5 years experience, max 2 pages otherwise.",
    "Use a clean, ATS-friendly format: avoid tables, images, and unusual fonts that parsers can't read.",
    "Add a 'Projects' section showcasing relevant personal or academic projects with GitHub links if applying for technical roles.",
    "Mention collaborative tools like Git, Docker, and CI/CD pipelines to show engineering maturity.",
    "For data roles, emphasize data visualization tools (Tableau, Power BI, matplotlib) and statistical analysis skills.",
    "Demonstrate impact by linking technical work to business outcomes, e.g., 'improved model accuracy from 82% to 91%, reducing manual review time by 40%'.",
    "Include certifications relevant to the target role (e.g., AWS Certified, Google Data Analytics) if you have them.",
    "If missing key skills from the JD, consider adding a 'Currently Learning' section or completing a short relevant project before applying.",
]
