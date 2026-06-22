"""
similarity.py
Computes similarity between resume and job description using:
  1. TF-IDF + Cosine Similarity (classic NLP baseline)
  2. BERT (Sentence-Transformers) embeddings + Cosine Similarity (semantic)
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

# Load once globally (heavy model)
_bert_model = None


def get_bert_model():
    global _bert_model
    if _bert_model is None:
        _bert_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _bert_model


def tfidf_similarity(resume_text: str, jd_text: str) -> float:
    """Returns cosine similarity (0-1) between resume and JD using TF-IDF."""
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
    sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return float(sim)


def bert_similarity(resume_text: str, jd_text: str) -> float:
    """Returns cosine similarity (0-1) between resume and JD using BERT embeddings."""
    model = get_bert_model()
    embeddings = model.encode([resume_text, jd_text], convert_to_numpy=True)
    sim = cosine_similarity(embeddings[0:1], embeddings[1:2])[0][0]
    return float(sim)


def combined_similarity(resume_text: str, jd_text: str,
                         tfidf_weight: float = 0.4, bert_weight: float = 0.6) -> dict:
    """Returns a dict with individual and weighted-combined similarity scores."""
    tfidf_score = tfidf_similarity(resume_text, jd_text)
    bert_score = bert_similarity(resume_text, jd_text)
    combined = tfidf_weight * tfidf_score + bert_weight * bert_score
    return {
        "tfidf_similarity": round(tfidf_score, 4),
        "bert_similarity": round(bert_score, 4),
        "combined_similarity": round(float(combined), 4),
    }
