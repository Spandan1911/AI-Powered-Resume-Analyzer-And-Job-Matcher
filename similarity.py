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
