"""
skill_extractor.py
Extracts skills from resume/JD text using a predefined skills database
with phrase matching (handles multi-word skills like 'machine learning').
"""

import re
from typing import Set, List

# A reasonably broad skills database. Extend as needed.
SKILLS_DB = [
    # Programming Languages
    "python", "java", "c++", "c#", "javascript", "typescript", "go", "rust",
    "r", "matlab", "scala", "kotlin", "swift", "php", "ruby", "sql",

    # ML / AI
    "machine learning", "deep learning", "natural language processing", "nlp",
    "computer vision", "reinforcement learning", "neural networks",
    "xgboost", "lightgbm", "random forest", "decision trees", "svm",
    "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn",
    "bert", "transformers", "gpt", "llm", "rag", "retrieval augmented generation",
    "huggingface", "opencv", "yolo",

    # NLP / Similarity
    "tf-idf", "tfidf", "cosine similarity", "word2vec", "spacy", "nltk",
    "faiss", "vector database", "embeddings", "semantic search",

    # Data Engineering / Tools
    "pandas", "numpy", "matplotlib", "seaborn", "plotly", "power bi", "tableau",
    "airflow", "spark", "hadoop", "kafka", "docker", "kubernetes",
    "git", "github", "jenkins", "ci/cd", "linux",

    # Cloud
    "aws", "azure", "gcp", "google cloud", "lambda", "ec2", "s3", "sagemaker",

    # Databases
    "mysql", "postgresql", "mongodb", "redis", "elasticsearch", "sqlite",

    # Web / App
    "react", "angular", "vue", "node.js", "django", "flask", "fastapi",
    "streamlit", "rest api", "graphql", "html", "css",

    # Soft / General
    "communication", "leadership", "project management", "agile", "scrum",
    "problem solving", "teamwork", "data analysis", "data visualization",
    "statistics", "a/b testing",
]

# Sort longer phrases first so multi-word skills are matched before substrings
SKILLS_DB_SORTED = sorted(set(SKILLS_DB), key=len, reverse=True)
