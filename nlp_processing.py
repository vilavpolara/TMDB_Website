"""
nlp_processing.py

Contains all natural language processing utilities:
- Text cleaning (lowercase, punctuation, stopwords)
- TF-IDF vectorization
- Cosine similarity calculations
- VADER sentiment scoring
- Natural-language query vectorization

This module is used by recommendation.py to generate:
- content similarity rankings
- sentiment-based rankings
- hybrid similarity scores
"""

import re
import nltk
import numpy as np
from typing import List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure required NLTK resources
nltk.download("stopwords", quiet=True)
nltk.download("vader_lexicon", quiet=True)

from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer


# ------------------------------------------------
# Helper: Clean and Normalize Text
# ------------------------------------------------

def clean_text(text: str) -> str:
    """
    Cleans text by:
    - Lowercasing
    - Removing punctuation
    - Removing non-alphabet characters
    - Removing stopwords

    Returns:
        str: Cleaned text
    """
    if not text:
        return ""

    # Lowercase
    text = text.lower()

    # Remove punctuation and non-letters
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Remove stopwords
    sw = set(stopwords.words("english"))
    tokens = [w for w in text.split() if w not in sw]

    return " ".join(tokens)


# ------------------------------------------------
# NLPProcessor: Handles TF-IDF and Similarity
# ------------------------------------------------

class NLPProcessor:
    """
    Handles all text-based feature extraction:
    - Cleaning
    - TF-IDF vectorization
    - Cosine similarity
    - Natural-language query vectorization
    """

    def __init__(self):
        self.vectorizer = None
        self.tfidf_matrix = None
        self.text_corpus = []  # store cleaned overviews

    # --------------------------------------------
    # Build TF-IDF Matrix
    # --------------------------------------------

    def fit(self, movie_descriptions: List[str]):
        """
        Builds TF-IDF matrix from a list of movie plot summaries.

        Args:
            movie_descriptions (List[str]): Cleaned plot overviews
        """
        cleaned_docs = [clean_text(d) for d in movie_descriptions]
        self.text_corpus = cleaned_docs

        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words="english"
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(cleaned_docs)

    # --------------------------------------------
    # Get TF-IDF vector for a specific movie index
    # --------------------------------------------

    def get_vector(self, idx: int):
        if self.tfidf_matrix is None:
            return None
        return self.tfidf_matrix[idx]

    # --------------------------------------------
    # Vectorize a natural language user query
    # --------------------------------------------

    def vectorize_query(self, query: str):
        """
        Converts a user natural-language input into a TF-IDF vector.
        """
        if not self.vectorizer:
            return None

        cleaned = clean_text(query)
        return self.vectorizer.transform([cleaned])

    # --------------------------------------------
    # Compute cosine similarity between:
    #   - a movie and all others (content-based)
    #   - a user query and all movies
    # --------------------------------------------

    def compute_similarity(self, vector) -> np.ndarray:
        """
        Computes cosine similarity between a vector and the full TF-IDF matrix.

        Args:
            vector: 1×N TF-IDF vector

        Returns:
            np.ndarray: 1-D similarity array
        """
        if self.tfidf_matrix is None:
            return np.array([])

        sim_matrix = cosine_similarity(vector, self.tfidf_matrix)
        return sim_matrix.flatten()

    # --------------------------------------------
    # Batch similarity (movie-to-movie)
    # --------------------------------------------

    def compute_item_similarity(self, idx: int) -> np.ndarray:
        """
        Computes similarity scores for one movie against all others.
        """
        vec = self.get_vector(idx)
        if vec is None:
            return np.array([])
        return self.compute_similarity(vec)


# ------------------------------------------------
# Sentiment Analyzer (Optional Feature)
# ------------------------------------------------

class SentimentAnalyzer:
    """
    Computes VADER sentiment polarity for plot summaries and queries.
    """

    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def sentiment_score(self, text: str) -> float:
        """
        Returns a compound sentiment score between −1 and +1.
        Higher = more positive “tone” in the summary.
        """
        if not text:
            return 0.0

        scores = self.analyzer.polarity_scores(text)
        return scores.get("compound", 0.0)

    def compute_sentiment_similarity(
        self, 
        target_score: float, 
        all_scores: List[float]
    ) -> np.ndarray:
        """
        Computes sentiment similarity as:
        1 - |difference|

        Returns:
            np.ndarray: similarity values in [0, 1]
        """
        diffs = [abs(target_score - s) for s in all_scores]
        sim = np.array([1 - d for d in diffs])
        return sim
