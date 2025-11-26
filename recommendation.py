"""
recommendation.py

Implements the full recommendation engine for the TMDB Movie Recommender System.

Includes 3 methods:
1. Content-based (TF-IDF + cosine similarity)
2. Sentiment-based (VADER tone similarity)
3. Hybrid (weighted average)

Also includes:
- Natural language query recommendations
- Detailed similarity explanations
- Fully modular design
"""

from typing import List, Dict, Optional
import numpy as np

from api_handler import TMDBHandler
from nlp_processing import NLPProcessor, SentimentAnalyzer
from filters import apply_filters


# -------------------------------------------------------------
# Helper: Build similarity explanation text
# -------------------------------------------------------------

def build_similarity_explanation(
    movie_a: Dict,
    movie_b: Dict,
    content_score: float,
    sentiment_score: float,
    hybrid_score: float
) -> str:
    """
    Returns a human-readable explanation of WHY the movie was recommended.
    """

    explanation = []

    # genre overlap
    genres_a = set(movie_a.get("genres", []))
    genres_b = set(movie_b.get("genres", []))
    overlap = genres_a.intersection(genres_b)
    if overlap:
        explanation.append(f"Shares genres: {', '.join(sorted(overlap))}")

    # keyword overlap
    kw_a = set(movie_a.get("keywords", []))
    kw_b = set(movie_b.get("keywords", []))
    kw_overlap = kw_a.intersection(kw_b)
    if kw_overlap:
        explanation.append(f"Shares keywords: {', '.join(list(kw_overlap)[:5])}")

    # shared cast
    cast_a = set(movie_a.get("cast", []))
    cast_b = set(movie_b.get("cast", []))
    cast_overlap = cast_a.intersection(cast_b)
    if cast_overlap:
        explanation.append(f"Common actors: {', '.join(list(cast_overlap)[:5])}")

    # shared director
    if movie_a.get("director") and movie_b.get("director"):
        if movie_a["director"] == movie_b["director"]:
            explanation.append(f"Same director: {movie_a['director']}")

    # scores
    explanation.append(
        f"Content similarity score: {content_score:.2f}"
    )
    explanation.append(
        f"Sentiment similarity: {sentiment_score:.2f}"
    )
    explanation.append(
        f"Hybrid score (final): {hybrid_score:.2f}"
    )

    return " | ".join(explanation)


# -------------------------------------------------------------
# Recommendation Engine
# -------------------------------------------------------------

class MovieRecommender:
    """
    Provides:
    - Content-based similarity
    - Sentiment-based similarity
    - Hybrid similarity
    - Natural-language query similarity
    """

    def __init__(
        self,
        tmdb_handler: TMDBHandler,
        nlp: NLPProcessor,
        sentiment: SentimentAnalyzer
    ):
        self.tmdb = tmdb_handler
        self.nlp = nlp
        self.sentiment = sentiment

    # ---------------------------------------------------------
    # Build dataset for recommendation
    # ---------------------------------------------------------

    def _load_dataset(self, movie_ids: List[int]) -> List[Dict]:
        """
        Loads full enriched movie metadata for all movie_ids.
        Returns a list of full movie dicts.
        """
        dataset = []
        for mid in movie_ids:
            movie = self.tmdb.get_full_movie(mid)
            if movie:
                dataset.append(movie)
        return dataset

    # ---------------------------------------------------------
    # Compute sentiment vector for dataset
    # ---------------------------------------------------------

    def _compute_sentiment_vector(self, movies: List[Dict]) -> List[float]:
        scores = []
        for m in movies:
            overview = m.get("overview", "")
            scores.append(self.sentiment.sentiment_score(overview))
        return scores

    # ---------------------------------------------------------
    # CONTENT-BASED SIMILARITY
    # ---------------------------------------------------------

    def content_based(
        self,
        dataset: List[Dict],
        target_index: int
    ) -> np.ndarray:
        """
        Returns cosine similarity of target movie vs all others.
        """
        return self.nlp.compute_item_similarity(target_index)

    # ---------------------------------------------------------
    # SENTIMENT-BASED SIMILARITY
    # ---------------------------------------------------------

    def sentiment_based(
        self,
        dataset: List[Dict],
        target_index: int
    ) -> np.ndarray:
        """
        Computes similarity based on:
            1 - |sentiment difference|
        """
        target_overview = dataset[target_index].get("overview", "")
        target_score = self.sentiment.sentiment_score(target_overview)

        all_scores = self._compute_sentiment_vector(dataset)

        return self.sentiment.compute_sentiment_similarity(target_score, all_scores)

    # ---------------------------------------------------------
    # HYBRID SIMILARITY
    # ---------------------------------------------------------

    def hybrid(
        self,
        dataset: List[Dict],
        target_index: int,
        weight_content: float = 0.7,
        weight_sentiment: float = 0.3
    ) -> np.ndarray:
        """
        Weighted combination of content + sentiment.
        """
        content_sim = self.content_based(dataset, target_index)
        sentiment_sim = self.sentiment_based(dataset, target_index)

        # Normalized combination
        hybrid_score = (
            weight_content * content_sim +
            weight_sentiment * sentiment_sim
        )

        return hybrid_score

    # ---------------------------------------------------------
    # Natural-Language Query Similarity
    # ---------------------------------------------------------

    def query_similarity(
        self,
        query: str,
        dataset: List[Dict]
    ) -> np.ndarray:
        """
        Returns similarity of user query vs all movies.
        """
        vector = self.nlp.vectorize_query(query)
        if vector is None:
            return np.zeros(len(dataset))

        return self.nlp.compute_similarity(vector)

    # ---------------------------------------------------------
    # MAIN ENTRYPOINT — GET RECOMMENDATIONS
    # ---------------------------------------------------------

    def recommend(
        self,
        movie_ids: List[int],
        target_movie_id: Optional[int] = None,
        query: Optional[str] = None,
        algo: str = "hybrid",
        top_k: int = 10,
        filter_params: Optional[dict] = None
    ) -> List[Dict]:
        """
        Computes top-N recommendations using:
        - movie-to-movie similarity  (if target_movie_id provided)
        - natural-language vector search (if query provided)

        Args:
            movie_ids (List[int]): The dataset of movie IDs
            target_movie_id (int): If provided, find similar movies
            query (str): If provided, find movies similar to text query
            algo (str): "content", "sentiment", "hybrid"
            filter_params (dict): Filters to apply
        """

        # Build dataset
        dataset = self._load_dataset(movie_ids)
        if not dataset:
            return []

        # Fit TF-IDF on overview text
        overviews = [m.get("overview", "") for m in dataset]
        self.nlp.fit(overviews)

        # Determine target index
        target_index = None
        if target_movie_id:
            for i, m in enumerate(dataset):
                if m.get("id") == target_movie_id:
                    target_index = i
                    break

        # --------------------------
        # Similarity computation
        # --------------------------

        if target_index is not None:
            # Movie → movie similarity

            if algo == "content":
                scores = self.content_based(dataset, target_index)
            elif algo == "sentiment":
                scores = self.sentiment_based(dataset, target_index)
            else:
                scores = self.hybrid(dataset, target_index)

        elif query:
            # Query → movie similarity
            scores = self.query_similarity(query, dataset)
        else:
            return []

        # --------------------------
        # Ranking
        # --------------------------

        indices = np.argsort(scores)[::-1]  # high → low

        results = []
        for idx in indices[:top_k + 10]:  # get extras to allow filtering
            movie = dataset[idx]
            if target_index is not None and idx == target_index:
                continue

            content_sim = self.content_based(dataset, target_index)[idx] if target_index is not None else 0
            sentiment_sim = self.sentiment_based(dataset, target_index)[idx] if target_index is not None else 0
            hybrid_sim = self.hybrid(dataset, target_index)[idx] if target_index is not None else scores[idx]

            explanation = build_similarity_explanation(
                dataset[target_index] if target_index is not None else dataset[idx],
                movie,
                content_sim,
                sentiment_sim,
                hybrid_sim
            )

            movie_out = {
                **movie,
                "similarity_score": float(scores[idx]),
                "explanation": explanation
            }
            results.append(movie_out)

        # --------------------------
        # Apply final filters
        # --------------------------

        if filter_params:
            results = apply_filters(results, **filter_params)

        # Only return top_k after filtering
        return results[:top_k]
