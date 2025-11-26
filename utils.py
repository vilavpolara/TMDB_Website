"""
utils.py

Utility/helper functions for the Movie Recommendation System.
Includes:
- Watchlist management
- Visualization utilities
- Metadata formatting
- Poster-safe loading
- Dataset analytics helpers
"""

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any
from collections import Counter


# ============================================================
# WATCHLIST SYSTEM
# ============================================================

def init_watchlist():
    """Initialize watchlist in Streamlit session state."""
    if "watchlist" not in st.session_state:
        st.session_state["watchlist"] = []


def add_to_watchlist(movie: Dict):
    """Add a movie to the watchlist (prevent duplicates)."""
    init_watchlist()
    if movie["id"] not in [m["id"] for m in st.session_state["watchlist"]]:
        st.session_state["watchlist"].append(movie)


def remove_from_watchlist(movie_id: int):
    """Remove movie by ID."""
    init_watchlist()
    st.session_state["watchlist"] = [
        m for m in st.session_state["watchlist"]
        if m["id"] != movie_id
    ]


def get_watchlist() -> List[Dict]:
    """Return the current user's watchlist."""
    init_watchlist()
    return st.session_state["watchlist"]


# ============================================================
# POSTER / IMAGE UTILITIES
# ============================================================

def get_poster(movie: Dict) -> str:
    """
    Returns poster URL or fallback image.
    """
    if movie.get("poster_url"):
        return movie["poster_url"]
    return "https://via.placeholder.com/300x450?text=No+Image"


# ============================================================
# METADATA UTILITIES
# ============================================================

def format_genres(movie: Dict) -> str:
    return ", ".join(movie.get("genres", [])) or "N/A"


def format_cast(movie: Dict) -> str:
    cast = movie.get("cast", [])
    if not cast:
        return "N/A"
    return ", ".join(cast[:5])


def format_keywords(movie: Dict) -> str:
    kw = movie.get("keywords", [])
    return ", ".join(kw[:5]) if kw else "N/A"


def format_director(movie: Dict) -> str:
    return movie.get("director", "N/A")


def format_runtime(movie: Dict) -> str:
    runtime = movie.get("runtime")
    if runtime:
        return f"{runtime} min"
    return "N/A"


def format_rating(movie: Dict) -> str:
    return f"{movie.get('vote_average', 0):.1f} / 10"


# ============================================================
# DATASET ANALYSIS (VISUALIZATION HELPERS)
# ============================================================

def get_genre_distribution(movies: List[Dict]) -> Dict[str, int]:
    """Return counts of each genre from the dataset."""
    counter = Counter()
    for m in movies:
        counter.update(m.get("genres", []))
    return dict(counter)


def get_year_distribution(movies: List[Dict]) -> Dict[int, int]:
    """Return histogram of release years."""
    years = [m.get("release_year") for m in movies if m.get("release_year")]
    return Counter(years)


def get_rating_distribution(movies: List[Dict]) -> Dict[int, int]:
    """Return histogram of rating frequency."""
    ratings = [int(m.get("vote_average", 0)) for m in movies]
    return Counter(ratings)


# ============================================================
# PLOTTING UTILITIES
# ============================================================

def plot_genre_distribution(genre_counts: Dict[str, int]):
    """Bar chart for genres."""
    if not genre_counts:
        st.write("No data available for genre distribution.")
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    genres = list(genre_counts.keys())
    values = list(genre_counts.values())
    ax.bar(genres, values)
    ax.set_xlabel("Genre")
    ax.set_ylabel("Count")
    ax.set_title("Genre Popularity Distribution")
    plt.xticks(rotation=45)
    st.pyplot(fig)


def plot_year_distribution(year_counts: Dict[int, int]):
    """Histogram of years."""
    if not year_counts:
        st.write("No data available for year distribution.")
        return

    years = list(year_counts.keys())
    counts = list(year_counts.values())

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(years, counts)
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Movies")
    ax.set_title("Release Year Distribution")
    plt.xticks(rotation=45)
    st.pyplot(fig)


def plot_rating_distribution(rating_counts: Dict[int, int]):
    """Histogram of rating distribution."""
    if not rating_counts:
        st.write("No data available for rating distribution.")
        return

    ratings = list(rating_counts.keys())
    counts = list(rating_counts.values())

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(ratings, counts)
    ax.set_xlabel("Rating (Integer)")
    ax.set_ylabel("Frequency")
    ax.set_title("Rating Distribution")
    plt.xticks(ratings)
    st.pyplot(fig)


# ============================================================
# SORTING / DEDUPLICATION UTILITIES
# ============================================================

def sort_by_similarity(movies: List[Dict]) -> List[Dict]:
    """Sort movies descending by similarity score."""
    return sorted(movies, key=lambda m: m.get("similarity_score", 0), reverse=True)


def dedupe_movies(movies: List[Dict]) -> List[Dict]:
    """Remove duplicates based on movie ID."""
    seen = set()
    result = []
    for m in movies:
        mid = m.get("id")
        if mid and mid not in seen:
            seen.add(mid)
            result.append(m)
    return result
