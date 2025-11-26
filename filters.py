"""
filters.py

Contains ALL filtering logic for the Movie Recommendation System.

The goal is to provide modular, combinable filters that operate on
structured movie dictionaries returned by api_handler.py.

Each filter receives:
    - a list of movie dicts
    - one or more filter parameters

Each filter returns:
    - a filtered list of movies

All filters must be fully composable.
"""

from typing import List, Dict, Optional


# -------------------------------------------------------
# Helper: Safe value extraction
# -------------------------------------------------------

def get(movie: Dict, key: str, default=None):
    """Shortcut for safely extracting data from movie dicts."""
    return movie.get(key, default)


# -------------------------------------------------------
# TEMPORAL FILTERS
# -------------------------------------------------------

def filter_by_year_range(movies: List[Dict], year_range: tuple) -> List[Dict]:
    min_year, max_year = year_range
    return [
        m for m in movies
        if get(m, "release_year") and min_year <= m["release_year"] <= max_year
    ]


def filter_by_decade(movies: List[Dict], decade: Optional[int]) -> List[Dict]:
    """
    decade = 1990, 2000, 2010, etc.
    """
    if not decade:
        return movies

    return [
        m for m in movies
        if get(m, "release_year") and decade <= m["release_year"] < decade + 10
    ]


def filter_by_release_period(movies: List[Dict], period: Optional[str]) -> List[Dict]:
    """
    period ∈ {"Q1", "Q2", "Q3", "Q4"}
    """
    if not period:
        return movies

    quarter_map = {
        "Q1": ("01", "02", "03"),
        "Q2": ("04", "05", "06"),
        "Q3": ("07", "08", "09"),
        "Q4": ("10", "11", "12"),
    }

    months = quarter_map.get(period)
    if not months:
        return movies

    results = []
    for m in movies:
        date = get(m, "release_date") or ""
        if len(date) >= 7:
            month = date[5:7]
            if month in months:
                results.append(m)
    return results


# -------------------------------------------------------
# QUALITY FILTERS
# -------------------------------------------------------

def filter_by_min_rating(movies: List[Dict], min_rating: float) -> List[Dict]:
    return [
        m for m in movies
        if get(m, "vote_average", 0) >= min_rating
    ]


def filter_by_min_vote_count(movies: List[Dict], min_votes: int) -> List[Dict]:
    return [
        m for m in movies
        if get(m, "vote_count", 0) >= min_votes
    ]


# -------------------------------------------------------
# CONTENT FILTERS
# -------------------------------------------------------

def filter_by_runtime(movies: List[Dict], runtime_range: tuple) -> List[Dict]:
    min_r, max_r = runtime_range
    return [
        m for m in movies
        if get(m, "runtime") and min_r <= m["runtime"] <= max_r
    ]


def filter_by_certification(movies: List[Dict], cert: Optional[str]) -> List[Dict]:
    if not cert or cert == "Any":
        return movies

    return [
        m for m in movies
        if get(m, "certification") == cert
    ]


def filter_by_language(movies: List[Dict], language: Optional[str]) -> List[Dict]:
    """
    Language codes are ISO-639-1 (ex: "en", "fr", "zh").
    """
    if not language or language == "Any":
        return movies

    return [
        m for m in movies
        if get(m, "language") == language
    ]


# -------------------------------------------------------
# PERSONNEL FILTERS
# -------------------------------------------------------

def filter_by_actor(movies: List[Dict], actor_name: Optional[str]) -> List[Dict]:
    if not actor_name:
        return movies

    actor_name = actor_name.lower()

    return [
        m for m in movies
        if any(actor_name in c.lower() for c in get(m, "cast", []))
    ]


def filter_by_director(movies: List[Dict], director_name: Optional[str]) -> List[Dict]:
    if not director_name:
        return movies

    director_name = director_name.lower()

    return [
        m for m in movies
        if get(m, "director") and director_name in m["director"].lower()
    ]


def filter_by_writer(movies: List[Dict], writer_name: Optional[str]) -> List[Dict]:
    if not writer_name:
        return movies

    writer_name = writer_name.lower()

    return [
        m for m in movies
        if any(writer_name in w.lower() for w in get(m, "writers", []))
    ]


# -------------------------------------------------------
# GENRE FILTERS
# -------------------------------------------------------

def filter_genre_AND(movies: List[Dict], genres: List[str]) -> List[Dict]:
    """
    Only keep movies that contain *all* selected genres.
    """
    if not genres:
        return movies

    genres = set([g.lower() for g in genres])

    return [
        m for m in movies
        if genres.issubset(set([x.lower() for x in get(m, "genres", [])]))
    ]


def filter_genre_OR(movies: List[Dict], genres: List[str]) -> List[Dict]:
    """
    Keep movies that contain *any* of the selected genres.
    """
    if not genres:
        return movies

    genres = set([g.lower() for g in genres])

    results = []
    for m in movies:
        movie_genres = set([x.lower() for x in get(m, "genres", [])])
        if movie_genres.intersection(genres):
            results.append(m)
    return results


# -------------------------------------------------------
# MASTER FILTER PIPELINE
# -------------------------------------------------------

def apply_filters(
    movies: List[Dict],
    year_range: tuple = (1900, 2050),
    decade: Optional[int] = None,
    period: Optional[str] = None,
    min_rating: float = 0.0,
    min_votes: int = 0,
    runtime_range: tuple = (0, 500),
    certification: Optional[str] = None,
    language: Optional[str] = None,
    actor: Optional[str] = None,
    director: Optional[str] = None,
    writer: Optional[str] = None,
    genre_and: Optional[List[str]] = None,
    genre_or: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Applies ALL filters in a fixed order for consistency.
    """
    filtered = movies

    # Temporal filters
    filtered = filter_by_year_range(filtered, year_range)
    filtered = filter_by_decade(filtered, decade)
    filtered = filter_by_release_period(filtered, period)

    # Quality filters
    filtered = filter_by_min_rating(filtered, min_rating)
    filtered = filter_by_min_vote_count(filtered, min_votes)

    # Content filters
    filtered = filter_by_runtime(filtered, runtime_range)
    filtered = filter_by_certification(filtered, certification)
    filtered = filter_by_language(filtered, language)

    # Personnel filters
    filtered = filter_by_actor(filtered, actor)
    filtered = filter_by_director(filtered, director)
    filtered = filter_by_writer(filtered, writer)

    # Genre filters
    filtered = filter_genre_AND(filtered, genre_and or [])
    filtered = filter_genre_OR(filtered, genre_or or [])

    return filtered
