"""
api_handler.py

Handles all communication with the TMDB API:
- Searching movies
- Fetching popular/trending titles
- Getting detailed metadata (cast, crew, runtime, etc.)
- Extracting keywords, genres, certifications
- Caching to reduce API calls
"""

import requests
import functools
from typing import Dict, List, Any, Optional
from config import get_tmdb_api_key, TMDB_BASE_URL, TMDB_IMAGE_BASE, CACHE_TTL
import time


# ------------------------------
# Simple Time-Based Cache Decorator
# ------------------------------

def timed_cache(ttl: int):
    """
    Simple caching decorator that stores results for `ttl` seconds.
    """
    def decorator(func):
        cache = {}

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            now = time.time()

            # Return cached entry if it exists and is fresh
            if key in cache:
                timestamp, value = cache[key]
                if now - timestamp < ttl:
                    return value

            # Compute fresh
            result = func(*args, **kwargs)
            cache[key] = (now, result)
            return result

        return wrapper
    return decorator


# ------------------------------
# TMDB API Handler Class
# ------------------------------

class TMDBHandler:
    """
    Handles TMDB API requests with:
    - structured returns
    - graceful fallback for missing fields
    - centralized HTTP calls
    """

    def __init__(self):
        self.api_key = get_tmdb_api_key()
        self.session = requests.Session()

    # ------------------------------
    # INTERNAL REQUEST HANDLER
    # ------------------------------

    def _request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """
        Sends GET request to TMDB API.
        Automatically injects API key and handles failures gracefully.
        """
        url = f"{TMDB_BASE_URL}/{endpoint}"
        params = params or {}
        params["api_key"] = self.api_key
        params["language"] = "en-US"

        try:
            r = self.session.get(url, params=params, timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception:
            return None

    # ------------------------------
    # BASIC ENDPOINTS
    # ------------------------------

    @timed_cache(CACHE_TTL)
    def search_movies(self, query: str, page: int = 1) -> List[Dict]:
        data = self._request("search/movie", {"query": query, "page": page})
        return data.get("results", []) if data else []

    @timed_cache(CACHE_TTL)
    def get_popular(self, page: int = 1) -> List[Dict]:
        data = self._request("movie/popular", {"page": page})
        return data.get("results", []) if data else []

    @timed_cache(CACHE_TTL)
    def get_trending(self, media_type: str = "movie", time_window: str = "week") -> List[Dict]:
        data = self._request(f"trending/{media_type}/{time_window}")
        return data.get("results", []) if data else []

    # ------------------------------
    # DETAILED MOVIE METADATA
    # ------------------------------

    @timed_cache(CACHE_TTL)
    def get_movie_details(self, movie_id: int) -> Dict:
        """
        Fetches metadata: title, overview, genres, runtime, year, rating, poster
        """
        data = self._request(f"movie/{movie_id}")
        if not data:
            return {}

        # Graceful parsing
        return {
            "id": data.get("id"),
            "title": data.get("title"),
            "overview": data.get("overview") or "",
            "genres": [g["name"] for g in data.get("genres", [])],
            "runtime": data.get("runtime") or 0,
            "release_year": int(data["release_date"][:4]) if data.get("release_date") else None,
            "vote_average": data.get("vote_average") or 0.0,
            "vote_count": data.get("vote_count") or 0,
            "poster_url": f"{TMDB_IMAGE_BASE}{data['poster_path']}" if data.get("poster_path") else None,
            "language": data.get("original_language"),
        }

    @timed_cache(CACHE_TTL)
    def get_keywords(self, movie_id: int) -> List[str]:
        data = self._request(f"movie/{movie_id}/keywords")
        if not data:
            return []
        return [kw["name"] for kw in data.get("keywords", [])]

    @timed_cache(CACHE_TTL)
    def get_credits(self, movie_id: int) -> Dict[str, Any]:
        """
        Returns cast + crew including directors and writers.
        """
        data = self._request(f"movie/{movie_id}/credits")
        if not data:
            return {"cast": [], "director": None, "writers": []}

        cast = [c["name"] for c in data.get("cast", [])[:10]]  # Top billed

        # Crew extraction
        directors = [c["name"] for c in data.get("crew", []) if c.get("job") == "Director"]
        writers = [c["name"] for c in data.get("crew", []) if c.get("job") in ("Writer", "Screenplay")]

        return {
            "cast": cast,
            "director": directors[0] if directors else None,
            "writers": writers,
        }

    @timed_cache(CACHE_TTL)
    def get_certification(self, movie_id: int) -> Optional[str]:
        """
        Returns the movie MPAA certification (e.g., PG-13).
        """
        data = self._request(f"movie/{movie_id}/release_dates")
        if not data:
            return None

        try:
            results = data.get("results", [])
            for country in results:
                if country.get("iso_3166_1") == "US":
                    releases = country.get("release_dates", [])
                    if releases:
                        return releases[0].get("certification")
        except Exception:
            return None

        return None

    # ------------------------------
    # AGGREGATED FULL MOVIE OBJECT
    # ------------------------------

    def get_full_movie(self, movie_id: int) -> Dict[str, Any]:
        """
        Returns a complete enriched movie dictionary combining:
        - Basic details
        - Keywords
        - Cast / director / writer
        - Certification
        """
        base = self.get_movie_details(movie_id)
        if not base:
            return {}

        credits = self.get_credits(movie_id)
        keywords = self.get_keywords(movie_id)
        cert = self.get_certification(movie_id)

        return {
            **base,
            "keywords": keywords,
            "cast": credits["cast"],
            "director": credits["director"],
            "writers": credits["writers"],
            "certification": cert,
        }
