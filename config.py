"""
config.py

Central configuration for the TMDB Movie Recommender System.
Handles environment variables, constants, and global settings.
"""

import os
from dotenv import load_dotenv

# Load .env variables if running locally
load_dotenv()

def get_tmdb_api_key() -> str:
    """
    Fetch the TMDB API key either from:
    - Streamlit secrets (Cloud deployment)
    - Environment variables (.env for local dev)

    Returns:
        str: The TMDB API key.

    Raises:
        ValueError: If no API key is found.
    """
    # Streamlit Cloud secrets
    try:
        import streamlit as st
        if "TMDB_API_KEY" in st.secrets:
            return st.secrets["TMDB_API_KEY"]
    except Exception:
        pass

    # Local environment variable
    key = os.getenv("TMDB_API_KEY")
    if key:
        return key

    raise ValueError(
        "TMDB API key not found. Add it to Streamlit secrets "
        "or export TMDB_API_KEY in your environment."
    )


# TMDB URL constants
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

# Caching duration in seconds
CACHE_TTL = 60 * 60  # 1 hour
