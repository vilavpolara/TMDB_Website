"""
app.py

Main Streamlit application for the TMDB Movie Recommender System.
Implements:
- Multi-page navigation
- Sidebar filters
- Movie/cards UI
- Recommendation interface
- Watchlist management
- Trending page
- Visualization page
"""

import streamlit as st
from api_handler import TMDBHandler
from recommendation import MovieRecommender
from nlp_processing import NLPProcessor, SentimentAnalyzer
from filters import apply_filters
from utils import (
    get_poster, format_genres, format_cast, format_keywords,
    format_runtime, format_rating, init_watchlist, add_to_watchlist,
    get_watchlist, remove_from_watchlist,
    plot_genre_distribution, plot_year_distribution, plot_rating_distribution,
    get_genre_distribution, get_year_distribution, get_rating_distribution
)

import random


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Movie Recommender System",
    layout="wide"
)

st.title("🎬 TMDB Movie Recommendation System")


# ============================================================
# INITIALIZE CORE OBJECTS
# ============================================================

tmdb = TMDBHandler()
nlp = NLPProcessor()
sentiment = SentimentAnalyzer()
recommender = MovieRecommender(tmdb, nlp, sentiment)
init_watchlist()


# ============================================================
# PAGE NAVIGATION
# ============================================================

page = st.sidebar.radio(
    "Navigation",
    ["🔍 Recommendation", "🔥 Trending", "⭐ Watchlist", "📊 Visualizations"]
)


# ============================================================
# SIDEBAR FILTERS (all apply globally)
# ============================================================

st.sidebar.header("Filters")

year_range = st.sidebar.slider(
    "Release Year Range",
    1900, 2025, (2000, 2025)
)

decade = st.sidebar.selectbox(
    "Decade",
    ["Any", "1980", "1990", "2000", "2010", "2020"]
)
decade = None if decade == "Any" else int(decade)

period = st.sidebar.selectbox(
    "Release Period (Quarter)",
    ["Any", "Q1", "Q2", "Q3", "Q4"]
)
period = None if period == "Any" else period

min_rating = st.sidebar.slider("Minimum Rating", 0.0, 10.0, 0.0, 0.5)
min_votes = st.sidebar.slider("Minimum Vote Count", 0, 20000, 0)

runtime_range = st.sidebar.slider(
    "Runtime (minutes)",
    0, 350, (0, 350)
)

cert = st.sidebar.selectbox(
    "Certification",
    ["Any", "G", "PG", "PG-13", "R", "NC-17"]
)
cert = None if cert == "Any" else cert

lang = st.sidebar.selectbox(
    "Primary Language",
    ["Any", "en", "fr", "es", "de", "zh", "ja", "ko"]
)
lang = None if lang == "Any" else lang

actor_name = st.sidebar.text_input("Actor Name (optional)")
director_name = st.sidebar.text_input("Director Name (optional)")
writer_name = st.sidebar.text_input("Writer Name (optional)")

genre_and = st.sidebar.multiselect(
    "Genres (AND filter)",
    ["Action", "Adventure", "Animation", "Comedy", "Crime", "Drama", "Fantasy",
     "Horror", "Romance", "Sci-Fi", "Thriller"]
)

genre_or = st.sidebar.multiselect(
    "Genres (OR filter)",
    ["Action", "Adventure", "Animation", "Comedy", "Crime", "Drama", "Fantasy",
     "Horror", "Romance", "Sci-Fi", "Thriller"]
)

# Assemble filter parameters dictionary
filter_params = {
    "year_range": year_range,
    "decade": decade,
    "period": period,
    "min_rating": min_rating,
    "min_votes": min_votes,
    "runtime_range": runtime_range,
    "certification": cert,
    "language": lang,
    "actor": actor_name,
    "director": director_name,
    "writer": writer_name,
    "genre_and": genre_and,
    "genre_or": genre_or
}


# ============================================================
# PAGE 1 — RECOMMENDATION PAGE
# ============================================================

if page == "🔍 Recommendation":

    st.subheader("Search or Select a Movie")

    col1, col2 = st.columns(2)

    with col1:
        search_query = st.text_input(
            "Natural Language Search (e.g., 'a space adventure with robots')"
        )

    with col2:
        algo = st.selectbox(
            "Recommendation Algorithm",
            ["Hybrid", "Content-Based", "Sentiment-Based"]
        )
        algo = algo.lower().replace("-", "")

    # User-provided movie search
    if search_query:
        # Use query to fetch a large set of candidates from TMDB
        results = tmdb.search_movies(search_query)
        movie_ids = [m["id"] for m in results]

        if not movie_ids:
            st.warning("No movies found for your search.")
        else:
            st.write(f"Found {len(movie_ids)} movies.")

            # Natural-language based recommendations
            recs = recommender.recommend(
                movie_ids=movie_ids,
                query=search_query,
                algo=algo,
                top_k=10,
                filter_params=filter_params
            )

            if recs:
                st.subheader("Recommended Movies")
                cols = st.columns(2)
                for idx, movie in enumerate(recs):
                    with cols[idx % 2]:
                        st.image(get_poster(movie), width=250)
                        st.write(f"### {movie['title']} ({movie['release_year']})")
                        st.write(format_genres(movie))
                        st.write(format_runtime(movie))
                        st.write(format_rating(movie))
                        st.caption(movie["explanation"])

                        if st.button(f"Add to Watchlist {movie['id']}", key=f"add_rec_{movie['id']}"):
                            add_to_watchlist(movie)
            else:
                st.warning("No recommendations available after filtering.")

    st.markdown("---")
    st.subheader("Find Similar to a Specific Movie")

    movie_title = st.text_input("Enter a movie title to find similar movies")

    if movie_title:
        search_results = tmdb.search_movies(movie_title)

        if not search_results:
            st.warning("No movies found with that name.")
        else:
            movie_options = {
                f"{m['title']} ({m.get('release_date', 'N/A')[:4]})": m["id"]
                for m in search_results
            }

            selected = st.selectbox("Select a movie", list(movie_options.keys()))
            selected_id = movie_options[selected]

            # Build a dataset from popular movies + trending + similar
            popular_ids = [m["id"] for m in tmdb.get_popular(page=1)]
            trending_ids = [m["id"] for m in tmdb.get_trending()]
            similar_ids = [m["id"] for m in tmdb.search_movies(movie_title)]

            dataset_ids = list(set(popular_ids + trending_ids + similar_ids))

            recs = recommender.recommend(
                movie_ids=dataset_ids,
                target_movie_id=selected_id,
                algo=algo,
                top_k=10,
                filter_params=filter_params
            )

            st.subheader(f"Movies Similar to {selected}")

            cols = st.columns(2)
            for idx, movie in enumerate(recs):
                with cols[idx % 2]:
                    st.image(get_poster(movie), width=250)
                    st.write(f"### {movie['title']} ({movie['release_year']})")
                    st.write(format_genres(movie))
                    st.write(format_runtime(movie))
                    st.write(format_cast(movie))
                    st.write(format_rating(movie))
                    st.caption(movie["explanation"])

                    if st.button(f"Add to Watchlist {movie['id']}", key=f"add_sim_{movie['id']}"):
                        add_to_watchlist(movie)


# ============================================================
# PAGE 2 — TRENDING PAGE
# ============================================================

elif page == "🔥 Trending":
    st.subheader("🔥 Trending Movies This Week")

    trending = tmdb.get_trending()
    if not trending:
        st.warning("Unable to load trending movies.")
    else:
        cols = st.columns(3)
        for idx, m in enumerate(trending[:30]):
            movie = tmdb.get_full_movie(m["id"])
            with cols[idx % 3]:
                st.image(get_poster(movie), width=250)
                st.write(f"### {movie.get('title')} ({movie.get('release_year')})")
                st.write(format_genres(movie))
                st.write(format_runtime(movie))
                st.write(format_rating(movie))

                if st.button(f"Add to Watchlist {movie['id']}", key=f"add_tr_{movie['id']}"):
                    add_to_watchlist(movie)


# ============================================================
# PAGE 3 — WATCHLIST PAGE
# ============================================================

elif page == "⭐ Watchlist":
    st.subheader("⭐ Your Watchlist")

    watchlist = get_watchlist()

    if not watchlist:
        st.info("Your watchlist is empty. Add movies from recommendations.")
    else:
        cols = st.columns(2)
        for idx, movie in enumerate(watchlist):
            with cols[idx % 2]:
                st.image(get_poster(movie), width=250)
                st.write(f"### {movie['title']} ({movie['release_year']})")
                st.write(format_genres(movie))
                st.write(format_runtime(movie))

                if st.button(f"Remove {movie['id']}", key=f"rm_wl_{movie['id']}"):
                    remove_from_watchlist(movie["id"])
                    st.experimental_rerun()


# ============================================================
# PAGE 4 — VISUALIZATIONS
# ============================================================

elif page == "📊 Visualizations":
    st.subheader("📊 Dataset Visualizations")

    st.write("Loading popular + trending movies for analytics…")

    movies = []
    movies.extend([tmdb.get_full_movie(m["id"]) for m in tmdb.get_popular()])
    movies.extend([tmdb.get_full_movie(m["id"]) for m in tmdb.get_trending()])

    movies = [m for m in movies if m]

    st.write(f"Loaded {len(movies)} movies.")

    genre_counts = get_genre_distribution(movies)
    year_counts = get_year_distribution(movies)
    rating_counts = get_rating_distribution(movies)

    st.markdown("### Genre Distribution")
    plot_genre_distribution(genre_counts)

    st.markdown("### Release Year Distribution")
    plot_year_distribution(year_counts)

    st.markdown("### Rating Distribution")
    plot_rating_distribution(rating_counts)
