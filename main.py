import streamlit as st
import pandas as pd
from recommender import (
    compute_cosine_similarity,
    train_svd_model,
    get_content_based_recommendations,
    get_svd_recommendations
)
from utils import fix_title

# Load data
movies = pd.read_csv("ml-latest-small/movies.csv")
ratings = pd.read_csv("ml-latest-small/ratings.csv")

avg_ratings = ratings.groupby("movieId")["rating"].mean().reset_index()
movies_with_ratings = pd.merge(movies, avg_ratings, on="movieId")

# Preprocess titles
movies['fixed_title'] = movies['title'].apply(fix_title)
movies['year'] = movies['title'].str.extract(r"\((\d{4})\)").astype(float)

# Add missing cols to movies_with_ratings
movies_with_ratings['fixed_title'] = movies['fixed_title']
movies_with_ratings['year'] = movies['year']

# Precompute similarity and train model
cosine_sim = compute_cosine_similarity(movies)
algo = train_svd_model(ratings)

# Streamlit UI
st.title("Movie Recommender")

selected_title = st.selectbox("Select a movie:", sorted(movies['fixed_title'].tolist()))
mode = st.radio("Recommendation type:", ("Genre-based", "User-based", "Top Rated"))

# Genre filter
all_genres = sorted(set(genre for genres in movies['genres'].str.split('|') for genre in genres))
selected_genres = st.multiselect("Filter by genre:", all_genres)

# Year filter
min_year = int(movies['year'].min())
max_year = int(movies['year'].max())
selected_year = st.slider("Filter by year:", min_year, max_year, (min_year, max_year))

if st.button("Show recommendations"):
    if mode == "Genre-based":
        recs = get_content_based_recommendations(selected_title, movies, cosine_sim)
        rec_df = movies[movies['fixed_title'].isin(recs)]
    elif mode == "User-based":
        recs = get_svd_recommendations(user_id=1, selected_title=selected_title, df=movies, model=algo)
        rec_df = movies[movies['fixed_title'].isin(recs)]
    else:  # Top Rated
        rec_df = movies_with_ratings.copy()
        rec_df = rec_df.sort_values("rating", ascending=False).head(10)
    else_used = True

    # Apply genre filter
    if selected_genres:
        rec_df = rec_df[rec_df['genres'].apply(lambda g: any(genre in g.split('|') for genre in selected_genres))]

    # Apply year filter
    rec_df = rec_df[
        (rec_df['year'] >= selected_year[0]) &
        (rec_df['year'] <= selected_year[1])
    ]

    st.subheader("Recommended Movies:")
    if rec_df.empty:
        st.write("No recommendations match the selected filters.")
    else:
        if 'predicted_rating' in rec_df.columns:
            for title, rating in zip(rec_df['fixed_title'], rec_df['predicted_rating']):
                st.write(f"- {title} ({rating:.2f})")
        else:
            for title in rec_df['fixed_title']:
                st.write(f"- {title}")
