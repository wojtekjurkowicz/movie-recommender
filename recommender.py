import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split


def compute_cosine_similarity(movies):
    """
    Computes cosine similarity matrix based on genres using TF-IDF.
    """
    tfidf = TfidfVectorizer(token_pattern=r"[\w\-]+")
    tfidf_matrix = tfidf.fit_transform(movies['genres'])
    return cosine_similarity(tfidf_matrix, tfidf_matrix)


def train_svd_model(ratings):
    """
    Trains a collaborative filtering model using SVD.
    """
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    trainset, _ = train_test_split(data, test_size=0.25)
    algo = SVD()
    algo.fit(trainset)
    return algo


def get_content_based_recommendations(fixed_title, df, cosine_sim):
    """
    Returns top 5 movies similar in genre to the selected one.
    """
    idx = df[df['fixed_title'] == fixed_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return df.iloc[movie_indices]['fixed_title']


def get_svd_recommendations(user_id, selected_title, df, model):
    """
    Returns top 5 recommendations using collaborative filtering.
    """
    selected_movie_id = df[df['fixed_title'] == selected_title]['movieId'].values[0]
    predictions = [(row['movieId'], model.predict(user_id, row['movieId']).est)
                   for _, row in df.iterrows()]
    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
    top_ids = [pid for pid, _ in predictions if pid != selected_movie_id][:5]
    return df[df['movieId'].isin(top_ids)][['fixed_title']].assign(predicted_rating=[r for _, r in predictions if _ in top_ids])

