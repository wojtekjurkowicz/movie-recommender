from utils import fix_title
import pandas as pd
from recommender import compute_cosine_similarity, get_content_based_recommendations, get_group_recommendations


def test_fix_title():
    assert fix_title("Matrix, The") == "The Matrix"
    assert fix_title("Godfather, The (1972)") == "The Godfather (1972)"
    assert fix_title("Inception") == "Inception"


def test_content_based():
    df = pd.DataFrame({
        "fixed_title": ["Movie A", "Movie B", "Movie C"],
        "genres": ["Action|Sci-Fi", "Action|Sci-Fi", "Comedy"]
    })
    sim = compute_cosine_similarity(df)
    recs = get_content_based_recommendations("Movie A", df, sim)
    assert isinstance(recs, pd.Series)
    assert "Movie A" not in recs.values


def test_group_recommendations():
    df = pd.DataFrame({
        "fixed_title": ["A", "B", "C", "D"],
        "genres": ["Action", "Action", "Comedy", "Action|Comedy"]
    })
    sim = compute_cosine_similarity(df)
    recs = get_group_recommendations(["A", "B"], df, sim)
    assert isinstance(recs, pd.Series)
    assert all(title not in ["A", "B"] for title in recs.values)


def test_content_based_invalid_title():
    df = pd.DataFrame({
        "fixed_title": ["X"], "genres": ["Drama"]
    })
    sim = compute_cosine_similarity(df)
    try:
        get_content_based_recommendations("Z", df, sim)
    except IndexError:
        assert True
    else:
        assert False
