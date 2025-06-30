![Tests](https://github.com/wojtekjurkowicz/movie-recommender/actions/workflows/python-tests.yml/badge.svg)

# 🎬 Movie Recommender System

A movie recommendation web app built in Python using **Streamlit**, **scikit-learn**, and **Surprise**.  
It suggests movies based on either **genre similarity** (content-based filtering) or **user ratings** (collaborative filtering with SVD).

---

## Features

- **Searchable movie dropdown**
- **Three recommendation modes**, auto-adapting to single or multiple selections:
  - Content-based (genre similarity)
  - Collaborative (user-based with SVD)
  - Top-rated (globally highest-rated movies)
- **Filter results** by:
  - **Genre**
  - **Year**
- Built with an interactive **Streamlit interface**
- **Smart multi-select**: if you choose more than one movie, recommendations are based on group similarity

---

## Project Structure

```

movie_recommender/
├── main.py               # Streamlit frontend
├── recommender.py        # Recommendation logic
├── utils.py              # Helper functions (e.g. title fixing)
├── ml-latest-small/      # MovieLens dataset
│   ├── movies.csv
│   └── ratings.csv
└── README.md

```

---

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/wojtekjurkowicz/movie-recommender.git
cd movie-recommender
````

2. **Set up the environment**

```bash
conda create -n movie python=3.10
conda activate movie
pip install "numpy<2"  # required by scikit-surprise
pip install -r requirements.txt
```

Alternatively, you can recreate the environment using Conda:
```bash
conda env create -f environment.yml
conda activate movie
```

3. **Download MovieLens dataset**

Download the [MovieLens Latest Small dataset](https://grouplens.org/datasets/movielens/) and place its CSV files in a folder named `ml-latest-small/` in the root.

---

## Run the App

```bash
streamlit run main.py
```

Then open `http://localhost:8501` in your browser.

---

## Running Tests

Unit tests are provided for content-based and group recommendation logic.

```bash
pytest
````

Tests are also automatically run via [GitHub Actions](https://github.com/wojtekjurkowicz/movie-recommender/actions).

---

## Example Algorithms

### Content-Based Filtering

* Genres vectorized with TF-IDF
* Cosine similarity between movie genres

### Collaborative Filtering

* Surprise `SVD` model trained on user–movie ratings
* Predicts ratings for unseen movies, returns top predictions

---

## License

This project is licensed under the MIT License.