# 🎬 Movie Recommender System

A movie recommendation web app built in Python using **Streamlit**, **scikit-learn**, and **Surprise**.  
It suggests movies based on either **genre similarity** (content-based filtering) or **user ratings** (collaborative filtering with SVD).

---

## Features

- **Searchable movie dropdown**
- **Three recommendation modes**:
  - Content-based (genre similarity)
  - Collaborative (user-based with SVD)
  - Top-rated (globally highest-rated movies)
- **Filter results** by:
  - **Genre**
  - **Year**
- Built with an interactive **Streamlit interface**

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

````

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
pip install -r requirements.txt
```

Alternatively, you can recreate the environment using Conda:
```bash
conda env create -f environment.yml
conda activate movie
```

3. **Download MovieLens dataset**

Place the [MovieLens 100K dataset](https://grouplens.org/datasets/movielens/) inside a folder called `ml-latest-small/` in the root of the project.

---

## Run the App

```bash
streamlit run main.py
```

Then open `http://localhost:8501` in your browser.

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