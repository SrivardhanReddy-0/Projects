import ast
import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


print("=" * 60)
print("MOVIE RECOMMENDATION SYSTEM")
print("=" * 60)


# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

movies_path = "data/tmdb_5000_movies.csv"
credits_path = "data/tmdb_5000_credits.csv"

if not os.path.exists(movies_path):
    raise FileNotFoundError(
        f"Movie dataset not found: {movies_path}"
    )

if not os.path.exists(credits_path):
    raise FileNotFoundError(
        f"Credits dataset not found: {credits_path}"
    )


movies = pd.read_csv(movies_path)
credits = pd.read_csv(credits_path)

print("\n--- Dataset Loaded ---")
print("Movies:", movies.shape)
print("Credits:", credits.shape)


# --------------------------------------------------
# 2. MERGE DATASETS
# --------------------------------------------------

credits = credits.rename(columns={"movie_id": "id"})

movies = movies.merge(
    credits[["id", "cast", "crew"]],
    on="id"
)

print("\n--- Merged Dataset ---")
print(movies.shape)


# --------------------------------------------------
# 3. SELECT IMPORTANT FEATURES
# --------------------------------------------------

movies = movies[
    [
        "id",
        "title",
        "overview",
        "genres",
        "keywords",
        "cast",
        "crew"
    ]
].copy()


# --------------------------------------------------
# 4. HANDLE MISSING VALUES
# --------------------------------------------------

movies["overview"] = movies["overview"].fillna("")
movies["genres"] = movies["genres"].fillna("[]")
movies["keywords"] = movies["keywords"].fillna("[]")
movies["cast"] = movies["cast"].fillna("[]")
movies["crew"] = movies["crew"].fillna("[]")


# --------------------------------------------------
# 5. CONVERT JSON-LIKE COLUMNS
# --------------------------------------------------

def convert_names(value):
    try:
        data = ast.literal_eval(value)

        return " ".join(
            item["name"].replace(" ", "").lower()
            for item in data
            if "name" in item
        )

    except (ValueError, SyntaxError, TypeError):
        return ""


movies["genres"] = movies["genres"].apply(convert_names)
movies["keywords"] = movies["keywords"].apply(convert_names)


# --------------------------------------------------
# 6. EXTRACT TOP CAST
# --------------------------------------------------

def convert_cast(value):
    try:
        data = ast.literal_eval(value)

        return " ".join(
            item["name"].replace(" ", "").lower()
            for item in data[:3]
            if "name" in item
        )

    except (ValueError, SyntaxError, TypeError):
        return ""


movies["cast"] = movies["cast"].apply(convert_cast)


# --------------------------------------------------
# 7. EXTRACT DIRECTOR
# --------------------------------------------------

def get_director(value):
    try:
        data = ast.literal_eval(value)

        for item in data:
            if item.get("job") == "Director":
                return item["name"].replace(" ", "").lower()

    except (ValueError, SyntaxError, TypeError):
        pass

    return ""


movies["director"] = movies["crew"].apply(get_director)


# --------------------------------------------------
# 8. CREATE COMBINED FEATURES
# --------------------------------------------------

movies["tags"] = (
    movies["overview"]
    + " "
    + movies["genres"]
    + " "
    + movies["keywords"]
    + " "
    + movies["cast"]
    + " "
    + movies["director"]
)


# --------------------------------------------------
# 9. CLEAN TEXT
# --------------------------------------------------

movies["tags"] = (
    movies["tags"]
    .str.lower()
    .str.replace(r"[^a-zA-Z\s]", " ", regex=True)
)


print("\n--- Features Prepared ---")
print(movies[["title", "tags"]].head())


# --------------------------------------------------
# 10. TF-IDF
# --------------------------------------------------

vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

vectors = vectorizer.fit_transform(movies["tags"])

print("\n--- TF-IDF Complete ---")
print("Feature matrix:", vectors.shape)


# --------------------------------------------------
# 11. COSINE SIMILARITY
# --------------------------------------------------

similarity = cosine_similarity(vectors)

print("\n--- Similarity Matrix Created ---")
print("Similarity matrix:", similarity.shape)


# --------------------------------------------------
# 12. MOVIE INDEX
# --------------------------------------------------

movie_indices = pd.Series(
    movies.index,
    index=movies["title"].str.lower()
).drop_duplicates()


# --------------------------------------------------
# 13. RECOMMENDATION FUNCTION
# --------------------------------------------------

def recommend(movie_name, number_of_movies=10):

    movie_name = movie_name.lower()

    if movie_name not in movie_indices:
        print(f"\nMovie '{movie_name}' not found.")

        print("\nTry one of these:")
        for title in movies["title"].head(20):
            print("-", title)

        return

    index = movie_indices[movie_name]

    similarity_scores = list(
        enumerate(similarity[index])
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    print("\n" + "=" * 60)
    print(f"Recommendations for: {movies.iloc[index]['title']}")
    print("=" * 60)

    count = 0

    for movie_index, score in similarity_scores[1:]:

        title = movies.iloc[movie_index]["title"]

        print(
            f"{count + 1}. {title} "
            f"(similarity: {score:.3f})"
        )

        count += 1

        if count == number_of_movies:
            break


# --------------------------------------------------
# 14. TEST RECOMMENDATION
# --------------------------------------------------

recommend("The Dark Knight", 10)


# --------------------------------------------------
# 15. INTERACTIVE MODE
# --------------------------------------------------

print("\n" + "=" * 60)
print("INTERACTIVE MOVIE RECOMMENDER")
print("=" * 60)

while True:

    movie_name = input(
        "\nEnter a movie name "
        "(or type 'exit'): "
    )

    if movie_name.lower() == "exit":
        print("\nGoodbye!")
        break

    recommend(movie_name, 10)


print("\n" + "=" * 60)
print("PROJECT COMPLETED")
print("=" * 60)