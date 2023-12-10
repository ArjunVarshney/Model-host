import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def predict(movie):
    df = pd.read_csv("./models/movie_recommendation/movies.csv")
    selected = ["genres", "keywords", "tagline", "cast", "director"]
    for feature in selected:
        df[feature] = df[feature].fillna("")
    combined_df = (
        df["genres"]
        + " "
        + df["keywords"]
        + " "
        + df["tagline"]
        + " "
        + df["cast"]
        + " "
        + df["director"]
    )
    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(combined_df)
    similarity = cosine_similarity(feature_vectors)

    movie_name = movie["movie_name"][0]
    list_of_all_titles = df["title"].tolist()
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
    close_match = find_close_match[0]
    index_of_the_movie = df[df.title == close_match]["index"].values[0]
    similarity_score = list(enumerate(similarity[index_of_the_movie]))
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    result = []

    for movie in sorted_similar_movies:
        index = movie[0]
        title_from_index = df[df.index == index]["title"].values[0]
        if len(result) <= 20:
            result.append(title_from_index)
        else:
            break

    return result
