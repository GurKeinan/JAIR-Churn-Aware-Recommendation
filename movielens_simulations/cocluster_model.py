# cocluster_model.py
import numpy as np
from sklearn.cluster import SpectralCoclustering

def normalize_ratings(user_item_matrix):
    """Normalize the ratings matrix."""
    normalized_ratings = user_item_matrix.values
    rating_means = normalized_ratings.mean(1)
    return normalized_ratings - rating_means.reshape(-1, 1)

def perform_coclustering(normalized_ratings, n_clusters=5):
    """Perform coclustering on normalized ratings."""
    model = SpectralCoclustering(n_clusters=n_clusters, random_state=0)
    model.fit(normalized_ratings)
    return model.row_labels_, model.column_labels_

def sort_clusters(normalized_ratings, user_labels, movie_labels):
    """Sort matrices according to cluster labels."""
    user_labels_sorted = np.argsort(user_labels)
    movie_labels_sorted = np.argsort(movie_labels)

    reordered_ratings = normalized_ratings[user_labels_sorted]
    reordered_ratings = reordered_ratings[:, movie_labels_sorted]

    return reordered_ratings, user_labels_sorted, movie_labels_sorted