# main.py
import os
from data_loader import load_movielens_data, create_user_item_matrix
from cocluster_model import normalize_ratings, perform_coclustering, sort_clusters
from cluster_analyzer import analyze_movie_clusters, save_cluster_assignments
from visualizer import plot_coclustered_matrix

VERBOSE = False

def main(n_clusters=10):
    # Load data
    users, movies, ratings = load_movielens_data()
    user_item_matrix = create_user_item_matrix(ratings, users, movies)

    # Normalize and cluster
    normalized_ratings = normalize_ratings(user_item_matrix)
    user_labels, movie_labels = perform_coclustering(normalized_ratings, n_clusters=n_clusters)

    # Create base clusters directory if it doesn't exist
    base_clusters_dir = os.path.join(os.path.dirname(__file__), 'clusters')
    if not os.path.exists(base_clusters_dir):
        os.makedirs(base_clusters_dir)

    # Create size-specific subdirectory
    clusters_dir = os.path.join(base_clusters_dir, f'size_{n_clusters}')
    if not os.path.exists(clusters_dir):
        os.makedirs(clusters_dir)

    # Sort and visualize
    reordered_ratings, user_sorted, movie_sorted = sort_clusters(
        normalized_ratings, user_labels, movie_labels
    )
    plot_coclustered_matrix(reordered_ratings, save_path=os.path.join(clusters_dir, 'coclustered_matrix.png'))

    # Analyze results
    cluster_analysis = analyze_movie_clusters(
        movies, ratings, user_item_matrix, movie_labels, n_clusters=n_clusters
    )

    # Save results
    movie_clusters, user_clusters = save_cluster_assignments(
        user_item_matrix, user_labels, movie_labels, output_dir=clusters_dir
    )

    # Print analysis
    if VERBOSE:
        for cluster, info in cluster_analysis.items():
            print(f"\n{cluster}:")
            print(f"Number of movies: {info['size']}")
            print("Top genres:", info['top_genres'])
            print("Top movies:", info['top_movies'])

    return clusters_dir

if __name__ == "__main__":
    main()