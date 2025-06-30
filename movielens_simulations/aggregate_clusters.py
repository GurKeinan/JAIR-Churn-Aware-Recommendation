import pandas as pd
import numpy as np
import os
import time

def load_data():
    # Read ratings file
    ratings_path = os.path.join(os.path.dirname(__file__), 'ml-1m', 'ratings.dat')
    ratings = pd.read_csv(ratings_path, sep='::',
                         names=['UserID', 'MovieID', 'Rating', 'Timestamp'],
                         engine='python')

    # Read movies file
    movies_path = os.path.join(os.path.dirname(__file__), 'ml-1m', 'movies.dat')
    movies = pd.read_csv(movies_path, sep='::',
                        names=['MovieID', 'Title', 'Genres'],
                        encoding='latin-1',
                        engine='python')

    return ratings, movies

def aggregate_clusters(cluster_size=None):
    # Determine which cluster directory to use
    base_clusters_dir = os.path.join(os.path.dirname(__file__), 'clusters')
    if cluster_size is not None:
        clusters_dir = os.path.join(base_clusters_dir, f'size_{cluster_size}')
    else:
        # Use the first size directory found if none specified
        size_dirs = [d for d in os.listdir(base_clusters_dir) if d.startswith('size_')]
        if not size_dirs:
            raise ValueError("No cluster size directories found")
        clusters_dir = os.path.join(base_clusters_dir, size_dirs[0])

    # Wait for cluster files to be created (max 30 seconds)
    movie_clusters_path = os.path.join(clusters_dir, 'movie_clusters.csv')
    user_clusters_path = os.path.join(clusters_dir, 'user_clusters.csv')

    timeout = 30
    start_time = time.time()
    while not (os.path.exists(movie_clusters_path) and os.path.exists(user_clusters_path)):
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Timeout waiting for cluster files in {clusters_dir}")
        time.sleep(1)

    # Read the cluster files
    movie_clusters = pd.read_csv(movie_clusters_path)
    user_clusters = pd.read_csv(user_clusters_path)

    # Load ratings and movies data
    ratings, movies = load_data()

    # First merge movies with their clusters
    movies_with_clusters = movies.merge(movie_clusters, on='Title', how='inner')

    # Then merge with ratings
    ratings_with_clusters = ratings.merge(movies_with_clusters[['MovieID', 'Cluster']],
                                        on='MovieID',
                                        how='inner')

    # Finally merge with user clusters
    ratings_with_clusters = ratings_with_clusters.merge(user_clusters,
                                                      on='UserID',
                                                      how='inner',
                                                      suffixes=('_movie', '_user'))

    # Get unique clusters and create index mappings
    unique_movie_clusters = movie_clusters['Cluster'].unique()
    unique_user_clusters = user_clusters['Cluster'].unique()

    movie_cluster_map = {old: new for new, old in enumerate(unique_movie_clusters)}
    user_cluster_map = {old: new for new, old in enumerate(unique_user_clusters)}

    # Create the aggregation matrix with proper dimensions
    n_movie_clusters = len(unique_movie_clusters)
    n_user_clusters = len(unique_user_clusters)

    aggregation_matrix = np.zeros((n_movie_clusters, n_user_clusters))
    count_matrix = np.zeros((n_movie_clusters, n_user_clusters))

    # Group by movie and user clusters and calculate mean rating
    grouped = ratings_with_clusters.groupby(['Cluster_movie', 'Cluster_user'])

    for (movie_cluster, user_cluster), group in grouped:
        mapped_movie_cluster = movie_cluster_map[movie_cluster]
        mapped_user_cluster = user_cluster_map[user_cluster]
        aggregation_matrix[mapped_movie_cluster, mapped_user_cluster] = group['Rating'].mean()
        count_matrix[mapped_movie_cluster, mapped_user_cluster] = len(group)

    # Normalize ratings by dividing by 5
    aggregation_matrix = aggregation_matrix / 5.0

    # Replace any NaN values with 0
    aggregation_matrix = np.nan_to_num(aggregation_matrix)

    return aggregation_matrix, count_matrix    # Determine which cluster directory to use
    base_clusters_dir = os.path.join(os.path.dirname(__file__), 'clusters')
    if cluster_size is not None:
        clusters_dir = os.path.join(base_clusters_dir, f'size_{cluster_size}')
    else:
        # Use the first size directory found if none specified
        size_dirs = [d for d in os.listdir(base_clusters_dir) if d.startswith('size_')]
        if not size_dirs:
            raise ValueError("No cluster size directories found")
        clusters_dir = os.path.join(base_clusters_dir, size_dirs[0])

    # Wait for cluster files to be created (max 30 seconds)
    movie_clusters_path = os.path.join(clusters_dir, 'movie_clusters.csv')
    user_clusters_path = os.path.join(clusters_dir, 'user_clusters.csv')

    timeout = 30
    start_time = time.time()
    while not (os.path.exists(movie_clusters_path) and os.path.exists(user_clusters_path)):
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Timeout waiting for cluster files in {clusters_dir}")
        time.sleep(1)

    # Read the cluster files
    movie_clusters = pd.read_csv(movie_clusters_path)
    user_clusters = pd.read_csv(user_clusters_path)

    # Load ratings and movies data
    ratings, movies = load_data()

    # First merge movies with their clusters
    movies_with_clusters = movies.merge(movie_clusters, on='Title', how='inner')

    # Then merge with ratings
    ratings_with_clusters = ratings.merge(movies_with_clusters[['MovieID', 'Cluster']],
                                        on='MovieID',
                                        how='inner')

    # Finally merge with user clusters
    ratings_with_clusters = ratings_with_clusters.merge(user_clusters,
                                                      on='UserID',
                                                      how='inner',
                                                      suffixes=('_movie', '_user'))

    # Get the number of unique clusters
    n_movie_clusters = movie_clusters['Cluster'].nunique()
    n_user_clusters = user_clusters['Cluster'].nunique()

    # Create the aggregation matrix
    aggregation_matrix = np.zeros((n_movie_clusters, n_user_clusters))
    count_matrix = np.zeros((n_movie_clusters, n_user_clusters))

    # Group by movie and user clusters and calculate mean rating
    grouped = ratings_with_clusters.groupby(['Cluster_movie', 'Cluster_user'])

    for (movie_cluster, user_cluster), group in grouped:
        aggregation_matrix[movie_cluster, user_cluster] = group['Rating'].mean()
        count_matrix[movie_cluster, user_cluster] = len(group)

    # Normalize by dividing by 5
    aggregation_matrix = aggregation_matrix / 5.0

    # Replace any NaN values with 0
    aggregation_matrix = np.nan_to_num(aggregation_matrix)

    return aggregation_matrix, count_matrix

if __name__ == "__main__":
    matrix, counts = aggregate_clusters()
    print("Aggregation Matrix Shape:", matrix.shape)
    print("\nAggregation Matrix (normalized ratings):")
    print(matrix)
    print("\nCount Matrix (number of ratings per cluster pair):")
    print(counts)