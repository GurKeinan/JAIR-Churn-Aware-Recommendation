# cluster_analyzer.py
import pandas as pd
import os

def analyze_movie_clusters(movies, ratings, user_item_matrix, movie_labels, n_clusters):
    """Analyze movie clusters for genre distribution and top movies."""
    movie_titles = user_item_matrix.columns
    clusters_analysis = {}

    merged_data = pd.merge(ratings, movies[['MovieID', 'Title', 'Genres']], on='MovieID')

    for i in range(n_clusters):
        cluster_movies = movie_titles[movie_labels == i]
        cluster_info = movies[movies['Title'].isin(cluster_movies)]

        # Genre analysis
        all_genres = []
        for genres in cluster_info['Genres'].str.split('|'):
            if genres is not None:  # Handle potential None values
                all_genres.extend(genres)
        genre_dist = pd.Series(all_genres).value_counts()

        # Rating analysis
        cluster_ratings = merged_data[merged_data['Title'].isin(cluster_movies)].groupby('Title')['Rating'].mean()
        top_movies = cluster_ratings.nlargest(5)

        clusters_analysis[f'Cluster {i}'] = {
            'size': len(cluster_movies),
            'top_genres': genre_dist.head(3).to_dict(),
            'top_movies': top_movies.to_dict()
        }

    return clusters_analysis

def save_cluster_assignments(user_item_matrix, user_labels, movie_labels, output_dir='clusters'):
    """Save cluster assignments to CSV files."""
    # Create user clusters DataFrame
    user_clusters = pd.DataFrame({
        'UserID': range(len(user_labels)),
        'Cluster': user_labels
    })

    # Create movie clusters DataFrame
    movie_clusters = pd.DataFrame({
        'Title': user_item_matrix.columns,
        'Cluster': movie_labels
    })

    # Save to CSV
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    user_clusters.to_csv(os.path.join(output_dir, 'user_clusters.csv'), index=False)
    movie_clusters.to_csv(os.path.join(output_dir, 'movie_clusters.csv'), index=False)

    return movie_clusters, user_clusters