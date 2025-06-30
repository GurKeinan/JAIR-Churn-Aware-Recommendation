# data_loader.py
import pandas as pd
import os

def load_movielens_data():
    """Load MovieLens datasets."""
    base_path = os.path.join(os.path.dirname(__file__), 'ml-1m/')
    users = pd.read_csv(f'{base_path}users.dat',
                        sep='::',
                        header=None,
                        names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'],
                        engine='python',
                        encoding='ISO-8859-1')

    movies = pd.read_csv(f'{base_path}movies.dat',
                        sep='::',
                        header=None,
                        names=['MovieID', 'Title', 'Genres'],
                        engine='python',
                        encoding='ISO-8859-1')

    ratings = pd.read_csv(f'{base_path}ratings.dat',
                        sep='::',
                        header=None,
                        names=['UserID', 'MovieID', 'Rating', 'Timestamp'],
                        engine='python',
                        encoding='ISO-8859-1')

    return users, movies, ratings

def create_user_item_matrix(ratings, users, movies):
    """Create and normalize user-item matrix."""
    data = pd.merge(pd.merge(ratings, users), movies)
    user_item_matrix = data.pivot_table(index='UserID',
                                    columns='Title',
                                    values='Rating')
    return user_item_matrix.fillna(0)