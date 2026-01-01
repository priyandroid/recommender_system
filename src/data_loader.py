import pandas as pd
import numpy as np
import os
import re
from surprise import Reader, Dataset

def extract_release_year(title):
    """
    Extracts the release year from the movie title string using regex.
    Example: 'Toy Story (1995)' -> 1995
    """
    match = re.search(r'\((\d{4})\)', title)
    if match:
        return int(match.group(1))
    return np.nan




# Path to extracted MovieLens dataset
DATA_DIR = './data/ml-25m'

# Optimized dtypes
rating_dtypes = {
    'userId': 'int32',
    'movieId': 'int32',
    'rating': 'float32',
    'timestamp': 'int32'
}
movie_dtypes = {
    'movieId': 'int32',
    'title': 'string',
    'genres': 'string'
}
tag_dtypes = {
    'userId': 'int32',
    'movieId': 'int32',
    'tag': 'string',
    'timestamp': 'int32'
}
link_dtypes = {
    'movieId': 'int32',
    'imdbId': 'float32',
    'tmdbId': 'float32'
}


def load_csv(filename, dtypes):
    path = os.path.join(DATA_DIR, filename)
    print(f"Loading {filename} ...")
    df = pd.read_csv(path, dtype=dtypes)
    print(f"Loaded {filename}: shape={df.shape}")
    return df


def load_and_clean_data(ratings_path='ratings.csv', movies_path='movies.csv', sample_size=1000000):
    """
    Loads MovieLens datasets, performs initial cleaning, and feature engineering.
    
    Args:
        ratings_path (str): Path to the ratings CSV.
        movies_path (str): Path to the movies CSV.
        sample_size (int): Number of ratings to sample for performance. Set to None for full data.
        
    Returns:
        tuple: (cleaned_ratings_df, movies_df, merged_df)
    """
    print(f"--- Loading data from {ratings_path} and {movies_path} ---")
    
    try:
        movies_df = load_csv('movies.csv', movie_dtypes)
        # Use cols to save memory on large 25M dataset
        ratings_df = load_csv('ratings.csv', rating_dtypes)
    except FileNotFoundError as e:
        print(f"Error: Could not find data files. {e}")
        return None, None, None

    # 1. Sampling for development speed
    if sample_size and len(ratings_df) > sample_size:
        print(f"Sampling {sample_size:,} rows for faster processing...")
        ratings_df = ratings_df.sample(n=sample_size, random_state=42)

    # 2. Basic Cleaning
    ratings_df.drop_duplicates(inplace=True)
    
    # 3. Feature Engineering: Timestamps
    ratings_df['rating_year'] = pd.to_datetime(ratings_df['timestamp'], unit='s').dt.year
    
    # 4. Feature Engineering: Movie Release Year
    movies_df['release_year'] = movies_df['title'].apply(extract_release_year)
    
    # 5. Create Clean Title (removing the year from the string)
    movies_df['clean_title'] = movies_df['title'].str.replace(r'\s*\(\d{4}\)', '', regex=True).str.strip()

    # 6. Merging for a comprehensive view (useful for EDA and G-Score calculation)
    merged_df = pd.merge(ratings_df, movies_df, on='movieId', how='left')
    
    print(f"Successfully loaded {len(ratings_df):,} ratings and {len(movies_df):,} movies.")
    return ratings_df, movies_df, merged_df

def get_surprise_dataset(ratings_df):
    """
    Converts a standard Pandas DataFrame into a Scikit-Surprise Dataset object.
    """
    reader = Reader(rating_scale=(0.5, 5.0))
    # Surprise expects columns in this specific order: user, item, rating
    data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
    return data

if __name__ == "__main__":
    # Test the loader
    r_df, m_df, full_df = load_and_clean_data()
    if r_df is not None:
        print("\nPreview of Cleaned Ratings:")
        print(r_df.head())
        print("\nPreview of Movies with Release Years:")
        print(m_df[['title', 'release_year']].head())