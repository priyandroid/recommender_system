import pandas as pd
import numpy as np

def calculate_genre_scores(merged_df):
    """
    Calculates the average rating for every unique genre in the dataset.
    This serves as the foundation for the G-Score fallback.
    """
    print("Calculating genre-based quality scores...")
    
    # Explode genres so each rating is associated with a single genre
    relevant_data = merged_df[['movieId', 'rating', 'genres']].copy()
    relevant_data['genres'] = relevant_data['genres'].str.split('|')
    exploded_data = relevant_data.explode('genres')
    
    # Calculate mean rating per genre
    genre_stats = exploded_data.groupby('genres')['rating'].mean().to_dict()
    
    # Provide a safe default for unknown genres
    global_mean = merged_df['rating'].mean()
    
    return genre_stats, global_mean

def get_movie_g_score(genres_str, genre_stats, global_mean):
    """
    Calculates a single 'G-Score' for a movie based on the average of its genres.
    """
    if pd.isna(genres_str) or genres_str == '(no genres listed)':
        return global_mean
    
    genres = genres_str.split('|')
    scores = [genre_stats.get(g, global_mean) for g in genres]
    
    return np.mean(scores)

def hybrid_prediction(user_id, movie_id, svd_model, movies_df, global_mean, min_ratings_threshold=50):
    """
    The core Hybrid Prediction Layer. 
    Decides whether to use SVD or a Fallback based on User/Item warmth.
    
    Returns: (predicted_rating, source_label)
    """
    
    # 1. Check User Status and get User Bias (b_u)
    is_user_cold = False
    user_bias = 0.0
    try:
        # Convert to SVD inner ID
        inner_uid = svd_model.trainset.to_inner_uid(user_id)
        user_bias = svd_model.bu[inner_uid]
    except (ValueError, AttributeError):
        is_user_cold = True
        user_bias = 0.0 # New users start with neutral bias

    # 2. Check Item Status
    # In a production setting, you would pre-calculate 'rating_count' for movies
    # For this implementation, we check if the movie exists in the training set
    is_item_cold = False
    item_bias = 0.0
    try:
        inner_iid = svd_model.trainset.to_inner_iid(movie_id)
        item_bias = svd_model.bi[inner_iid]
    except (ValueError, AttributeError):
        is_item_cold = True

    # 3. Decision Logic (The 4 Scenarios)
    
    # SCENARIO A: WARM USER & WARM ITEM -> Standard SVD Personalization
    if not is_user_cold and not is_item_cold:
        prediction = svd_model.predict(user_id, movie_id).est
        return prediction, "SVD (Personalized)"

    # SCENARIO B: NEW USER & WARM ITEM -> Popularity-Based Baseline (mu + b_i)
    elif is_user_cold and not is_item_cold:
        # We use the item's learned popularity (item_bias) relative to the global mean
        prediction = global_mean + item_bias
        return max(0.5, min(5.0, prediction)), "Popularity Baseline (mu + b_i)"

    # SCENARIO C: WARM USER & NEW ITEM -> Personalized Content Fallback (mu + b_u + G_bias)
    elif not is_user_cold and is_item_cold:
        movie_row = movies_df[movies_df['movieId'] == movie_id]
        if not movie_row.empty:
            g_score = movie_row['G_score'].iloc[0]
            # Use the movie's genre quality as a proxy for item bias
            item_content_bias = g_score - global_mean
            prediction = global_mean + user_bias + item_content_bias
            return max(0.5, min(5.0, prediction)), "Hybrid Content (mu + b_u + G_bias)"
        else:
            return global_mean + user_bias, "User Bias Only (Unknown Movie)"

    # SCENARIO D: NEW USER & NEW ITEM -> Pure Content Fallback (G_score)
    else:
        movie_row = movies_df[movies_df['movieId'] == movie_id]
        if not movie_row.empty:
            return movie_row['G_score'].iloc[0], "Genre-Based Fallback (G_score)"
        return global_mean, "Global Mean (Fallback)"

if __name__ == "__main__":
    print("Hybrid Logic Module loaded. Use hybrid_prediction() in main.py.")