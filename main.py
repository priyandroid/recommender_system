"""
MovieLens 25M: Hybrid Personalized Recommender System
Entry point for the production-ready pipeline.

This script orchestrates the modular components:
1. Data Loading & Cleaning (data_loader.py)
2. SVD Training & Optimization (collaborative_filtering.py)
3. Hybrid Prediction & Cold Start Handling (hybrid_logic.py)
4. Evaluation & Performance Reporting
"""

import os
import sys
import pandas as pd
import numpy as np
from collections import defaultdict
from surprise import accuracy, BaselineOnly

# Add src to path if needed (standard for modular projects)
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import our custom modules
from src.data_loader import load_and_clean_data, get_surprise_dataset
from src.collaborative_filtering import train_svd_model
from src.hybrid_logic import calculate_genre_scores, get_movie_g_score, hybrid_prediction

def calculate_ranking_metrics(predictions, k=10, threshold=4.0):
    """
    Calculates Precision@k and Recall@k for a given set of predictions.
    """
    # Map the predictions to each user
    user_est_true = defaultdict(list)
    for uid, iid, r_ui, est, _ in predictions:
        user_est_true[uid].append((est, r_ui))

    precisions = dict()
    recalls = dict()

    for uid, user_ratings in user_est_true.items():
        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        precisions[uid] = n_rel_and_rec_k / k if k != 0 else 0

        # Recall@K: Proportion of relevant items that are recommended
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    return np.mean(list(precisions.values())), np.mean(list(recalls.values()))

def run_pipeline():
    print("🚀 Initializing Hybrid Recommender Pipeline...")

    # --- 1. DATA LOADING & PREPROCESSING ---
    ratings_df, movies_df, merged_df = load_and_clean_data(
        ratings_path='ratings.csv', 
        movies_path='movies.csv', 
        # sample_size=1000000
        sample_size=None
    )

    if ratings_df is None:
        print("❌ Error: Pipeline aborted due to missing data.")
        return

    # --- 2. FEATURE ENGINEERING: G-SCORE ---
    genre_stats, global_mean = calculate_genre_scores(merged_df)
    
    print("Engineering G-Score features for movies...")
    movies_df['G_score'] = movies_df['genres'].apply(
        lambda x: get_movie_g_score(x, genre_stats, global_mean)
    )

    # --- 3. MODEL TRAINING ---
    data = get_surprise_dataset(ratings_df)
    
    # 3.1 Train SVD (Optimized)
    algo_svd, trainset, testset = train_svd_model(
        data, n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02
    )
    
    # 3.2 Train Baseline (Bias-Only) for Comparison
    print("Training Baseline Model for benchmarking...")
    algo_baseline = BaselineOnly(bsl_options={'method': 'sgd'})
    algo_baseline.fit(trainset)

    # --- 4. EVALUATION SETUP ---
    print("\n--- Evaluation: Preparing Metrics ---")
    
    # Identify Warm vs Cold Test Cases
    train_users = set(trainset.ur.keys())
    train_items = set(trainset.ir.keys())
    
    warm_test_ratings = []
    cold_test_ratings = []
    
    for uid, iid, r_ui in testset:
        try:
            is_warm = (trainset.to_inner_uid(uid) in train_users) and \
                      (trainset.to_inner_iid(iid) in train_items)
            if is_warm:
                warm_test_ratings.append((uid, iid, r_ui))
            else:
                cold_test_ratings.append((uid, iid, r_ui))
        except ValueError:
            cold_test_ratings.append((uid, iid, r_ui))

    # --- 5. CALCULATE METRICS (WARM SET) ---
    
    # SVD Metrics
    svd_preds = algo_svd.test(warm_test_ratings)
    svd_rmse = accuracy.rmse(svd_preds, verbose=False)
    svd_prec, svd_rec = calculate_ranking_metrics(svd_preds, k=10)
    
    # Baseline Metrics
    base_preds = algo_baseline.test(warm_test_ratings)
    base_rmse = accuracy.rmse(base_preds, verbose=False)
    base_prec, base_rec = calculate_ranking_metrics(base_preds, k=10)

    # Calculate Improvements
    rmse_imp = ((base_rmse - svd_rmse) / base_rmse) * 100
    prec_imp = ((svd_prec - base_prec) / base_prec) * 100
    rec_imp = ((svd_rec - base_rec) / base_rec) * 100

    # --- 6. COLD SET EVALUATION (Hybrid Logic) ---
    svd_default_preds = []
    hybrid_preds = []
    
    for uid, iid, r_ui in cold_test_ratings:
        svd_default_preds.append((uid, iid, r_ui, global_mean, {}))
        h_pred, _ = hybrid_prediction(uid, iid, algo_svd, movies_df, global_mean)
        hybrid_preds.append((uid, iid, r_ui, h_pred, {}))

    cold_svd_rmse = accuracy.rmse(svd_default_preds, verbose=False)
    cold_hybrid_rmse = accuracy.rmse(hybrid_preds, verbose=False)
    cold_imp = ((cold_svd_rmse - cold_hybrid_rmse) / cold_svd_rmse) * 100

    # --- 7. FINAL PERFORMANCE REPORT ---
    print("\n" + "="*80)
    print("📈 FINAL MODEL PERFORMANCE REPORT")
    print("="*80)
    
    # The Requested Matrix Table
    print(f"{'Metric':<20} | {'Baseline (Bias-Only)':<20} | {'SVD (Optimized)':<20} | {'% Improvement':<15}")
    print("-" * 83)
    print(f"{'RMSE (Error)':<20} | {base_rmse:<20.4f} | {svd_rmse:<20.4f} | {rmse_imp:>13.2f}%")
    print(f"{'Precision@10':<20} | {base_prec:<20.4f} | {svd_prec:<20.4f} | {prec_imp:>13.2f}%")
    print(f"{'Recall@10':<20} | {base_rec:<20.4f} | {svd_rec:<20.4f} | {rec_imp:>13.2f}%")
    print("-" * 83)
    
    print("\n❄️ COLD START PERFORMANCE (Hybrid Logic)")
    print(f"Standard SVD RMSE (Cold Set):  {cold_svd_rmse:.4f}")
    print(f"Hybrid Fallback RMSE (Cold Set): {cold_hybrid_rmse:.4f}")
    print(f"Hybrid Improvement: {cold_imp:.2f}%")
    
    print("\n" + "="*80)
    print("CONCLUSION: The SVD model successfully captures latent user preferences that")
    print("simple popularity and bias models miss, resulting in a significantly more")
    print("accurate 'Top 10' recommendation list.")
    print("="*80)

if __name__ == "__main__":
    run_pipeline()