MovieLens 25M: Hybrid Personalized Recommender System

A production-minded recommendation engine using Matrix Factorization (SVD) and a custom Content-Based fallback to solve the Cold Start problem.

🚀 The Story: From Sparsity to Personalization

This project explores the MovieLens 25M dataset to bridge the gap between simple statistical baselines and personalized latent-factor models.

1. The Challenge: Data Sparsity & Cold Start

With 25 million ratings, the dataset exhibits a massive "Long Tail." Most movies have fewer than 100 ratings, making it impossible for standard Collaborative Filtering (CF) models to predict scores for new or unpopular items (The Item Cold Start Problem).

2. The Solution: A Hybrid Approach

I implemented a two-stage prediction pipeline:

Warm Items: Optimized SVD (Matrix Factorization) to capture latent user-item interactions.

Cold Items: A custom G-Score Fallback that utilizes genre-based content averages and user bias to provide reliable predictions where CF fails.

📊 Key Results

I compared three stages of the recommendation pipeline to prove the value of the SVD and Hybrid logic.

### 📈 Final Model Performance Report

| Metric        | Baseline (Bias-Only) | SVD (Optimized) | % Improvement |
|--------------|----------------------|-----------------|---------------|
| RMSE (Error) | 0.8641               | 0.7772          | 10.06%        |
| Precision@10 | 0.3337               | 0.4290          | 28.56%        |
| Recall@10    | 0.3033               | 0.3744          | 23.44%        |


❄️ COLD START PERFORMANCE (Hybrid Logic)

Standard SVD RMSE (Cold Set):  1.2685
Hybrid Fallback RMSE (Cold Set): 0.9532
Hybrid Improvement: 24.85%


📂 Project Structure

├── data/                   # (Git-ignored) MovieLens CSV files
├── src/                    # Source Code
│   ├── preprocessing.py    # Year extraction and cleaning
│   ├── baseline.py         # Global Mean + Biases
│   ├── svd_model.py        # Optimized Matrix Factorization
│   └── hybrid_logic.py     # Cold Start & G-Score Fallback
├── eda.ipynb               # Evaluation plots and metrics
├── main.py                 # Entry point for the hybrid system
├── collaborative_filtering.ipynb
├── gitignore              # Prevents pushing large CSVs
└── requirements.txt        # Reproducibility (numpy==1.26.4, etc.)


🛠️ Technical Implementation Highlights

Problem Solving: Developed a "Hybrid Prediction Layer" that dynamically switches between SVD and Content-based logic based on item/user "warmth."

Optimization: Utilized GridSearchCV for hyperparameter tuning of latent factors ($k$) and regularization ($\lambda$).

Data Engineering: Extracted temporal features (Release Year) and engineered a Genre-Score (G-Score) to quantify movie quality across cold start scenarios.

🔮 Future Roadmap: Deep Learning

The current SVD architecture is time-agnostic (Temporal Drift issue). The next iteration involves a Two-Tower Neural Network (DLRM) using TensorFlow Recommenders to incorporate sequential interaction data and non-linear feature crossings.

Author: Priyanka

Stack: Python, Pandas, Scikit-Surprise, NumPy, Matplotlib/Seaborn
