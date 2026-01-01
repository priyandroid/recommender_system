import pandas as pd
from surprise import SVD, accuracy
from surprise.model_selection import train_test_split, GridSearchCV

def train_svd_model(data, n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02):
    """
    Trains an SVD model using the scikit-surprise library.
    
    Args:
        data (surprise.Dataset): The surprise dataset object.
        n_factors (int): Number of latent factors (k).
        n_epochs (int): Number of SGD iterations.
        lr_all (float): Learning rate for all parameters.
        reg_all (float): Regularization term for all parameters.
        
    Returns:
        tuple: (trained_algo, trainset, testset)
    """
    print(f"--- Training Matrix Factorization (SVD) ---")
    print(f"Parameters: k={n_factors}, epochs={n_epochs}, lr={lr_all}, reg={reg_all}")
    
    # Split the data into train and test sets (80/20)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    
    # Initialize the SVD algorithm
    algo = SVD(
        n_factors=n_factors, 
        n_epochs=n_epochs, 
        lr_all=lr_all, 
        reg_all=reg_all, 
        random_state=42
    )
    
    # Fit the model to the training set
    algo.fit(trainset)
    
    return algo, trainset, testset

def evaluate_svd(algo, testset):
    """
    Evaluates the SVD model on a given testset and returns core metrics.
    """
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)
    
    return rmse, mae, predictions

def optimize_svd(data):
    """
    Performs a Grid Search to find the optimal hyperparameters for the SVD model.
    This demonstrates advanced model tuning capabilities.
    """
    print("Starting Hyperparameter Optimization (Grid Search)...")
    
    param_grid = {
        'n_factors': [50, 100, 150],
        'n_epochs': [20, 30],
        'lr_all': [0.002, 0.005],
        'reg_all': [0.02, 0.05]
    }
    
    # Cross-validation with 3 folds
    gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
    gs.fit(data)
    
    print(f"Optimization Complete.")
    print(f"Best RMSE: {gs.best_score['rmse']:.4f}")
    print(f"Best Parameters: {gs.best_params['rmse']}")
    
    return gs.best_params['rmse']

if __name__ == "__main__":
    print("Collaborative Filtering Module loaded successfully.")
    print("Use train_svd_model() to initialize your recommendation engine.")