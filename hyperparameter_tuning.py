from sklearn.model_selection import RandomizedSearchCV
import numpy as np

def optimize_hyperparameters(model_name, model, X, y, cv=5):
    """Optymalizuje hiperparametry dla wybranego modelu"""
    
    if model_name == 'Random Forest':
        # Przestrzeń przeszukiwania dla Random Forest
        param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    # Dodaj ten fragment wewnątrz funkcji optimize_hyperparameters, po warunku dla 'KNN':
    elif model_name == 'Naive Bayes':
    # Przestrzeń przeszukiwania dla Naive Bayes (GaussianNB)
        param_grid = {
            'var_smoothing': np.logspace(0, -9, num=10)  # Jedyny hiperparametr GaussianNB
        }
    elif model_name == 'Gradient Boosting':
        # Przestrzeń przeszukiwania dla Gradient Boosting
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
    
    elif model_name == 'Neural Network (MLP)' or model_name == 'Deep Neural Network':
        # Przestrzeń przeszukiwania dla sieci neuronowej
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive']
        }
    
    elif model_name == 'Logistic Regression':
        # Przestrzeń przeszukiwania dla regresji logistycznej
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'solver': ['liblinear', 'lbfgs', 'newton-cg']
        }
    
    elif model_name == 'KNN':
        # Przestrzeń przeszukiwania dla KNN
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]  # Manhattan lub Euclidean
        }
    
    else:
        print(f"Brak zdefiniowanej przestrzeni przeszukiwania dla modelu {model_name}")
        return model, {}, 0.0
    
    # Użyj RandomizedSearchCV, bo jest szybszy niż GridSearchCV
    search = RandomizedSearchCV(
        model, param_grid, n_iter=10, cv=cv, verbose=1, 
        n_jobs=-1, random_state=42
    )
    
    # Trenuj model
    print(f"Optymalizacja hiperparametrów dla modelu {model_name}...")
    search.fit(X, y)
    
    print(f"Najlepsze parametry dla {model_name}:")
    print(search.best_params_)
    print(f"Najlepsza dokładność CV: {search.best_score_:.4f}")
    
    return search.best_estimator_, search.best_params_, search.best_score_