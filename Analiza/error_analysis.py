import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_errors(X_test, y_test, y_pred, feature_names):
    """Analizuje błędy modelu"""
    # Utwórz DataFrame z danymi testowymi
    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df['actual'] = y_test
    test_df['predicted'] = y_pred
    test_df['correct'] = test_df['actual'] == test_df['predicted']
    
    # Znajdź przypadki, w których model popełnił błędy
    errors = test_df[~test_df['correct']]
    
    # Podstawowe statystyki dla błędów
    print(f"Liczba błędów: {len(errors)} z {len(test_df)} ({len(errors)/len(test_df):.2%})")
    
    # Analiza rozkładu cech dla błędnych predykcji
    fig, axes = plt.subplots(len(feature_names), 2, figsize=(15, 4*len(feature_names)))
    
    for i, feature in enumerate(feature_names):
        # Histogram dla poprawnie sklasyfikowanych przypadków
        sns.histplot(
            test_df[test_df['correct']][feature], 
            ax=axes[i, 0], 
            color='green',
            alpha=0.6
        )
        axes[i, 0].set_title(f"{feature} - Poprawne klasyfikacje")
        
        # Histogram dla błędnie sklasyfikowanych przypadków
        sns.histplot(
            errors[feature], 
            ax=axes[i, 1], 
            color='red',
            alpha=0.6
        )
        axes[i, 1].set_title(f"{feature} - Błędne klasyfikacje")
    
    plt.tight_layout()
    
    return fig, errors