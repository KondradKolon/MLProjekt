import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc

def plot_model_comparison(model_results, output_path=None):

    model_names = list(model_results.keys())
    accuracies = [model_results[name]['accuracy'] for name in model_names]
    
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, accuracies, color='skyblue')
    
    
    for bar, acc in zip(bars, accuracies):
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.005,
            f'{acc:.4f}',
            ha='center',
            fontsize=10
        )
    
    
    plt.title('Porównanie dokładności modeli', fontsize=14)
    plt.ylabel('Dokładność (accuracy)', fontsize=12)
    plt.ylim(0.5, 1.0)  
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Wykres zapisano jako {output_path}")
    else:
        plt.show()
    
    plt.close()

def plot_confusion_matrices(model_results, y_test, predictions, output_path=None):

   
    n_models = len(model_results)
    
   
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    if n_models == 1:
        axes = [axes]  
    
 
    for i, (name, y_pred) in enumerate(predictions.items()):
        cm = confusion_matrix(y_test, y_pred)
        
       
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues', 
            ax=axes[i],
            cbar=False,
            annot_kws={"size": 16}
        )
        

        axes[i].set_title(f'{name}\nDokładność: {model_results[name]["accuracy"]:.4f}')
        axes[i].set_ylabel('Wartość rzeczywista')
        axes[i].set_xlabel('Wartość przewidywana')
        axes[i].set_xticklabels(['Porażka', 'Wygrana'])
        axes[i].set_yticklabels(['Porażka', 'Wygrana'])
    
    plt.tight_layout()
    
  
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Wykres macierzy pomyłek zapisano jako {output_path}")
    else:
        plt.show()
    
    plt.close()

def plot_feature_importance(feature_names, importances, model_name, output_path=None):
   
    # Przygotowanie danych do wykresu
    features_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    # Utworzenie wykresu
    plt.figure(figsize=(10, 6))
    plt.barh(features_df['Feature'], features_df['Importance'], color='forestgreen')
    
    # Dodanie tytułu i etykiet
    plt.title(f'Ważność cech - {model_name}', fontsize=14)
    plt.xlabel('Ważność')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Zapisz wykres lub wyświetl
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Wykres ważności cech zapisano jako {output_path}")
    else:
        plt.show()
    
    plt.close()

def plot_roc_curves(model_results, y_test, predictions_proba, output_path=None):
    """
    Tworzy wykres krzywych ROC dla wszystkich modeli.
    
    Args:
        model_results: Słownik z wynikami modeli
        y_test: Rzeczywiste wartości dla zbioru testowego
        predictions_proba: Słownik z prawdopodobieństwami przynależności do klasy 1
        output_path: Ścieżka do zapisania wykresu (opcjonalnie)
    """
    plt.figure(figsize=(10, 8))
    
    # Dodaj krzywą ROC dla każdego modelu
    for name, y_prob in predictions_proba.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.4f})')
    
    # Dodaj linię odniesienia (losowa klasyfikacja)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    # Dodaj tytuł i etykiety
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Odsetek fałszywie pozytywnych')
    plt.ylabel('Odsetek prawdziwie pozytywnych')
    plt.title('Krzywe ROC dla różnych modeli')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    # Zapisz wykres lub wyświetl
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Wykres krzywych ROC zapisano jako {output_path}")
    else:
        plt.show()
    
    plt.close()

def plot_learning_curves(train_sizes, train_scores, test_scores, model_name, output_path=None):
    """
    Tworzy wykres krzywych uczenia dla danego modelu.
    
    Args:
        train_sizes: Rozmiary zbiorów treningowych
        train_scores: Wyniki na zbiorze treningowym
        test_scores: Wyniki na zbiorze testowym
        model_name: Nazwa modelu (do tytułu wykresu)
        output_path: Ścieżka do zapisania wykresu (opcjonalnie)
    """
    plt.figure(figsize=(10, 6))
    
    # Oblicz średnie i odchylenia standardowe
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Narysuj krzywe uczenia
    plt.plot(train_sizes, train_mean, 'o-', color='r', label='Zbiór treningowy')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
    
    plt.plot(train_sizes, test_mean, 'o-', color='g', label='Zbiór testowy')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')
    
    # Dodaj tytuł i etykiety
    plt.title(f'Krzywe uczenia - {model_name}')
    plt.xlabel('Liczba przykładów treningowych')
    plt.ylabel('Dokładność')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    
    # Zapisz wykres lub wyświetl
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Wykres krzywych uczenia zapisano jako {output_path}")
    else:
        plt.show()
    
    plt.close()

def plot_seasonal_accuracy(results_by_season, output_path=None):
    """
    Tworzy wykres dokładności modelu w zależności od sezonu.
    
    Args:
        results_by_season: Słownik {sezon: dokładność}
        output_path: Ścieżka do zapisania wykresu (opcjonalnie)
    """
    seasons = list(results_by_season.keys())
    accuracies = list(results_by_season.values())
    
    plt.figure(figsize=(12, 6))
    plt.plot(seasons, accuracies, 'o-', linewidth=2, markersize=8)
    
    # Dodaj tytuł i etykiety
    plt.title('Dokładność predykcji w poszczególnych sezonach')
    plt.xlabel('Sezon')
    plt.ylabel('Dokładność')
    plt.grid(True, alpha=0.3)
    
    # Dodaj linię trendu (średnia ruchoma)
    if len(accuracies) > 3:
        window = min(5, len(accuracies) // 2)
        trend = pd.Series(accuracies).rolling(window=window, center=True).mean()
        plt.plot(seasons, trend, 'r--', linewidth=2, label=f'Trend (średnia {window}-sezonowa)')
        plt.legend()
    
    plt.tight_layout()
    
    # Zapisz wykres lub wyświetl
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Wykres sezonowy zapisano jako {output_path}")
    else:
        plt.show()
    
    plt.close()
# Dodaj tę funkcję na samym końcu pliku:

def plot_learning_curves(model, X, y, cv, train_sizes=np.linspace(0.1, 1.0, 10), output_path=None):
    """Tworzy wykres krzywych uczenia dla modelu"""
    from sklearn.model_selection import learning_curve
    
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, train_sizes=train_sizes, scoring='accuracy', n_jobs=-1
    )
    
    # Oblicz średnią i odchylenie standardowe
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Rysuj wykres
    plt.figure(figsize=(10, 6))
    plt.grid(True)
    
    plt.fill_between(
        train_sizes, train_mean - train_std, train_mean + train_std, 
        alpha=0.1, color="r"
    )
    plt.fill_between(
        train_sizes, test_mean - test_std, test_mean + test_std, 
        alpha=0.1, color="g"
    )
    plt.plot(train_sizes, train_mean, 'o-', color="r", label="Trening")
    plt.plot(train_sizes, test_mean, 'o-', color="g", label="Walidacja")
    
    plt.title(f"Krzywa uczenia dla modelu {type(model).__name__}")
    plt.xlabel("Liczba próbek treningowych")
    plt.ylabel("Dokładność")
    plt.legend(loc="best")
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return plt