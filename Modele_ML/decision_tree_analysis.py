import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, classification_report
import os

def train_decision_tree(X_train, y_train, X_test, y_test, feature_names, max_depth=4):
    """
    Trenuje drzewo decyzyjne i zwraca wyniki.
    
    Args:
        X_train, y_train: Dane treningowe
        X_test, y_test: Dane testowe
        feature_names: Nazwy cech
        max_depth: Maksymalna głębokość drzewa
        
    Returns:
        Wyniki modelu i wytrenowane drzewo
    """
    # Trenuj drzewo decyzyjne
    dt_model = DecisionTreeClassifier(
        max_depth=max_depth, 
        random_state=42,
        class_weight='balanced'
    )
    
    dt_model.fit(X_train, y_train)
    
    # Predykcje
    y_pred = dt_model.predict(X_test)
    
    # Ocena modelu
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    print(f"Dokładność drzewa decyzyjnego (głębokość={max_depth}): {accuracy:.4f}")
    
    results = {
        'model': dt_model,
        'accuracy': accuracy,
        'classification_report': report,
        'feature_names': feature_names,
        'feature_importance': dt_model.feature_importances_
    }
    
    return results

def plot_decision_tree_visualization(tree_results, output_path=None):
    """
    Wizualizuje drzewo decyzyjne.
    
    Args:
        tree_results: Wyniki z funkcji train_decision_tree
        output_path: Ścieżka do zapisania wykresu
    """
    model = tree_results['model']
    feature_names = tree_results['feature_names']
    
    # Utwórz wizualizację drzewa
    plt.figure(figsize=(20, 10))
    plot_tree(
        model, 
        feature_names=feature_names, 
        filled=True, 
        rounded=True, 
        class_names=['Przegrana', 'Wygrana'],
        fontsize=10
    )
    plt.title(f"Drzewo Decyzyjne (głębokość={model.get_depth()})", fontsize=16)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Zapisano wizualizację drzewa do {output_path}")
    else:
        plt.show()

def plot_feature_importance_dt(tree_results, output_path=None):
    """
    Wizualizuje ważność cech w drzewie decyzyjnym.
    
    Args:
        tree_results: Wyniki z funkcji train_decision_tree
        output_path: Ścieżka do zapisania wykresu
    """
    feature_names = tree_results['feature_names']
    feature_importance = tree_results['feature_importance']
    
    # Sortuj cechy według ważności
    indices = np.argsort(feature_importance)
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), feature_importance[indices], color='skyblue')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Ważność Cechy')
    plt.title('Ważność Cech w Drzewie Decyzyjnym', fontsize=14)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Zapisano wykres ważności cech do {output_path}")
    else:
        plt.show()

def get_decision_rules(tree_results):
    """
    Wyświetla reguły decyzyjne w formie tekstowej.
    
    Args:
        tree_results: Wyniki z funkcji train_decision_tree
    
    Returns:
        Tekst z regułami decyzyjnymi
    """
    model = tree_results['model']
    feature_names = tree_results['feature_names']
    
    tree_rules = export_text(
        model, 
        feature_names=list(feature_names),
        spacing=3,
        show_weights=True
    )
    
    return tree_rules

def analyze_decision_paths(tree_results, X_test, y_test, n_samples=5):
    """
    Analizuje ścieżki decyzyjne dla przykładowych meczów.
    
    Args:
        tree_results: Wyniki z funkcji train_decision_tree
        X_test: Dane testowe
        y_test: Etykiety testowe
        n_samples: Liczba próbek do analizy
    """
    model = tree_results['model']
    feature_names = tree_results['feature_names']
    
    # Losowo wybierz przykłady
    indices = np.random.choice(len(X_test), size=n_samples, replace=False)
    
    for i, idx in enumerate(indices):
        x = X_test[idx]
        true_label = y_test.iloc[idx] if hasattr(y_test, 'iloc') else y_test[idx]
        pred_label = model.predict([x])[0]
        
        # Znajdź ścieżkę decyzyjną
        node_indicator = model.decision_path([x])
        leaf_id = model.apply([x])[0]
        
        # Pobierz indeksy węzłów na ścieżce
        node_index = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]
        
        print(f"\nPrzykład {i+1}:")
        print(f"Prawdziwa etykieta: {'Wygrana' if true_label else 'Przegrana'}")
        print(f"Predykcja: {'Wygrana' if pred_label else 'Przegrana'}")
        
        print("Cechy przykładu:")
        for j, feature_name in enumerate(feature_names):
            print(f"  {feature_name}: {x[j]:.4f}")
        
        print("\nŚcieżka decyzyjna:")
        for node_id in node_index:
            if node_id == leaf_id:
                # Węzeł liściowy
                print(f"  Liść: klasa = {'Wygrana' if pred_label else 'Przegrana'}")
            else:
                # Węzeł decyzyjny
                feature = model.tree_.feature[node_id]
                threshold = model.tree_.threshold[node_id]
                
                if x[feature] <= threshold:
                    comparison = "<="
                else:
                    comparison = ">"
                
                print(f"  {feature_names[feature]} {comparison} {threshold:.4f}")

def optimize_decision_tree(X_train, y_train, X_test, y_test, feature_names):
    """
    Optymalizuje głębokość drzewa decyzyjnego.
    
    Args:
        X_train, y_train: Dane treningowe
        X_test, y_test: Dane testowe
        feature_names: Nazwy cech
    
    Returns:
        Wyniki najlepszego modelu i wykres dokładności
    """
    max_depths = range(1, 11)
    train_accuracy = []
    test_accuracy = []
    best_accuracy = 0
    best_model = None
    
    for depth in max_depths:
        # Trenowanie modelu
        dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
        dt.fit(X_train, y_train)
        
        # Ocena na zbiorze treningowym
        y_train_pred = dt.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        train_accuracy.append(train_acc)
        
        # Ocena na zbiorze testowym
        y_test_pred = dt.predict(X_test)
        test_acc = accuracy_score(y_test, y_test_pred)
        test_accuracy.append(test_acc)
        
        print(f"Głębokość = {depth}, Dokładność treningowa = {train_acc:.4f}, Dokładność testowa = {test_acc:.4f}")
        
        # Zapisz najlepszy model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_depth = depth
            best_model = dt
    
    # Wykres dokładności
    plt.figure(figsize=(10, 6))
    plt.plot(max_depths, train_accuracy, marker='o', linestyle='-', label='Zbiór treningowy')
    plt.plot(max_depths, test_accuracy, marker='s', linestyle='-', label='Zbiór testowy')
    plt.xlabel('Maksymalna głębokość drzewa')
    plt.ylabel('Dokładność')
    plt.title('Dokładność modelu w zależności od głębokości drzewa')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Zaznacz najlepszą głębokość
    plt.axvline(x=best_depth, color='r', linestyle='--', alpha=0.5, 
                label=f'Najlepsza głębokość = {best_depth}')
    plt.legend()
    
    print(f"\nNajlepsza głębokość drzewa: {best_depth}")
    print(f"Najlepsza dokładność testowa: {best_accuracy:.4f}")
    
    # Przygotuj wyniki
    results = {
        'model': best_model,
        'accuracy': best_accuracy,
        'feature_names': feature_names,
        'feature_importance': best_model.feature_importances_,
        'best_depth': best_depth,
        'accuracy_plot': plt.gcf()
    }
    
    return results