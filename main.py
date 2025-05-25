import matplotlib.pyplot as plt
import pandas as pd
import os
import time
import numpy as np
from data_loader import load_data, get_basic_stats
from model_training import train_and_evaluate_models
from visualizations import plot_model_comparison, plot_confusion_matrices, plot_feature_importance, plot_seasonal_accuracy
from data_preprocessing import prepare_features, prepare_predictive_features, prepare_predictive_features_sample
from error_analysis import analyze_errors
# Dodaj ten import na początku pliku, razem z innymi importami:
from sklearn.metrics import accuracy_score
from hyperparameter_tuning import optimize_hyperparameters
# Konfiguracja projektu
config = {
    'n_games': 10,          # Ile poprzednich meczów analizować
    'use_sample': True,     # Czy używać próbki zamiast pełnego zbioru
    'sample_size': 30000,   # Rozmiar próbki (jeśli use_sample=True)
    'sample_method': 'recent',  # Metoda próbkowania: 'recent', 'random' lub 'season'
    'use_cache': True,      # Czy używać cache'u
    'cache_file': 'cached_features.csv',  # Nazwa pliku cache'u
    'compare_with_original': True,  # Czy porównywać z oryginalnym modelem
    'save_results': True,   # Czy zapisywać wyniki
    'results_dir': 'wyniki'  # Katalog na wyniki
}

# Utwórz katalog na wyniki jeśli nie istnieje
if config['save_results'] and not os.path.exists(config['results_dir']):
    os.makedirs(config['results_dir'])

# Ścieżka do plików
data_path = "/home/user/Semestr4/ProjektyS4/MLProjekt/csv/"

# 1. Najpierw zaktualizuj nazwę pliku cache
if config['use_sample']:
    config['cache_file'] = f"cached_features_{config['sample_method']}_{config['sample_size']}.csv"
else:
    config['cache_file'] = "cached_features_full.csv"

# 2. Dopiero potem utwórz ścieżkę
cache_path = os.path.join(os.path.dirname(data_path), config['cache_file'])

print(f"Używam pliku cache: {cache_path}")
print(f"Plik istnieje: {os.path.exists(cache_path)}")

print("=" * 80)
print("NBA - System Predykcji Wyników Meczów")
print("=" * 80)

# Mierzenie czasu wykonania
start_time = time.time()

# ETAP 1: Wczytanie danych (10% postępu)
print("\n[1/8] Wczytywanie danych...")
data = load_data(data_path)

# Podstawowe statystyki
stats = get_basic_stats(data)
print(f"Liczba meczów: {stats['num_games']}")
print(f"Zakres dat: {stats['date_range'][0]} - {stats['date_range'][1]}")
print(f"Liczba drużyn: {stats['num_teams']}")
print(f"Procent zwycięstw gospodarzy: {stats['home_advantage']:.2%}")
print(f"Postęp: 10% (wczytano dane)")

# ETAP 2: Przygotowanie cech predykcyjnych (30% postępu)
print("\n[2/8] Przygotowywanie cech predykcyjnych...")

# Sprawdzenie czy istnieje plik cache'u
if config['use_cache'] and os.path.exists(cache_path):
    print(f"Wczytywanie cech z cache'u: {cache_path}")
    predictive_features_df = pd.read_csv(cache_path)
    # Konwersja kolumny z datą 
    if 'game_date' in predictive_features_df.columns:
        predictive_features_df['game_date'] = pd.to_datetime(predictive_features_df['game_date'])
else:
    # Wybór metody przygotowania cech
    if config['use_sample']:
        print(f"Używanie próbki {config['sample_size']} meczów (metoda: {config['sample_method']})")
        predictive_features_df = prepare_predictive_features_sample(
            data['game'], 
            data['team'],
            sample_size=config['sample_size'], 
            n_games=config['n_games'], 
            sample_method=config['sample_method']
        )
    else:
        print("Używanie pełnego zbioru danych (może potrwać długo!)")
        predictive_features_df = prepare_predictive_features(
            data['game'], 
            data['team'], 
            n_games=config['n_games']
        )
    
    # Zapisz do cache'u
    if config['use_cache']:
        print(f"Zapisywanie cech do cache'u: {cache_path}")
        predictive_features_df.to_csv(cache_path, index=False)

print(f"WAŻNE: Faktycznie wygenerowano {len(predictive_features_df)} meczów")
print(f"Postęp: 30% (przygotowano cechy)")

# ETAP 3: Wybór cech i czyszczenie danych (40% postępu)
print("\n[3/8] Wybór cech i czyszczenie danych...")

# Wybór cech predykcyjnych
selected_predictive_features = [
    'win_pct_diff', 'fg_pct_diff', 'fg3_pct_diff', 'ft_pct_diff', 
    'reb_diff', 'ast_diff', 'stl_diff', 'blk_diff', 'tov_diff', 'home_adv'
]

# Usunięcie NaN
features_clean = predictive_features_df.dropna(subset=selected_predictive_features)
print(f"Po usunięciu NaN pozostało {len(features_clean)} wierszy")
print(f"Postęp: 40% (wybrano cechy i oczyszczono dane)")

# ETAP 4: Podział na zbiory treningowy i testowy (50% postępu)
print("\n[4/8] Podział na zbiory treningowy i testowy...")

# Podział na zbiory treningowy i testowy
features_sorted = features_clean.sort_values('game_date')
train_size = int(0.8 * len(features_sorted))
train_data = features_sorted.iloc[:train_size]
test_data = features_sorted.iloc[train_size:]

X_train = train_data[selected_predictive_features]
y_train = train_data['home_win']
X_test = test_data[selected_predictive_features]
y_test = test_data['home_win']

print(f"Zbiór treningowy: {len(X_train)} meczów")
print(f"Zbiór testowy: {len(X_test)} meczów")
print(f"Postęp: 50% (podzielono dane)")

# ETAP 5: Trenowanie i ewaluacja modeli (70% postępu)
print("\n[5/8] Trenowanie i ewaluacja modeli...")

# Trenowanie i ewaluacja modeli
model_results, predictions, X_train_scaled, X_test_scaled = train_and_evaluate_models(
    X_train, y_train, X_test, y_test
)
print(f"Postęp: 60% (wytrenowano modele)")


# Dodaj ten import na początku pliku
from decision_tree_analysis import train_decision_tree, plot_decision_tree_visualization, plot_feature_importance_dt, get_decision_rules, analyze_decision_paths, optimize_decision_tree

# Dodaj ten kod po sekcji trenowania modeli:

# ETAP 5.5: Analiza z wykorzystaniem Drzewa Decyzyjnego
print("\n[5.5/8] Szczegółowa analiza za pomocą drzewa decyzyjnego...")

# Trenuj drzewo decyzyjne z domyślną głębokością
tree_results = train_decision_tree(
    X_train_scaled, y_train, 
    X_test_scaled, y_test, 
    selected_predictive_features
)

# Wizualizuj drzewo
if config['save_results']:
    tree_viz_path = os.path.join(config['results_dir'], 'decision_tree.png')
    importance_path = os.path.join(config['results_dir'], 'dt_feature_importance.png')
else:
    tree_viz_path = None
    importance_path = None

plot_decision_tree_visualization(tree_results, tree_viz_path)
plot_feature_importance_dt(tree_results, importance_path)

# Pokaż reguły decyzyjne
print("\nReguły decyzyjne wyekstrahowane z drzewa:")
rules_text = get_decision_rules(tree_results)
print(rules_text)

# Analiza ścieżek decyzyjnych dla przykładowych meczów
print("\nAnaliza ścieżek decyzyjnych dla przykładowych meczów:")
analyze_decision_paths(tree_results, X_test_scaled, y_test, n_samples=3)

# Optymalizacja głębokości drzewa
print("\nOptymalizacja głębokości drzewa decyzyjnego...")
optimized_tree = optimize_decision_tree(X_train_scaled, y_train, X_test_scaled, y_test, selected_predictive_features)

# Zapisz wykres dokładności dla różnych głębokości
if config['save_results']:
    accuracy_plot_path = os.path.join(config['results_dir'], 'dt_accuracy_vs_depth.png')
    optimized_tree['accuracy_plot'].savefig(accuracy_plot_path, dpi=300, bbox_inches='tight')
    print(f"Zapisano wykres dokładności do {accuracy_plot_path}")

# Dodaj optymalny model drzewa do wyników ogólnych
model_results['Decision Tree (Optimized)'] = {
    'model': optimized_tree['model'],
    'accuracy': optimized_tree['accuracy']
}

print(f"Postęp: 60% (przeprowadzono analizę drzewa decyzyjnego)")


# ETAP 6: Analiza reguł asocjacyjnych (70% postępu)
print("\n[6/8] Analiza reguł asocjacyjnych...")

try:
    from mlxtend.frequent_patterns import apriori, association_rules

    # Przygotowanie danych dla reguł asocjacyjnych
    print("Przygotowanie danych dla analizy reguł asocjacyjnych...")
    
    # Dyskretyzacja danych - konwersja zmiennych ciągłych na kategorialne
    features_for_rules = features_clean.copy()
    
    # Przekształć cechy ciągłe na kategorialne
    for feature in selected_predictive_features:
        # Podział na 3 kategorie: niski, średni, wysoki
        features_for_rules[f"{feature}_cat"] = pd.qcut(
            features_for_rules[feature], 
            q=3, 
            labels=["niski", "średni", "wysoki"],
            duplicates='drop'
        )
    
    # Dodaj informację o wyniku
    features_for_rules['wynik'] = features_for_rules['home_win'].map({True: 'wygrana_gospodarzy', False: 'wygrana_gości'})
    
    # Przekształć dane do formatu "koszyka" (one-hot encoding)
    categorical_features = [f"{feat}_cat" for feat in selected_predictive_features] + ['wynik']
    
    # One-hot encoding dla kategorii
    basket_data = pd.get_dummies(features_for_rules[categorical_features])
    
    print("Wyszukiwanie częstych wzorców...")
    # Znajdź częste wzorce
    frequent_itemsets = apriori(basket_data, min_support=0.1, use_colnames=True)
    print(f"Znaleziono {len(frequent_itemsets)} częstych wzorców")
    
    # Generowanie reguł asocjacyjnych
    if len(frequent_itemsets) > 0:
        print("Generowanie reguł asocjacyjnych...")
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
        
        # Filtruj reguły związane z wynikiem meczu
        win_rules = rules[rules['consequents'].apply(lambda x: 'wynik_wygrana_gospodarzy' in str(x))]
        loss_rules = rules[rules['consequents'].apply(lambda x: 'wynik_wygrana_gości' in str(x))]
        
        print(f"Znaleziono {len(rules)} reguł asocjacyjnych")
        print(f"W tym {len(win_rules)} reguł prowadzących do wygranej gospodarzy")
        print(f"I {len(loss_rules)} reguł prowadzących do wygranej gości")
        
        # Pokaż najlepsze reguły
        if len(win_rules) > 0:
            print("\nNajlepsze reguły dla wygranej gospodarzy (według lift):")
            top_win_rules = win_rules.sort_values('lift', ascending=False).head(5)
            for i, (_, rule) in enumerate(top_win_rules.iterrows(), 1):
                antecedents = list(rule['antecedents'])
                consequents = list(rule['consequents'])
                print(f"{i}. {', '.join(str(x) for x in antecedents)} => {', '.join(str(x) for x in consequents)}")
                print(f"   Lift: {rule['lift']:.3f}, Confidence: {rule['confidence']:.3f}, Support: {rule['support']:.3f}")
    else:
        print("Nie znaleziono częstych wzorców (spróbuj zmniejszyć min_support)")
        rules = pd.DataFrame()
        
    print(f"Postęp: 70% (przeprowadzono analizę reguł asocjacyjnych)")
    
except ImportError:
    print("Nie znaleziono pakietu mlxtend - pomijam analizę reguł asocjacyjnych")
    print("Aby zainstalować: pip install mlxtend")
    rules = pd.DataFrame()
    print(f"Postęp: 70% (pominięto analizę reguł asocjacyjnych)")


# ETAP 7: Wizualizacje i analiza wyników (90% postępu)
print("\n[7/8] Tworzenie wizualizacji i analiza wyników...")

# Wizualizacja wyników
if config['save_results']:
    output_path = os.path.join(config['results_dir'], 'model_comparison_predictive.png')
else:
    output_path = None

plot_model_comparison(model_results, output_path)

# Tworzenie macierzy pomyłek
if config['save_results']:
    output_path = os.path.join(config['results_dir'], 'confusion_matrices.png')
else:
    output_path = None

plot_confusion_matrices(model_results, y_test, predictions, output_path)

# Analiza ważności cech

# Analiza ważności cech
best_model_name = max(model_results.items(), key=lambda x: x[1]['accuracy'])[0]
# Dodaj tę linię, aby zdefiniować best_model dla wszystkich typów modeli
best_model = model_results[best_model_name]['model']

if best_model_name in ['Random Forest', 'Gradient Boosting']:
    if config['save_results']:
        output_path = os.path.join(config['results_dir'], 'feature_importance.png')
    else:
        output_path = None
    
    plot_feature_importance(
        selected_predictive_features, 
        best_model.feature_importances_,
        best_model_name,
        output_path
    )

# Analiza sezonowa (jeśli istnieje kolumna game_date)
if 'game_date' in features_clean.columns:
    features_clean['season'] = pd.to_datetime(features_clean['game_date']).dt.year
    seasons = features_clean['season'].unique()
    
    # Słownik do przechowywania dokładności dla każdego sezonu
    seasonal_results = {}
    
    print("\nAnaliza sezonowa:")
    for season in seasons:
        season_data = features_clean[features_clean['season'] == season]
        
        if len(season_data) < 50:  # Pomiń sezony z małą liczbą meczów
            continue
            
        # Podziel dane z danego sezonu
        season_train_size = int(0.8 * len(season_data))
        season_train = season_data.iloc[:season_train_size]
        season_test = season_data.iloc[season_train_size:]
        
        if len(season_test) < 10:  # Pomiń jeśli zbyt mały zbiór testowy
            continue
            
        X_season_train = season_train[selected_predictive_features]
        y_season_train = season_train['home_win']
        X_season_test = season_test[selected_predictive_features]
        y_season_test = season_test['home_win']
        
        # Użyj najlepszego modelu
        best_model.fit(X_season_train, y_season_train)
        y_season_pred = best_model.predict(X_season_test)
        season_accuracy = (y_season_pred == y_season_test).mean()
        
        seasonal_results[season] = season_accuracy
        print(f"  Sezon {season}: dokładność = {season_accuracy:.4f} (liczba meczów: {len(season_data)})")
    
    # Wizualizacja wyników sezonowych
    if len(seasonal_results) > 1:
        if config['save_results']:
            output_path = os.path.join(config['results_dir'], 'seasonal_accuracy.png')
        else:
            output_path = None
        plot_seasonal_accuracy(seasonal_results, output_path)

print(f"Postęp: 90% (utworzono wizualizacje)")
# ETAP 7.5: Optymalizacja hiperparametrów
print("\n[7.5/8] Optymalizacja hiperparametrów najlepszego modelu...")

try:
    from hyperparameter_tuning import optimize_hyperparameters
    
    # Optymalizuj hiperparametry dla najlepszego modelu
    print(f"Optymalizacja hiperparametrów dla modelu {best_model_name}...")
    best_tuned_model, best_params, best_cv_score = optimize_hyperparameters(
        best_model_name, model_results[best_model_name]['model'], 
        X_train_scaled, y_train, cv=5
    )

    # Test zoptymalizowanego modelu
    y_pred_tuned = best_tuned_model.predict(X_test_scaled)
    tuned_accuracy = accuracy_score(y_test, y_pred_tuned)

    print(f"Dokładność modelu przed strojeniem: {model_results[best_model_name]['accuracy']:.4f}")
    print(f"Dokładność modelu po strojeniu: {tuned_accuracy:.4f}")
    
    # Zastąp najlepszy model zoptymalizowanym modelem
    model_results[best_model_name]['model'] = best_tuned_model
    model_results[best_model_name]['accuracy'] = tuned_accuracy
    predictions[best_model_name] = y_pred_tuned
    
    print(f"Postęp: 92% (zoptymalizowano hiperparametry)")

except ImportError:
    print("Nie znaleziono modułu hyperparameter_tuning - pomijam optymalizację hiperparametrów")
    print(f"Postęp: 92% (pominięto optymalizację hiperparametrów)")

# ETAP 7.6: Analiza błędów modelu
print("\n[7.6/8] Analiza błędów modelu...")

try:
    from error_analysis import analyze_errors
    
    # Analiza błędów najlepszego modelu
    if config['save_results']:
        output_path = os.path.join(config['results_dir'], 'error_analysis.png')
    else:
        output_path = None
    
    # Wykonaj analizę błędów
    fig, error_df = analyze_errors(
        X_test_scaled, y_test, predictions[best_model_name], selected_predictive_features
    )
    
    # Zapisz wykres
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Wykres analizy błędów zapisano jako {output_path}")
    
    # Pokaż kilka przypadków błędnej klasyfikacji
    if len(error_df) > 0:
        print("\nPrzykłady błędnych klasyfikacji:")
        for i, (_, row) in enumerate(error_df.head(5).iterrows()):
            print(f"Przypadek {i+1}:")
            for feature in selected_predictive_features:
                print(f"  {feature}: {row[feature]:.4f}")
            print(f"  Rzeczywisty wynik: {'Wygrana' if row['actual'] else 'Porażka'}")
            print(f"  Przewidywany wynik: {'Wygrana' if row['predicted'] else 'Porażka'}\n")
    
    print(f"Postęp: 95% (przeprowadzono analizę błędów)")

except ImportError:
    print("Nie znaleziono modułu error_analysis - pomijam analizę błędów")
    print(f"Postęp: 95% (pominięto analizę błędów)")

# ETAP 7.7: Krzywa uczenia
print("\n[7.7/8] Generowanie krzywej uczenia...")

try:
    from visualizations import plot_learning_curves
    
    # Generowanie krzywej uczenia dla najlepszego modelu
    if config['save_results']:
        output_path = os.path.join(config['results_dir'], 'learning_curve.png')
    else:
        output_path = None
    
    # Wygeneruj krzywą uczenia
    plot_learning_curves(
        model_results[best_model_name]['model'],
        X_train_scaled, y_train,
        cv=5,
        output_path=output_path
    )
    
    if output_path:
        print(f"Wykres krzywej uczenia zapisano jako {output_path}")
    
    print(f"Postęp: 98% (wygenerowano krzywą uczenia)")

except Exception as e:
    print(f"Błąd przy generowaniu krzywej uczenia: {e}")
    print(f"Postęp: 98% (pominięto krzywą uczenia)")
# ETAP 8: Porównanie z oryginalnym modelem i podsumowanie (100% postępu)
print("\n[8/8] Porównanie modeli i podsumowanie...")

# Porównanie z oryginalnym modelem (opcjonalnie)
if config['compare_with_original']:
    print("\nPorównanie z oryginalnym podejściem (używającym statystyk z meczu)...")
    original_features_df = prepare_features(data['game'], data['other_stats'])
    original_selected_features = [
        'fg_pct_diff', 'fg3_pct_diff', 'ft_pct_diff', 
        'reb_diff', 'ast_diff', 'stl_diff', 'blk_diff', 'tov_diff', 'pf_diff'
    ]
    original_clean = original_features_df.dropna(subset=original_selected_features)
    print(f"Dane oryginalne po usunięciu NaN: {len(original_clean)} wierszy")
    
    # Podział na zbiory treningowy i testowy
    original_sorted = original_clean.sort_values('game_date')
    train_size = int(0.8 * len(original_sorted))
    original_train_data = original_sorted.iloc[:train_size]
    original_test_data = original_sorted.iloc[train_size:]
    
    X_train_original = original_train_data[original_selected_features]
    y_train_original = original_train_data['home_win']
    X_test_original = original_test_data[original_selected_features]
    y_test_original = original_test_data['home_win']
    
    print(f"Zbiór treningowy (oryginalny): {len(X_train_original)} meczów")
    print(f"Zbiór testowy (oryginalny): {len(X_test_original)} meczów")
    
    # Trenowanie i ewaluacja modeli na oryginalnych cechach
    original_results, original_predictions, _, _ = train_and_evaluate_models(
        X_train_original, y_train_original, X_test_original, y_test_original
    )
    
    # Porównanie najlepszych wyników
    best_predictive = max(model_results.items(), key=lambda x: x[1]['accuracy'])
    best_original = max(original_results.items(), key=lambda x: x[1]['accuracy'])
    
    print("\nPodsumowanie porównania:")
    print(f"Najlepszy model predykcyjny: {best_predictive[0]} - dokładność: {best_predictive[1]['accuracy']:.4f}")
    print(f"Najlepszy model oryginalny: {best_original[0]} - dokładność: {best_original[1]['accuracy']:.4f}")
    
    # Zapisz podsumowanie do pliku
    if config['save_results']:
        with open(os.path.join(config['results_dir'], 'podsumowanie.txt'), 'w') as f:
            f.write("PODSUMOWANIE EKSPERYMENTU\n\n")
            f.write(f"Liczba analizowanych meczów: {len(features_clean)}\n")
            f.write(f"Liczba cech predykcyjnych: {len(selected_predictive_features)}\n\n")
            
            f.write("Wyniki modelu predykcyjnego (używającego statystyk historycznych):\n")
            for name, result in model_results.items():
                f.write(f"- {name}: dokładność = {result['accuracy']:.4f}\n")
            
            f.write("\nWyniki modelu oryginalnego (używającego statystyk z meczu):\n")
            for name, result in original_results.items():
                f.write(f"- {name}: dokładność = {result['accuracy']:.4f}\n")
            
            f.write(f"\nNajlepszy model predykcyjny: {best_predictive[0]} - dokładność: {best_predictive[1]['accuracy']:.4f}\n")
            f.write(f"Najlepszy model oryginalny: {best_original[0]} - dokładność: {best_original[1]['accuracy']:.4f}\n")
            
            # Dodaj informacje o regułach asocjacyjnych
            if len(rules) > 0:
                f.write("\n\nANALIZA REGUŁ ASOCJACYJNYCH\n\n")
                f.write(f"Znaleziono {len(rules)} reguł asocjacyjnych.\n\n")
                
                # Sortuj według lift
                top_rules_df = rules.sort_values('lift', ascending=False).head(10)
                f.write("Top 10 reguł asocjacyjnych według lift:\n")
                for i, (_, rule) in enumerate(top_rules_df.iterrows(), 1):
                    antecedents = list(rule['antecedents'])
                    consequents = list(rule['consequents'])
                    f.write(f"{i}. {antecedents} => {consequents}\n")
                    f.write(f"   Lift: {rule['lift']:.3f}, Confidence: {rule['confidence']:.3f}, Support: {rule['support']:.3f}\n")

# Zapisz model
if config['save_results'] and best_model_name in model_results:
    try:
        import joblib
        best_model = model_results[best_model_name]['model']
        model_path = os.path.join(config['results_dir'], f'best_model_{best_model_name.replace(" ", "_")}.pkl')
        joblib.dump(best_model, model_path)
        print(f"\nZapisano najlepszy model ({best_model_name}) do {model_path}")
    except ImportError:
        print("\nBrak pakietu joblib - nie zapisano modelu")
        print("Aby zainstalować: pip install joblib")

# Czas wykonania
end_time = time.time()
execution_time = end_time - start_time
print(f"\nCzas wykonania: {execution_time:.2f} sekund ({execution_time/60:.2f} minut)")
print(f"Postęp: 100% (zakończono)")
print("\nAnaliza wyników NBA zakończona pomyślnie!")