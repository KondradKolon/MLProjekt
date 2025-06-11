import matplotlib.pyplot as plt
import pandas as pd
import os
import time
import numpy as np
from data_loader import load_data, get_basic_stats
from model_training import train_and_evaluate_models
from visualizations import plot_model_comparison, plot_confusion_matrices, plot_feature_importance, plot_seasonal_accuracy, plot_feature_distributions
from data_preprocessing import prepare_features, prepare_predictive_features, prepare_predictive_features_sample
from error_analysis import analyze_errors
from sklearn.metrics import accuracy_score
from hyperparameter_tuning import optimize_hyperparameters
from decision_tree_analysis import train_decision_tree, plot_decision_tree_visualization, plot_feature_importance_dt, get_decision_rules, analyze_decision_paths, optimize_decision_tree

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
    'results_dir': 'wyniki',  # Katalog na wyniki
    'optimize_multiple_models': True,  # Nowa opcja - czy optymalizować wiele modeli
    'models_to_optimize': ['Naive Bayes', 'Random Forest', 'Gradient Boosting', 'Logistic Regression', 'KNN']  # Lista modeli do optymalizacji
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
print("\n[1/9] Wczytywanie danych...")
data = load_data(data_path)

# Podstawowe statystyki
stats = get_basic_stats(data)
print(f"Liczba meczów: {stats['num_games']}")
print(f"Zakres dat: {stats['date_range'][0]} - {stats['date_range'][1]}")
print(f"Liczba drużyn: {stats['num_teams']}")
print(f"Procent zwycięstw gospodarzy: {stats['home_advantage']:.2%}")
print(f"Postęp: 10% (wczytano dane)")

# ETAP 2: Przygotowanie cech predykcyjnych (30% postępu)
print("\n[2/9] Przygotowywanie cech predykcyjnych...")

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
print("\n[3/9] Wybór cech i czyszczenie danych...")

# Wybór cech predykcyjnych
selected_predictive_features = [
    'win_pct_diff', 'fg_pct_diff', 'fg3_pct_diff', 'ft_pct_diff', 
    'reb_diff', 'ast_diff', 'stl_diff', 'blk_diff', 'tov_diff', 'home_adv'
]

# Usunięcie NaN
features_clean = predictive_features_df.dropna(subset=selected_predictive_features)
print(f"Po usunięciu NaN pozostało {len(features_clean)} wierszy")
print(f"Postęp: 40% (wybrano cechy i oczyszczono dane)")

# ETAP 3.5: Generowanie wykresów rozkładu cech
print("\n[3.5/9] Generowanie wykresów rozkładu cech predykcyjnych...")
if config['save_results']:
    features_dist_path = os.path.join(config['results_dir'], 'feature_distributions.png')
    plot_feature_distributions(features_clean, selected_predictive_features, features_dist_path)
    print(f"Zapisano wykres rozkładu cech do {features_dist_path}")
else:
    plot_feature_distributions(features_clean, selected_predictive_features)

print("Postęp: 45% (wygenerowano wykresy rozkładu cech)")

# ETAP 4: Podział na zbiory treningowy i testowy (50% postępu)
print("\n[4/9] Podział na zbiory treningowy i testowy...")

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

# ETAP 5: Trenowanie i ewaluacja modeli (60% postępu)
print("\n[5/9] Trenowanie i ewaluacja modeli...")

# Trenowanie i ewaluacja modeli
model_results, predictions, X_train_scaled, X_test_scaled = train_and_evaluate_models(
    X_train, y_train, X_test, y_test
)
print(f"Postęp: 60% (wytrenowano modele)")

# ETAP 6: Optymalizacja hiperparametrów dla wielu modeli (70% postępu)
print("\n[6/9] Optymalizacja hiperparametrów dla wielu modeli...")

# Funkcja do generowania wykresu porównawczego dla parametrów
def plot_hyperparameter_comparison(model_name, param_name, param_values, accuracies, output_path=None):
    """Generuje wykres porównawczy dla różnych wartości hiperparametrów"""
    plt.figure(figsize=(10, 6))
    plt.plot(param_values, accuracies, marker='o', linestyle='-', linewidth=2)
    plt.grid(True, alpha=0.3)
    plt.title(f'Wpływ parametru {param_name} na dokładność modelu {model_name}', fontsize=14)
    plt.xlabel(param_name, fontsize=12)
    plt.ylabel('Dokładność modelu', fontsize=12)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Zapisano wykres parametrów do {output_path}")
    else:
        plt.show()

if config['optimize_multiple_models']:
    # Słownik do przechowywania wyników optymalizacji
    optimization_results = {}
    param_comparison_charts = {}
    
    # Utwórz katalog na wyniki optymalizacji
    hp_tuning_dir = os.path.join(config['results_dir'], 'hyperparameter_tuning')
    if config['save_results'] and not os.path.exists(hp_tuning_dir):
        os.makedirs(hp_tuning_dir)
    
    # Optymalizuj wybrane modele
    for model_name in config['models_to_optimize']:
        if model_name in model_results:
            print(f"\nOptymalizacja modelu: {model_name}")
            
            # Wykonaj standardową optymalizację
            tuned_model, best_params, best_score = optimize_hyperparameters(
                model_name, model_results[model_name]['model'], 
                X_train_scaled, y_train, cv=5
            )
            
            # Test modelu na zbiorze testowym
            y_pred_tuned = tuned_model.predict(X_test_scaled)
            tuned_accuracy = accuracy_score(y_test, y_pred_tuned)
            
            # Zapisz wyniki
            optimization_results[model_name] = {
                'model': tuned_model,
                'best_params': best_params,
                'cv_accuracy': best_score,
                'test_accuracy': tuned_accuracy,
                'improvement': tuned_accuracy - model_results[model_name]['accuracy']
            }
            
            print(f"Dokładność modelu {model_name} przed strojeniem: {model_results[model_name]['accuracy']:.4f}")
            print(f"Dokładność modelu {model_name} po strojeniu: {tuned_accuracy:.4f}")
            print(f"Poprawa: {tuned_accuracy - model_results[model_name]['accuracy']:.4f} ({(tuned_accuracy - model_results[model_name]['accuracy'])*100:.2f}%)")
            
            # Wygeneruj wykresy dla najważniejszych parametrów
            if model_name == 'Random Forest':
                # Wykres dla liczby drzew
                n_estimators = [50, 100, 200, 300, 500]
                acc_scores = []
                
                for n_est in n_estimators:
                    from sklearn.ensemble import RandomForestClassifier
                    model = RandomForestClassifier(n_estimators=n_est, random_state=42)
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    acc = accuracy_score(y_test, y_pred)
                    acc_scores.append(acc)
                
                if config['save_results']:
                    output_path = os.path.join(hp_tuning_dir, f'{model_name}_n_estimators.png')
                else:
                    output_path = None
                
                plot_hyperparameter_comparison(model_name, 'n_estimators', n_estimators, acc_scores, output_path)
                
            elif model_name == 'KNN':
                # Wykres dla liczby sąsiadów
                neighbors = [3, 5, 7, 9, 11, 15]
                acc_scores = []
                
                for n in neighbors:
                    from sklearn.neighbors import KNeighborsClassifier
                    model = KNeighborsClassifier(n_neighbors=n)
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    acc = accuracy_score(y_test, y_pred)
                    acc_scores.append(acc)
                
                if config['save_results']:
                    output_path = os.path.join(hp_tuning_dir, f'{model_name}_n_neighbors.png')
                else:
                    output_path = None
                
                plot_hyperparameter_comparison(model_name, 'n_neighbors', neighbors, acc_scores, output_path)
                
            elif model_name == 'Gradient Boosting':
                # Wykres dla learning_rate
                learning_rates = [0.01, 0.05, 0.1, 0.2, 0.3]
                acc_scores = []
                
                for lr in learning_rates:
                    from sklearn.ensemble import GradientBoostingClassifier
                    model = GradientBoostingClassifier(learning_rate=lr, random_state=42)
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    acc = accuracy_score(y_test, y_pred)
                    acc_scores.append(acc)
                
                if config['save_results']:
                    output_path = os.path.join(hp_tuning_dir, f'{model_name}_learning_rate.png')
                else:
                    output_path = None
                
                plot_hyperparameter_comparison(model_name, 'learning_rate', learning_rates, acc_scores, output_path)
                
            elif model_name == 'Logistic Regression':
                # Wykres dla parametru C
                c_values = [0.01, 0.1, 1.0, 10.0, 100.0]
                acc_scores = []
                
                for c in c_values:
                    from sklearn.linear_model import LogisticRegression
                    model = LogisticRegression(C=c, max_iter=1000, random_state=42)
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    acc = accuracy_score(y_test, y_pred)
                    acc_scores.append(acc)
                
                if config['save_results']:
                    output_path = os.path.join(hp_tuning_dir, f'{model_name}_C_param.png')
                else:
                    output_path = None
                
                plot_hyperparameter_comparison(model_name, 'C (regularization)', c_values, acc_scores, output_path)
                
            elif model_name == 'Naive Bayes':
                # Wykres dla var_smoothing
                var_smoothing_values = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
                acc_scores = []
                
                for vs in var_smoothing_values:
                    from sklearn.naive_bayes import GaussianNB
                    model = GaussianNB(var_smoothing=vs)
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    acc = accuracy_score(y_test, y_pred)
                    acc_scores.append(acc)
                
                if config['save_results']:
                    output_path = os.path.join(hp_tuning_dir, f'{model_name}_var_smoothing.png')
                else:
                    output_path = None
                
                plot_hyperparameter_comparison(model_name, 'var_smoothing', [str(vs) for vs in var_smoothing_values], acc_scores, output_path)
            
            # Aktualizuj model w oryginalnych wynikach
            model_results[model_name]['model'] = tuned_model
            model_results[model_name]['accuracy'] = tuned_accuracy
            predictions[model_name] = y_pred_tuned
    
    # Wykres porównawczy poprawy dokładności dla różnych modeli
    if len(optimization_results) > 0:
        plt.figure(figsize=(12, 8))
        
        # Przygotuj dane do wykresu
        model_names = list(optimization_results.keys())
        base_accuracies = [model_results[name]['accuracy'] - optimization_results[name]['improvement'] for name in model_names]
        tuned_accuracies = [model_results[name]['accuracy'] for name in model_names]
        
        # Ustaw indeksy dla słupków
        x = np.arange(len(model_names))
        width = 0.35
        
        # Utwórz wykres słupkowy
        fig, ax = plt.subplots(figsize=(12, 7))
        rects1 = ax.bar(x - width/2, base_accuracies, width, label='Domyślne parametry', color='skyblue')
        rects2 = ax.bar(x + width/2, tuned_accuracies, width, label='Zoptymalizowane parametry', color='lightgreen')
        
        # Dodaj etykiety, tytuł i legendę
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Dokładność', fontsize=12)
        ax.set_title('Porównanie dokładności modeli przed i po optymalizacji parametrów', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        
        # Dodaj wartości nad słupkami
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.4f}',
                            xy=(rect.get_x() + rect.get_width()/2, height),
                            xytext=(0, 3),  # 3 punkty nad słupkiem
                            textcoords="offset points",
                            ha='center', va='bottom')
        
        autolabel(rects1)
        autolabel(rects2)
        
        plt.tight_layout()
        
        if config['save_results']:
            output_path = os.path.join(hp_tuning_dir, 'model_improvement_comparison.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Zapisano wykres porównawczy poprawy dokładności do {output_path}")
        
        # Utwórz tabelę z podsumowaniem optymalizacji
        optimization_summary = pd.DataFrame({
            'Model': model_names,
            'Dokładność (domyślne)': base_accuracies,
            'Dokładność (zoptymalizowane)': tuned_accuracies,
            'Poprawa': [optimization_results[name]['improvement'] for name in model_names],
            'Poprawa (%)': [optimization_results[name]['improvement'] * 100 for name in model_names]
        })
        
        # Zapisz tabelę
        if config['save_results']:
            summary_path = os.path.join(hp_tuning_dir, 'optimization_summary.csv')
            optimization_summary.to_csv(summary_path, index=False)
            print(f"Zapisano podsumowanie optymalizacji do {summary_path}")
        
        print("\nPodsumowanie optymalizacji parametrów:")
        print(optimization_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        
        # Znajdź najlepszy model po optymalizacji
        best_model_name = max(model_results.items(), key=lambda x: x[1]['accuracy'])[0]
        print(f"\nNajlepszy model po optymalizacji: {best_model_name} z dokładnością {model_results[best_model_name]['accuracy']:.4f}")
    
    print(f"Postęp: 70% (zoptymalizowano parametry dla {len(optimization_results)} modeli)")
else:
    # Standardowa optymalizacja tylko dla najlepszego modelu
    try:
        # Znajdź najlepszy model
        best_model_name = max(model_results.items(), key=lambda x: x[1]['accuracy'])[0]
        best_model = model_results[best_model_name]['model']
        
        # Optymalizuj hiperparametry dla najlepszego modelu
        print(f"Optymalizacja hiperparametrów dla modelu {best_model_name}...")
        best_tuned_model, best_params, best_cv_score = optimize_hyperparameters(
            best_model_name, best_model, X_train_scaled, y_train, cv=5
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
        
        print(f"Postęp: 70% (zoptymalizowano hiperparametry dla najlepszego modelu)")
    except ImportError:
        print("Nie znaleziono modułu hyperparameter_tuning - pomijam optymalizację hiperparametrów")
        print(f"Postęp: 70% (pominięto optymalizację hiperparametrów)")

# ETAP 7: Analiza reguł asocjacyjnych i drzewa decyzyjnego (80% postępu)
print("\n[7/9] Analiza reguł asocjacyjnych i drzewa decyzyjnego...")

# Najpierw wykonaj analizę drzewa decyzyjnego
print("Przeprowadzanie analizy drzewa decyzyjnego...")
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

# Teraz wykonaj analizę reguł asocjacyjnych
try:
    # Używamy funkcji z pliku association_rules.py zamiast bezpośredniej implementacji
    from association_rules import discretize_features, prepare_data_for_rules, find_association_rules, visualize_top_rules

    print("Przygotowanie danych dla analizy reguł asocjacyjnych...")
    
    # Dyskretyzacja danych numerycznych
    features_for_rules = discretize_features(
        features_clean, 
        columns=selected_predictive_features,
        n_bins=3
    )
    
    # Przygotowanie danych do analizy reguł
    rules_data = prepare_data_for_rules(
        game_df=data['game'],
        features_df=features_for_rules,
        target_col='home_win'
    )
    
    print("Wyszukiwanie reguł asocjacyjnych...")
    # Znajdź reguły asocjacyjne
    rules = find_association_rules(rules_data, min_support=0.1, min_confidence=0.7)
    
    if len(rules) > 0:
        print(f"Znaleziono {len(rules)} reguł asocjacyjnych")
        
        # Filtruj reguły związane z wynikiem meczu
        win_rules = rules[rules['consequents'].apply(lambda x: 'home_win=win' in str(x))]
        loss_rules = rules[rules['consequents'].apply(lambda x: 'home_win=loss' in str(x))]
        
        print(f"W tym {len(win_rules)} reguł prowadzących do wygranej gospodarzy")
        print(f"I {len(loss_rules)} reguł prowadzących do wygranej gości")
        
        # Wizualizacja najważniejszych reguł
        if config['save_results']:
            win_rules_path = os.path.join(config['results_dir'], 'win_rules.png')
            loss_rules_path = os.path.join(config['results_dir'], 'loss_rules.png')
        else:
            win_rules_path = None
            loss_rules_path = None
        
        # Wizualizuj reguły prowadzące do wygranej gospodarzy
        top_win_rules = visualize_top_rules(
            rules, 
            target='home_win=win', 
            n_rules=10, 
            output_path=win_rules_path
        )
        
        # Wizualizuj reguły prowadzące do wygranej gości
        top_loss_rules = visualize_top_rules(
            rules, 
            target='home_win=loss', 
            n_rules=10, 
            output_path=loss_rules_path
        )
        
        # Wyświetl najlepsze reguły w konsoli
        if len(win_rules) > 0:
            print("\nNajlepsze reguły dla wygranej gospodarzy (według lift):")
            for i, (_, rule) in enumerate(top_win_rules.head(5).iterrows(), 1):
                antecedents = list(rule['antecedents'])
                consequents = list(rule['consequents'])
                print(f"{i}. {', '.join(str(x) for x in antecedents)} => {', '.join(str(x) for x in consequents)}")
                print(f"   Lift: {rule['lift']:.3f}, Confidence: {rule['confidence']:.3f}, Support: {rule['support']:.3f}")
    else:
        print("Nie znaleziono żadnych reguł (spróbuj zmniejszyć min_support)")
    
except ImportError:
    print("Nie znaleziono pakietu mlxtend - pomijam analizę reguł asocjacyjnych")
    print("Aby zainstalować: pip install mlxtend")
    rules = pd.DataFrame()

print(f"Postęp: 80% (przeprowadzono analizę reguł asocjacyjnych i drzewa decyzyjnego)")

# ETAP 8: Wizualizacje i analiza wyników (90% postępu)
print("\n[8/9] Tworzenie wizualizacji i analiza wyników...")

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

# Znajdź najlepszy model
best_model_name = max(model_results.items(), key=lambda x: x[1]['accuracy'])[0]
best_model = model_results[best_model_name]['model']

# Analiza ważności cech
if best_model_name in ['Random Forest', 'Gradient Boosting', 'Decision Tree (Optimized)']:
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

# Analiza błędów modelu
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

except ImportError:
    print("Nie znaleziono modułu error_analysis - pomijam analizę błędów")

# Generowanie krzywej uczenia
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

except Exception as e:
    print(f"Błąd przy generowaniu krzywej uczenia: {e}")

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

print(f"Postęp: 90% (utworzono wizualizacje i przeprowadzono analizy)")

# ETAP 9: Porównanie z oryginalnym modelem i podsumowanie (100% postępu)
print("\n[9/9] Porównanie modeli i podsumowanie...")

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
    
    # Wizualizacja porównania
    plt.figure(figsize=(12, 8))
    
    # Przygotuj dane do wykresu
    model_names = list(set(list(model_results.keys()) + list(original_results.keys())))
    predictive_accuracies = [model_results[name]['accuracy'] if name in model_results else 0 for name in model_names]
    original_accuracies = [original_results[name]['accuracy'] if name in original_results else 0 for name in model_names]
    
    # Ustaw indeksy dla słupków
    x = np.arange(len(model_names))
    width = 0.35
    
    # Utwórz wykres słupkowy
    fig, ax = plt.subplots(figsize=(14, 8))
    rects1 = ax.bar(x - width/2, predictive_accuracies, width, label='Model predykcyjny (dane historyczne)', color='skyblue')
    rects2 = ax.bar(x + width/2, original_accuracies, width, label='Model oryginalny (statystyki z meczu)', color='lightcoral')
    
    # Dodaj etykiety, tytuł i legendę
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Dokładność', fontsize=12)
    ax.set_title('Porównanie dokładności modeli - predykcja vs klasyfikacja po meczu', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    
    # Dodaj wartości nad słupkami
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:  # Dodaj etykietę tylko jeśli wartość jest większa od 0
                ax.annotate(f'{height:.4f}',
                            xy=(rect.get_x() + rect.get_width()/2, height),
                            xytext=(0, 3),  # 3 punkty nad słupkiem
                            textcoords="offset points",
                            ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    
    if config['save_results']:
        output_path = os.path.join(config['results_dir'], 'predictive_vs_original_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Zapisano wykres porównawczy modeli do {output_path}")
    
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
            
            # Dodaj informacje o optymalizacji hiperparametrów
            if config['optimize_multiple_models'] and 'optimization_results' in locals():
                f.write("\n\nWYNIKI OPTYMALIZACJI HIPERPARAMETRÓW\n\n")
                for model_name, results in optimization_results.items():
                    f.write(f"Model: {model_name}\n")
                    f.write(f"- Dokładność przed optymalizacją: {model_results[model_name]['accuracy'] - results['improvement']:.4f}\n")
                    f.write(f"- Dokładność po optymalizacji: {model_results[model_name]['accuracy']:.4f}\n")
                    f.write(f"- Poprawa: {results['improvement']:.4f} ({results['improvement']*100:.2f}%)\n")
                    f.write(f"- Najlepsze parametry: {results['best_params']}\n\n")
            
            # Dodaj informacje o regułach asocjacyjnych
            if 'rules' in locals() and len(rules) > 0:
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

# Zapisz najlepszy model
if config['save_results'] and 'best_model_name' in locals():
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