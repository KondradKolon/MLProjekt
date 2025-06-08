import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns

def create_categorical_features(features_df):
    """Przekształca cechy numeryczne na kategorie dla reguł asocjacyjnych"""
    categorical_df = features_df.copy()
    
    # Kategoryzuj różnice statystyk
    stat_columns = ['fg_pct_diff', 'fg3_pct_diff', 'ft_pct_diff', 'reb_diff', 
                    'ast_diff', 'stl_diff', 'blk_diff', 'tov_diff']
    
    for col in stat_columns:
        if col in categorical_df.columns:
            # Podziel na 3 kategorie: słabe, średnie, dobre
            categorical_df[f'{col}_cat'] = pd.cut(
                categorical_df[col], 
                bins=3, 
                labels=[f'{col}_weak', f'{col}_medium', f'{col}_strong']
            )
    
    # Dodaj kategorię przewagi
    if 'home_adv' in categorical_df.columns:
        categorical_df['home_adv_cat'] = pd.cut(
            categorical_df['home_adv'], 
            bins=[-np.inf, -0.1, 0.1, np.inf], 
            labels=['away_favored', 'even_match', 'home_favored']
        )
    
    # Dodaj wynik jako kategorię
    categorical_df['result'] = categorical_df['home_win'].map({
        1: 'home_wins', 
        0: 'away_wins'
    })
    
    # Wybierz tylko kolumny kategoryczne
    categorical_columns = [col for col in categorical_df.columns if col.endswith('_cat') or col == 'result']
    return categorical_df[categorical_columns]

def prepare_transactions(categorical_df):
    """Przygotowuje dane w formacie transakcji dla reguł asocjacyjnych"""
    # Zamień NaN na None i konwertuj na string
    transactions = []
    
    for _, row in categorical_df.iterrows():
        transaction = []
        for col, value in row.items():
            if pd.notna(value):
                transaction.append(str(value))
        transactions.append(transaction)
    
    return transactions

def generate_nba_association_rules(features_df, min_support=0.01, min_confidence=0.6, output_dir='wyniki'):
    """Generuje reguły asocjacyjne dla danych NBA"""
    print("🏀 Tworzenie reguł asocjacyjnych NBA...")
    
    # Przekształć dane na kategorie
    categorical_df = create_categorical_features(features_df)
    print(f"Utworzono {len(categorical_df.columns)} cech kategorycznych")
    
    # Przygotuj transakcje
    transactions = prepare_transactions(categorical_df)
    print(f"Przygotowano {len(transactions)} transakcji")
    
    # Kodowanie transakcji
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    
    print(f"Macierz kodowana: {df_encoded.shape}")
    print(f"Dostępne elementy: {list(te.columns_)}")
    
    # Znajdź częste zbiory elementów
    print(f"Szukanie częstych zbiorów z min_support={min_support}...")
    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
    
    if len(frequent_itemsets) == 0:
        print("⚠️ Nie znaleziono częstych zbiorów. Zmniejsz min_support.")
        return None, None
    
    print(f"Znaleziono {len(frequent_itemsets)} częstych zbiorów")
    
    # Generuj reguły asocjacyjne
    print(f"Generowanie reguł z min_confidence={min_confidence}...")
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    
    if len(rules) == 0:
        print("⚠️ Nie znaleziono reguł. Zmniejsz min_confidence.")
        return frequent_itemsets, None
    
    # Sortuj według lift
    rules = rules.sort_values(by='lift', ascending=False)
    print(f"Wygenerowano {len(rules)} reguł asocjacyjnych")
    
    # Filtruj reguły dotyczące wyników
    win_rules = rules[
        rules['consequents'].astype(str).str.contains('home_wins') |
        rules['consequents'].astype(str).str.contains('away_wins')
    ]
    
    print(f"Znaleziono {len(win_rules)} reguł dotyczących wyników meczów")
    
    # Zapisz wyniki
    save_association_results(rules, win_rules, frequent_itemsets, output_dir)
    
    # Wizualizacje
    create_association_visualizations(rules, win_rules, output_dir)
    
    return frequent_itemsets, rules

def save_association_results(rules, win_rules, frequent_itemsets, output_dir):
    """Zapisuje wyniki analizy reguł asocjacyjnych"""
    import os
    
    # Wszystkie reguły
    with open(f'{output_dir}/association_rules_all.txt', 'w', encoding='utf-8') as f:
        f.write("🏀 NBA ASSOCIATION RULES - WSZYSTKIE REGUŁY\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("TOP 20 REGUŁ WEDŁUG LIFT:\n")
        f.write("-" * 40 + "\n")
        
        for i, (_, rule) in enumerate(rules.head(20).iterrows(), 1):
            antecedents = list(rule['antecedents'])
            consequents = list(rule['consequents'])
            
            f.write(f"{i}. JEŚLI: {' & '.join(antecedents)}\n")
            f.write(f"   TO: {' & '.join(consequents)}\n")
            f.write(f"   Support: {rule['support']:.3f} | Confidence: {rule['confidence']:.3f} | Lift: {rule['lift']:.3f}\n\n")
    
    # Reguły dotyczące wyników
    with open(f'{output_dir}/association_rules_wins.txt', 'w', encoding='utf-8') as f:
        f.write("🏀 NBA ASSOCIATION RULES - REGUŁY WYGRANYCH\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("REGUŁY PROWADZĄCE DO WYGRANEJ GOSPODARZY:\n")
        f.write("-" * 50 + "\n")
        
        home_win_rules = win_rules[win_rules['consequents'].astype(str).str.contains('home_wins')]
        
        for i, (_, rule) in enumerate(home_win_rules.head(10).iterrows(), 1):
            antecedents = list(rule['antecedents'])
            
            f.write(f"{i}. JEŚLI: {' & '.join(antecedents)}\n")
            f.write(f"   TO: Gospodarz wygrywa\n")
            f.write(f"   Support: {rule['support']:.3f} | Confidence: {rule['confidence']:.3f} | Lift: {rule['lift']:.3f}\n\n")
        
        f.write("\nREGUŁY PROWADZĄCE DO WYGRANEJ GOŚCI:\n")
        f.write("-" * 50 + "\n")
        
        away_win_rules = win_rules[win_rules['consequents'].astype(str).str.contains('away_wins')]
        
        for i, (_, rule) in enumerate(away_win_rules.head(10).iterrows(), 1):
            antecedents = list(rule['antecedents'])
            
            f.write(f"{i}. JEŚLI: {' & '.join(antecedents)}\n")
            f.write(f"   TO: Goście wygrywają\n")
            f.write(f"   Support: {rule['support']:.3f} | Confidence: {rule['confidence']:.3f} | Lift: {rule['lift']:.3f}\n\n")

def create_association_visualizations(rules, win_rules, output_dir):
    """Tworzy wizualizacje reguł asocjacyjnych"""
    
    # 1. Scatter plot Support vs Confidence
    plt.figure(figsize=(12, 8))
    plt.scatter(rules['support'], rules['confidence'], alpha=0.6, s=rules['lift']*20, c=rules['lift'], cmap='viridis')
    plt.xlabel('Support')
    plt.ylabel('Confidence')
    plt.title('Reguły asocjacyjne NBA - Support vs Confidence\n(rozmiar i kolor = Lift)')
    plt.colorbar(label='Lift')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/association_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Top reguły według confidence
    if len(rules) > 0:
        top_rules = rules.head(15)
        
        plt.figure(figsize=(14, 10))
        
        # Przygotuj etykiety
        labels = []
        for _, rule in top_rules.iterrows():
            antecedents = list(rule['antecedents'])
            consequents = list(rule['consequents'])
            
            # Skróć nazwy dla czytelności
            ant_short = [item.replace('_diff_', '_').replace('_cat', '') for item in antecedents]
            con_short = [item.replace('_diff_', '_').replace('_cat', '') for item in consequents]
            
            label = f"{' & '.join(ant_short[:2])} → {' & '.join(con_short)}"
            if len(label) > 50:
                label = label[:47] + "..."
            labels.append(label)
        
        plt.barh(range(len(labels)), top_rules['confidence'], color='skyblue', alpha=0.8)
        plt.yticks(range(len(labels)), labels)
        plt.xlabel('Confidence')
        plt.title('Top 15 reguł asocjacyjnych według Confidence')
        plt.grid(True, axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/association_top_confidence.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Analiza reguł wygranych
    if len(win_rules) > 0:
        plt.figure(figsize=(12, 8))
        
        home_wins = win_rules[win_rules['consequents'].astype(str).str.contains('home_wins')]
        away_wins = win_rules[win_rules['consequents'].astype(str).str.contains('away_wins')]
        
        plt.scatter(home_wins['support'], home_wins['confidence'], 
                   alpha=0.7, s=100, c='blue', label='Wygrana gospodarzy')
        plt.scatter(away_wins['support'], away_wins['confidence'], 
                   alpha=0.7, s=100, c='red', label='Wygrana gości')
        
        plt.xlabel('Support')
        plt.ylabel('Confidence')
        plt.title('Reguły asocjacyjne - Przewidywanie wyników meczów NBA')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/association_wins_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✅ Zapisano wizualizacje reguł asocjacyjnych w folderze '{output_dir}'")

# Funkcja do uruchomienia z main.py
def run_association_analysis(features_df, output_dir='wyniki'):
    """Główna funkcja do uruchomienia analizy reguł asocjacyjnych"""
    print("\n" + "="*60)
    print("🏀 ETAP 8: ANALIZA REGUŁ ASOCJACYJNYCH")
    print("="*60)
    
    try:
        frequent_itemsets, rules = generate_nba_association_rules(
            features_df, 
            min_support=0.01,  # Możesz dostosować
            min_confidence=0.6,  # Możesz dostosować
            output_dir=output_dir
        )
        
        if rules is not None and len(rules) > 0:
            print(f"✅ Analiza reguł asocjacyjnych zakończona pomyślnie!")
            print(f"📊 Znaleziono {len(rules)} reguł")
            
            # Pokaż przykładowe reguły
            print("\n🔍 PRZYKŁADOWE REGUŁY:")
            win_rules = rules[
                rules['consequents'].astype(str).str.contains('home_wins') |
                rules['consequents'].astype(str).str.contains('away_wins')
            ].head(3)
            
            for i, (_, rule) in enumerate(win_rules.iterrows(), 1):
                antecedents = list(rule['antecedents'])
                consequents = list(rule['consequents'])
                print(f"{i}. JEŚLI: {' & '.join(antecedents)}")
                print(f"   TO: {' & '.join(consequents)}")
                print(f"   Confidence: {rule['confidence']:.3f}, Lift: {rule['lift']:.3f}")
                print()
            
            return True
        else:
            print("❌ Nie udało się wygenerować reguł asocjacyjnych")
            return False
            
    except Exception as e:
        print(f"❌ Błąd podczas analizy reguł asocjacyjnych: {e}")
        return False