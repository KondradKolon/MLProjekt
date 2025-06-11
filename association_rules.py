import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

def discretize_features(df, columns, n_bins=3):
    """Dyskretyzacja zmiennych ciągłych na kategorie"""
    result = df.copy()
    for col in columns:
        result[col] = pd.qcut(df[col], q=n_bins, labels=False, duplicates='drop')
    return result

def prepare_data_for_rules(game_df, features_df, target_col='home_win'):
    """Przygotowuje dane do analizy reguł asocjacyjnych"""
    # Łączymy dane
    merged = pd.merge(game_df[['game_id', 'game_date']], features_df, on='game_id')
    
    # Numeryczne kolumny do dyskretyzacji
    numeric_cols = features_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    # Dyskretyzacja
    discretized = discretize_features(merged, numeric_cols)
    
    # Dodajemy prefixy dla lepszej interpretacji
    for col in numeric_cols:
        labels = []
        for i in range(discretized[col].nunique()):
            if i == 0:
                labels.append(f"{col}_low")
            elif i == 1:
                labels.append(f"{col}_medium")
            else:
                labels.append(f"{col}_high")
        
        discretized[col] = discretized[col].map(dict(enumerate(labels)))
    
    # Przekształcenie kolumny celu
    if target_col in discretized:
        if discretized[target_col].dtype == 'bool':
            discretized[target_col] = discretized[target_col].map({True: 'win', False: 'loss'})
        elif discretized[target_col].dtype == 'int64':
            discretized[target_col] = discretized[target_col].map({1: 'win', 0: 'loss'})
    
    return discretized

def find_association_rules(df, min_support=0.1, min_confidence=0.5):
    """Znajduje reguły asocjacyjne w danych"""
    # Przekształcenie danych do formatu transakcyjnego
    transactions = []
    
    for _, row in df.iterrows():
        # Dla każdego wiersza, tworzymy listę par (feature, value)
        transaction = []
        for col in df.columns:
            if pd.notna(row[col]) and col != 'game_id' and col != 'game_date':
                transaction.append(f"{col}={row[col]}")
        transactions.append(transaction)
    
    # Kodowanie transakcji
    te = TransactionEncoder()
    te_ary = te.fit_transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    
    # Znajdowanie częstych itemsetów
    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
    
    # Generowanie reguł
    rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=min_confidence)
    
    return rules

def visualize_top_rules(rules, target='home_win=win', n_rules=10, output_path=None):
    """Wizualizuje najważniejsze reguły asocjacyjne"""
    # Filtrowanie reguł związanych z wygraną/przegraną
    target_rules = rules[rules['consequents'].apply(lambda x: target in str(x))]
    
    # Sortowanie według lift
    top_rules = target_rules.sort_values('lift', ascending=False).head(n_rules)
    
    if len(top_rules) == 0:
        print(f"Nie znaleziono reguł dla targetu: {target}")
        return
    
    # Przygotowanie danych do wizualizacji
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    # Tworzenie czytelnych etykiet dla reguł
    labels = []
    for _, rule in top_rules.iterrows():
        antecedents = list(rule['antecedents'])
        antecedents_str = ', '.join([str(item) for item in antecedents])
        if len(antecedents_str) > 50:
            antecedents_str = antecedents_str[:47] + '...'
        labels.append(antecedents_str)
    
    # Wykres 1: Lift
    axes[0].barh(labels, top_rules['lift'], color='skyblue')
    axes[0].set_title(f'Top {n_rules} Rules by Lift (Target: {target})')
    axes[0].set_xlabel('Lift')
    
    # Wykres 2: Confidence
    axes[1].barh(labels, top_rules['confidence'], color='lightgreen')
    axes[1].set_title('Confidence')
    axes[1].set_xlabel('Confidence')
    
    # Wykres 3: Support
    axes[2].barh(labels, top_rules['support'], color='salmon')
    axes[2].set_title('Support')
    axes[2].set_xlabel('Support')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Wykres zapisano jako {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    return top_rules