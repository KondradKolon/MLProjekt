import pandas as pd
import numpy as np

def prepare_features(game_df, other_stats_df):
    """Przygotowuje cechy do modelu"""
    # Łączenie danych
    game_stats = pd.merge(game_df, other_stats_df, on='game_id', how='left')
    features_df = game_stats.copy()
    
    # Tworzenie różnic statystyk
    stat_pairs = [
        ('fg_pct_home', 'fg_pct_away', 'fg_pct_diff'),
        ('fg3_pct_home', 'fg3_pct_away', 'fg3_pct_diff'),
        ('ft_pct_home', 'ft_pct_away', 'ft_pct_diff'),
        ('reb_home', 'reb_away', 'reb_diff'),
        ('ast_home', 'ast_away', 'ast_diff'),
        ('stl_home', 'stl_away', 'stl_diff'),
        ('blk_home', 'blk_away', 'blk_diff'),
        ('tov_home', 'tov_away', 'tov_diff'),
        ('pf_home', 'pf_away', 'pf_diff')
    ]
    
    for home_stat, away_stat, diff_name in stat_pairs:
        features_df[diff_name] = features_df[home_stat] - features_df[away_stat]
    
    # Zmienna celu
    features_df['home_win'] = (features_df['wl_home'] == 'W').astype(int)
    
    return features_df



def prepare_predictive_features(game_df, team_df=None, n_games=10):
    """Przygotowuje cechy predykcyjne na podstawie danych dostępnych przed meczem"""
    features = []
    
    # Sortuj mecze chronologicznie
    game_df_sorted = game_df.sort_values('game_date')
    
    # Dla każdego meczu
    for idx, game in game_df_sorted.iterrows():
        game_id = game['game_id']
        game_date = pd.to_datetime(game['game_date'])
        home_team_id = game['team_id_home']
        away_team_id = game['team_id_away']
        
        # Pobierz mecze obu drużyn przed bieżącym meczem
        previous_home_games = game_df_sorted[
            (pd.to_datetime(game_df_sorted['game_date']) < game_date) & 
            ((game_df_sorted['team_id_home'] == home_team_id) | 
             (game_df_sorted['team_id_away'] == home_team_id))
        ].tail(n_games)  # Ostatnie n_games meczów
        
        previous_away_games = game_df_sorted[
            (pd.to_datetime(game_df_sorted['game_date']) < game_date) & 
            ((game_df_sorted['team_id_home'] == away_team_id) | 
             (game_df_sorted['team_id_away'] == away_team_id))
        ].tail(n_games)  # Ostatnie n_games meczów
        
        # Oblicz statystyki na podstawie poprzednich meczów
        home_team_stats = calculate_team_stats(previous_home_games, home_team_id)
        away_team_stats = calculate_team_stats(previous_away_games, away_team_id)
        
        # Dodaj informację o przewadze boiska
        home_win_pct = home_team_stats.get('home_win_pct', 0.5)
        away_win_pct = away_team_stats.get('away_win_pct', 0.5)
        
        # Oblicz różnice statystyk
        stat_diffs = {
            'game_id': game_id,
            'game_date': game_date,
            'win_pct_diff': home_team_stats['win_pct'] - away_team_stats['win_pct'],
            'fg_pct_diff': home_team_stats['fg_pct'] - away_team_stats['fg_pct'],
            'fg3_pct_diff': home_team_stats['fg3_pct'] - away_team_stats['fg3_pct'],
            'ft_pct_diff': home_team_stats['ft_pct'] - away_team_stats['ft_pct'],
            'reb_diff': home_team_stats['reb_pg'] - away_team_stats['reb_pg'],
            'ast_diff': home_team_stats['ast_pg'] - away_team_stats['ast_pg'],
            'stl_diff': home_team_stats['stl_pg'] - away_team_stats['stl_pg'],
            'blk_diff': home_team_stats['blk_pg'] - away_team_stats['blk_pg'],
            'tov_diff': home_team_stats['tov_pg'] - away_team_stats['tov_pg'],
            'home_adv': home_win_pct - away_win_pct  # Przewaga gospodarzy
        }
        
        # Dodaj wynik meczu (dla uczenia)
        stat_diffs['home_win'] = 1 if game['wl_home'] == 'W' else 0
        
        features.append(stat_diffs)
    
    features_df = pd.DataFrame(features)
    return features_df

def calculate_team_stats(previous_games, team_id):
    """Oblicza statystyki drużyny na podstawie poprzednich meczów"""
    if len(previous_games) == 0:
        return {
            'win_pct': 0.5,  # Domyślne wartości, gdy brak danych
            'fg_pct': 0.45,
            'fg3_pct': 0.35,
            'ft_pct': 0.75,
            'reb_pg': 45,
            'ast_pg': 22,
            'stl_pg': 7,
            'blk_pg': 5,
            'tov_pg': 14,
            'home_win_pct': 0.6,
            'away_win_pct': 0.4,
        }
    
    # Inicjalizacja zmiennych zliczających
    wins = 0
    home_games = 0
    home_wins = 0
    away_games = 0
    away_wins = 0
    
    # Inicjalizacja wszystkich zmiennych statystycznych
    fg_pct_total = 0
    fg3_pct_total = 0
    ft_pct_total = 0
    reb_total = 0
    ast_total = 0
    stl_total = 0
    blk_total = 0
    tov_total = 0
    
    for _, game in previous_games.iterrows():
        is_home = game['team_id_home'] == team_id
        team_suffix = 'home' if is_home else 'away'
        opponent_suffix = 'away' if is_home else 'home'
        
        # Liczba zwycięstw
        game_result = game[f'wl_{team_suffix}']
        if game_result == 'W':
            wins += 1
            if is_home:
                home_wins += 1
            else:
                away_wins += 1
                
        # Zliczanie meczów u siebie/na wyjeździe
        if is_home:
            home_games += 1
        else:
            away_games += 1
        
        # Statystyki
        for stat, column_prefix in [
            ('fg_pct', 'fg_pct_'), 
            ('fg3_pct', 'fg3_pct_'), 
            ('ft_pct', 'ft_pct_'),
            ('reb', 'reb_'),
            ('ast', 'ast_'),
            ('stl', 'stl_'),
            ('blk', 'blk_'),
            ('tov', 'tov_')
        ]:
            column = f"{column_prefix}{team_suffix}"
            if column in game and not pd.isna(game[column]):
                if stat == 'fg_pct':
                    fg_pct_total += game[column]
                elif stat == 'fg3_pct':
                    fg3_pct_total += game[column]
                elif stat == 'ft_pct':
                    ft_pct_total += game[column]
                elif stat == 'reb':
                    reb_total += game[column]
                elif stat == 'ast':
                    ast_total += game[column]
                elif stat == 'stl':
                    stl_total += game[column]
                elif stat == 'blk':
                    blk_total += game[column]
                elif stat == 'tov':
                    tov_total += game[column]
    
    # Oblicz średnie
    n_games = len(previous_games)
    stats = {
        'win_pct': wins / n_games if n_games > 0 else 0.5,
        'fg_pct': fg_pct_total / n_games if n_games > 0 else 0.45,
        'fg3_pct': fg3_pct_total / n_games if n_games > 0 else 0.35,
        'ft_pct': ft_pct_total / n_games if n_games > 0 else 0.75,
        'reb_pg': reb_total / n_games if n_games > 0 else 45,
        'ast_pg': ast_total / n_games if n_games > 0 else 22,
        'stl_pg': stl_total / n_games if n_games > 0 else 7,
        'blk_pg': blk_total / n_games if n_games > 0 else 5,
        'tov_pg': tov_total / n_games if n_games > 0 else 14,
    }
    
    # Dodanie statystyk dla meczów u siebie/na wyjeździe
    stats['home_win_pct'] = home_wins / home_games if home_games > 0 else 0.6
    stats['away_win_pct'] = away_wins / away_games if away_games > 0 else 0.4
    
    return stats

# Dodaj na końcu pliku:

def prepare_predictive_features_sample(game_df, team_df=None, sample_size=1000, n_games=10, sample_method='recent'):
    """
    Przygotowuje cechy predykcyjne dla podzbioru meczów.
    
    Args:
        game_df: DataFrame z meczami
        team_df: DataFrame z informacjami o drużynach
        sample_size: Liczba meczów do analizy
        n_games: Liczba poprzednich meczów do analizy
        sample_method: Metoda wyboru próbki ('recent', 'random', 'season')
    
    Returns:
        DataFrame z cechami predykcyjnymi
    """
    # Sortuj mecze chronologicznie
    game_df_sorted = game_df.sort_values('game_date')
    
    # Pierwsza linia wydruku - informacja o próbce
    print(f"Przygotowuję próbkę o rozmiarze {sample_size} meczów metodą '{sample_method}'")
    
    # Wybierz podzbiór meczów do analizy
    if sample_method == 'recent':
        # Weź najnowsze mecze
        sample_df = game_df_sorted.tail(sample_size)
    elif sample_method == 'random':
        # Losowa próbka
        sample_df = game_df_sorted.sample(sample_size, random_state=42)
        sample_df = sample_df.sort_values('game_date')  # Sortuj ponownie
    elif sample_method == 'season':
        # Wybierz jeden sezon (np. ostatni dostępny sezon)
        game_df_sorted['season'] = pd.to_datetime(game_df_sorted['game_date']).dt.year
        latest_season = game_df_sorted['season'].max()
        sample_df = game_df_sorted[game_df_sorted['season'] == latest_season]
        # Jeśli sezon ma za dużo meczów, weź tylko sample_size
        if len(sample_df) > sample_size:
            sample_df = sample_df.tail(sample_size)
    else:
        raise ValueError(f"Nieznana metoda próbkowania: {sample_method}")
    
    print(f"Wybrano {len(sample_df)} meczów do analizy metodą '{sample_method}'")
    print(f"Zakres dat: {sample_df['game_date'].min()} - {sample_df['game_date'].max()}")
    
    # Wywołaj standardową funkcję przygotowania cech, ale na mniejszym zbiorze
    return prepare_predictive_features(sample_df, team_df, n_games)