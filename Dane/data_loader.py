import pandas as pd

def load_data(data_path):
    """Wczytuje wszystkie potrzebne pliki CSV"""
    game_df = pd.read_csv(data_path + "game.csv")
    game_info_df = pd.read_csv(data_path + "game_info.csv")
    team_df = pd.read_csv(data_path + "team.csv")
    line_score_df = pd.read_csv(data_path + "line_score.csv")
    other_stats_df = pd.read_csv(data_path + "other_stats.csv")
    
    return {
        'game': game_df,
        'game_info': game_info_df,
        'team': team_df,
        'line_score': line_score_df,
        'other_stats': other_stats_df
    }

def get_basic_stats(data):
    """Zwraca podstawowe statystyki o danych"""
    game_df = data['game']
    team_df = data['team']
    
    stats = {
        'num_games': len(game_df),
        'date_range': (game_df['game_date'].min(), game_df['game_date'].max()),
        'num_teams': len(team_df),
        'home_advantage': len(game_df[game_df['wl_home'] == 'W']) / len(game_df)
    }
    
    return stats

