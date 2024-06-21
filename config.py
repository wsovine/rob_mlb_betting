from pathlib import Path

the_odds_api = {
    'api_key': '',
    'data_directory': 'data',
    'odds_file': 'odds.parquet',
    'h2h_odds_file': 'odds_h2h.parquet',
    'totals_odds_file': 'odds_totals.parquet'
}

mlb_stats_api = {
    'data_directory': 'data',
    'game_file': 'mlb_games.parquet',
    'start_season': 2022,
    'start_date': '2022-04-07',
    'historic_lineup_file': 'historic_lineups.parquet',
    'probable_lineup_file': 'probable_lineups.parquet',
    'weather_file': 'weather.parquet'
}

baseball_ref = {
    'data_directory': 'data',
    'skip_dates': [],
    'pitching_stats_file': 'pitching_stats.parquet',
    'batting_stats_file': 'batting_stats.parquet'
}

games_and_odds_file = Path('data/games_and_odds.parquet')
games_odds_lineups_file = Path('data/games_odds_lineups.parquet')

pitching_stats = ['Pit', 'ERA', 'FIP', 'K/BB']
pitching_stat_games = [None, 10, 5, 2]
batting_stats = ['PA', 'BA', 'OBP', 'SLG', 'OPS']
batting_stat_games = [None, 10, 5, 2]

minimum_pitches = 60
minimum_plate_appearances = 3

complete_data_parquet = Path('data/complete_dataset.parquet')
complete_data_csv = Path('data/complete_dataset.csv')
