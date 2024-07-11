from pathlib import Path

import streamlit as st

s3 = True

local_path = 'data'

s3_bucket = 'rob-mlb-betting'
s3_path = f's3://{s3_bucket}'

AWS_ACCESS_KEY_ID = st.secrets['aws_access_key'] if s3 else None
AWS_SECRET_ACCESS_KEY = st.secrets['aws_secret_access_key'] if s3 else None

the_odds_api = {
    'api_key': st.secrets['odds_api_key'],
    'data_directory': s3_path if s3 else 'data',
    'odds_file': 'odds.parquet',
    'h2h_odds_file': 'odds_h2h.parquet',
    'totals_odds_file': 'odds_totals.parquet'
}

mlb_stats_api = {
    'data_directory': s3_path if s3 else 'data',
    'game_file': 'mlb_games.parquet',
    'start_season': 2022,
    'start_date': '2022-04-07',
    'historic_lineup_file': 'historic_lineups.parquet',
    'probable_lineup_file': 'probable_lineups.parquet',
    'weather_file': 'weather.parquet'
}

baseball_ref = {
    'data_directory': s3_path if s3 else 'data',
    'skip_dates': [],
    'pitching_stats_file': 'pitching_stats.parquet',
    'batting_stats_file': 'batting_stats.parquet'
}

games_and_odds_file = f'{s3_path}/games_and_odds.parquet' if s3 else Path('data/games_and_odds.parquet')
games_odds_lineups_file = f'{s3_path}/games_odds_lineups.parquet' if s3 else Path('data/games_odds_lineups.parquet')

pitching_stats = ['Pit', 'ERA', 'FIP', 'K/BB']
pitching_stat_games = [None, 10, 5, 2]
batting_stats = ['PA', 'BA', 'OBP', 'SLG', 'OPS']
batting_stat_games = [None, 10, 5, 2]

minimum_pitches = 60
minimum_plate_appearances = 3

complete_data_parquet = f'{s3_path}/complete_dataset.parquet' if s3 else Path('data/complete_dataset.parquet')
complete_data_csv = Path('data/complete_dataset.csv')
