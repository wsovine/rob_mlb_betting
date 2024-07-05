import pandas as pd
import numpy as np
import statsapi
import datetime
import requests
import json
import pybaseball as pb
from s3fs.core import S3FileSystem

from pathlib import Path
from pandas import DataFrame
from tqdm.auto import tqdm
from time import sleep

import config
from config import *
from utils import aggregated_pitching_stats, aggregated_batting_stats
from models import model_ou_probability

pd.set_option('future.no_silent_downcasting', True)

use_s3 = config.s3
filesystem = None
if use_s3:
    # s3 = boto3.client(
    #     "s3",
    #     aws_access_key_id=AWS_ACCESS_KEY_ID,
    #     aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    #     aws_session_token=AWS_SESSION_TOKEN,
    # )
    s3 = S3FileSystem(anon=False, key=AWS_ACCESS_KEY_ID, secret=AWS_SECRET_ACCESS_KEY)
    filesystem = s3


def _directory_check_and_create(path: Path):
    """
    Check if path exists, create it if not
    :param path:
    :return:
    """
    if not use_s3:
        if not path.exists():
            path.mkdir(parents=True)


# 1. Games & Odds
def _mlb_game_data():
    """
    Load MLB game data from the earliest season specified in config file.
    Save it to parquet file and return the resulting dataframe.
    :return: MLB games dataframe
    """
    # Check for game data file
    data_directory = mlb_stats_api['data_directory']
    file = mlb_stats_api['game_file']
    if not use_s3:
        path = Path(data_directory)
        game_data_file = path / file
        new_run = not game_data_file.exists()
    else:
        game_data_file = f'{data_directory}/{file}'
        new_run = not s3.exists(game_data_file)

    # Identify current season
    season_data = statsapi.latest_season()
    current_season = int(season_data['seasonId'])

    # Fetch data from statsapi
    season_dfs = []

    if new_run:
        load_seasons = range(mlb_stats_api['start_season'], current_season + 1)
    else:
        load_seasons = [current_season]
        old_mlb_games_df = pd.read_parquet(game_data_file, filesystem=filesystem)

    for season in (pbar := tqdm(load_seasons)):
        pbar.set_description(f'Loading MLB Stats API Season {season}')
        df = pd.DataFrame(statsapi.schedule(start_date=f'{season}-01-01', end_date=f'{season}-12-31'))
        season_dfs.append(df)

    mlb_games_df = pd.concat(season_dfs)

    # Clean and Prepare the data
    # Filter for only regular season and post-season
    mlb_games_df = mlb_games_df[mlb_games_df.game_type.isin(['R', 'F', 'D', 'L', 'W'])]

    # Handle dates
    mlb_games_df['game_datetime'] = pd.to_datetime(mlb_games_df.game_datetime)
    mlb_games_df['season'] = mlb_games_df['game_datetime'].dt.year

    # There are columns with similar names
    mlb_games_df.losing_team = mlb_games_df.losing_team.fillna(mlb_games_df.losing_Team)
    mlb_games_df.drop('losing_Team', axis=1, inplace=True)

    # Drop duplicates
    mlb_games_df = mlb_games_df.drop_duplicates(subset=['game_id'])

    # There are blank spaces in the dataframe, replace these with NaN
    mlb_games_df = mlb_games_df.replace(r'^\s*$', np.nan, regex=True)

    # Convert score columns to int
    mlb_games_df['away_score'] = mlb_games_df.away_score.astype('int')
    mlb_games_df['home_score'] = mlb_games_df.home_score.astype('int')

    # Save the game data dataframe
    mlb_games_df.set_index('game_id', inplace=True)

    if not new_run:
        old_mlb_games_df.update(mlb_games_df)
        mlb_games_df = mlb_games_df[~mlb_games_df.index.isin(old_mlb_games_df.index)]
        merged_mlb_games_df = pd.concat([old_mlb_games_df, mlb_games_df])
    else:
        merged_mlb_games_df = mlb_games_df.copy()
    merged_mlb_games_df.to_parquet(game_data_file, filesystem=filesystem)
    return merged_mlb_games_df


def _historic_odds_data(game_data: pd.DataFrame) -> pd.DataFrame:
    """
    Fetch and save historic odds data from the-odds-api
    :param game_data: DataFrame
    :return: DataFrame
    """
    # Check if file exists or if new run
    data_directory = the_odds_api['data_directory']
    file = the_odds_api['odds_file']
    if not use_s3:
        path = Path(data_directory)
        odds_data_file = path / file
        new_run = not odds_data_file.exists()
    else:
        odds_data_file = f'{data_directory}/{file}'
        new_run = not s3.exists(odds_data_file)

    df_odds_old = pd.DataFrame()

    # Need to know the last extract date in order to know when
    # to start pulling historic lines
    if new_run:
        last_extract_date = game_data.game_datetime.min()
    else:
        df_odds_old = pd.read_parquet(odds_data_file, filesystem=filesystem)
        # Need to set last_extract_date equal to the latest game time with closing lines
        last_extract_date = max(
            df_odds_old.reset_index().bookmaker_last_update.max(),
            df_odds_old.reset_index().market_last_update.max()
        )

    # define the range of datetimes for which we need to load opening lines
    # these are fetched once per day at a specific time
    # 8am UTC is 3am Central Time
    start = last_extract_date if last_extract_date.hour > 8 else last_extract_date + datetime.timedelta(days=1)
    start = start.replace(hour=8, minute=0, second=0)

    today = datetime.datetime.now(datetime.UTC)
    end = today if today.hour > 8 else today - datetime.timedelta(days=1)
    end = end.replace(hour=8, minute=0, second=0)

    open_datetimes = pd.date_range(start=start, end=end, freq='1d')
    open_datetimes = [d for d in open_datetimes if d.date() in game_data.game_datetime.dt.date.unique()]

    # define the datetimes for which we need to load closing lines
    # these are fetched on the hour mark prior to each game
    game_datetimes = game_data[game_data.game_datetime < today].game_datetime
    game_datetimes_hourly = game_datetimes.dt.floor('h').unique()

    close_datetimes = game_datetimes_hourly[game_datetimes_hourly > last_extract_date].tolist()

    # Fetch historic data from the-odds-api
    # api_key = the_odds_api['api_key']
    api_key = st.secrets['odds_api_key']

    all_datetimes = open_datetimes + close_datetimes
    all_datetimes = sorted(all_datetimes)

    print(f'Loading odds historic points in time: {len(all_datetimes)}')

    all_odds = []
    for dt in (pbar := tqdm(all_datetimes)):
        dt = dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        pbar.set_description(dt)
        response = requests.get(
            f'https://api.the-odds-api.com/v4/sports/baseball_mlb/odds-history/?apiKey={api_key}&regions=us&markets=h2h,totals&oddsFormat=american&date={dt}'
        )
        response_json = json.loads(response.text)

        df = pd.DataFrame(response_json['data'])

        try:
            df = df[df.bookmakers.str.len() > 0]
            # Explode bookmakers
            s = df.explode('bookmakers', ignore_index=True)
            df_exp = s.join(pd.DataFrame([*s.pop('bookmakers')], index=s.index))
            df_exp = df_exp.rename(columns={
                'key': 'bookmaker_key',
                'title': 'bookmaker_title',
                'last_update': 'bookmaker_last_update'
            })

            # Explode markets
            s = df_exp.explode('markets', ignore_index=True)
            df_exp = s.join(pd.DataFrame([*s.pop('markets')], index=s.index))
            df_exp = df_exp.rename(columns={
                'key': 'market_key',
                'last_update': 'market_last_update'
            })

            # Explode outcomes
            s = df_exp.explode('outcomes', ignore_index=True)
            df_exp = s.join(pd.DataFrame([*s.pop('outcomes')], index=s.index))

            all_odds.append(df_exp)
        except AttributeError as e:
            display(df)
            raise e

    # Merge and save new lines
    if len(all_odds) > 0:
        df_odds_new = pd.concat(all_odds)

        df_odds_new['commence_time'] = pd.to_datetime(df_odds_new['commence_time'])
        df_odds_new['bookmaker_last_update'] = pd.to_datetime(df_odds_new['bookmaker_last_update'])
        df_odds_new['market_last_update'] = pd.to_datetime(df_odds_new['market_last_update'])

        df_odds_new = df_odds_new.set_index(['id', 'bookmaker_key', 'market_last_update', 'name'])

        df_odds_new = df_odds_new[~df_odds_new.index.isin(df_odds_old.index)]
        df_odds_merged = pd.concat([df_odds_old, df_odds_new])
    else:
        df_odds_merged = df_odds_old.copy()

    df_odds_merged.to_parquet(odds_data_file, filesystem=filesystem)
    return df_odds_merged


def _current_odds_data() -> pd.DataFrame:
    """
    Fetch and save current odds from the-odds-api
    :return: DataFrame
    """
    # Check if file exists or if new run
    data_directory = the_odds_api['data_directory']
    file = the_odds_api['odds_file']
    if not use_s3:
        path = Path(data_directory)
        odds_data_file = path / file
        new_run = not odds_data_file.exists()
    else:
        odds_data_file = f'{data_directory}/{file}'
        new_run = not s3.exists(odds_data_file)

    df_odds_old = pd.DataFrame()
    if not new_run:
        df_odds_old = pd.read_parquet(odds_data_file, filesystem=filesystem)

    # api_key = the_odds_api['api_key']
    api_key = st.secrets['odds_api_key']

    response = requests.get(
        f'https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/?apiKey={api_key}&regions=us&markets=h2h,totals&oddsFormat=american'
    )
    response_json = json.loads(response.text)

    df = pd.DataFrame(response_json)

    try:
        df = df[df.bookmakers.str.len() > 0]
        # Explode bookmakers
        s = df.explode('bookmakers', ignore_index=True)
        df_exp = s.join(pd.DataFrame([*s.pop('bookmakers')], index=s.index))
        df_exp = df_exp.rename(columns={
            'key': 'bookmaker_key',
            'title': 'bookmaker_title',
            'last_update': 'bookmaker_last_update'
        })

        # Explode markets
        s = df_exp.explode('markets', ignore_index=True)
        df_exp = s.join(pd.DataFrame([*s.pop('markets')], index=s.index))
        df_exp = df_exp.rename(columns={
            'key': 'market_key',
            'last_update': 'market_last_update'
        })

        # Explode outcomes
        s = df_exp.explode('outcomes', ignore_index=True)
        df_exp = s.join(pd.DataFrame([*s.pop('outcomes')], index=s.index))

    except AttributeError as e:
        display(df)
        raise e

    df_exp['commence_time'] = pd.to_datetime(df_exp['commence_time'])
    df_exp['bookmaker_last_update'] = pd.to_datetime(df_exp['bookmaker_last_update'])
    df_exp['market_last_update'] = pd.to_datetime(df_exp['market_last_update'])

    df_exp = df_exp.set_index(['id', 'bookmaker_key', 'market_last_update', 'name'])
    df_exp = df_exp[~df_exp.index.isin(df_odds_old.index)]
    df_odds_merged = pd.concat([df_odds_old, df_exp])

    df_odds_merged.to_parquet(odds_data_file, filesystem=filesystem)
    return df_odds_merged


def _build_odds_datasets() -> tuple:
    """
    Construct datasets for H2h and for Totals based on extracted data
    :return:
    """
    # Load the odds data file
    odds_data_directory = the_odds_api['data_directory']
    file = the_odds_api['odds_file']
    if not use_s3:
        odds_data_path = Path(odds_data_directory)
        odds_data_file = odds_data_path / file
    else:
        odds_data_file = f'{odds_data_directory}/{file}'
    df_odds = pd.read_parquet(odds_data_file, filesystem=filesystem).reset_index()

    # Identify the earliest and latest datetimes available for each H2H line
    updates = df_odds[df_odds['market_last_update'] < df_odds['commence_time']]
    updates = updates[updates['market_key'] == 'h2h']
    updates = updates.groupby(['id', 'bookmaker_key', 'market_key']).agg({'market_last_update': ['min', 'max']})
    updates.columns = updates.columns.droplevel(0)
    updates = updates.reset_index()
    updates = updates.rename(columns={'min': 'opening_datetime', 'max': 'closing_datetime'})

    # H2H Opening Lines
    # Obtain the lines and calculate the consensus
    h2h_open = df_odds.merge(
        updates,
        left_on=['id', 'bookmaker_key', 'market_key', 'market_last_update'],
        right_on=['id', 'bookmaker_key', 'market_key', 'opening_datetime'],
    )

    h2h_open_cons = (
        h2h_open
        .groupby(['id', 'name', 'home_team', 'away_team'])
        .agg(
            h2h_open_price=('price', 'median'),
            commence_time=('commence_time', 'max'),
            h2h_open_datetime=('opening_datetime', 'mean')
        )
    )

    h2h_open_cons['h2h_open_price'] = np.where(
        (h2h_open_cons['h2h_open_price'] >= -100) & (h2h_open_cons['h2h_open_price'] < 100),
        100,
        h2h_open_cons['h2h_open_price']
    )
    h2h_open_cons['h2h_open_price'] = h2h_open_cons['h2h_open_price'].apply(np.floor)
    h2h_open_cons['h2h_open_price'] = h2h_open_cons['h2h_open_price'].astype(int)

    h2h_open_cons['h2h_open_implied_probability'] = np.where(
        h2h_open_cons['h2h_open_price'] > 0,
        100 / (h2h_open_cons['h2h_open_price'] + 100),
        h2h_open_cons['h2h_open_price'] / (h2h_open_cons['h2h_open_price'] - 100)
    )
    h2h_open_cons['h2h_open_implied_probability'] = h2h_open_cons['h2h_open_implied_probability'].round(4)

    # H2H Closing Lines
    # Obtain the lines and calculate the consensus
    h2h_close = df_odds.merge(
        updates,
        left_on=['id', 'bookmaker_key', 'market_key', 'market_last_update'],
        right_on=['id', 'bookmaker_key', 'market_key', 'closing_datetime'],
    )

    h2h_close_cons = (
        h2h_close
        .groupby(['id', 'name', 'home_team', 'away_team'])
        .agg(
            h2h_close_price=('price', 'median'),
            h2h_close_datetime=('closing_datetime', 'mean')
        )
    )

    h2h_close_cons['h2h_close_price'] = np.where(
        (h2h_close_cons['h2h_close_price'] >= -100) & (h2h_close_cons['h2h_close_price'] < 100),
        100,
        h2h_close_cons['h2h_close_price']
    )
    h2h_close_cons['h2h_close_price'] = h2h_close_cons['h2h_close_price'].apply(np.floor)
    h2h_close_cons['h2h_close_price'] = h2h_close_cons['h2h_close_price'].astype(int)

    h2h_close_cons['h2h_close_implied_probability'] = np.where(
        h2h_close_cons['h2h_close_price'] > 0,
        100 / (h2h_close_cons['h2h_close_price'] + 100),
        h2h_close_cons['h2h_close_price'] / (h2h_close_cons['h2h_close_price'] - 100)
    )
    h2h_close_cons['h2h_close_implied_probability'] = h2h_close_cons['h2h_close_implied_probability'].round(4)

    # Combine H2H opening and closing, save file
    df_h2h = h2h_open_cons.join(h2h_close_cons)

    directory = the_odds_api['data_directory']
    file = the_odds_api['h2h_odds_file']
    if not use_s3:
        path = Path(directory)
        file_path = path / file
    else:
        file_path = f'{directory}/{file}'
    df_h2h.to_parquet(file_path, filesystem=filesystem)

    # Identify the earliest and latest datetimes available for each Totals line
    updates = df_odds[df_odds['market_last_update'] < df_odds['commence_time']]
    updates = updates[updates['market_key'] == 'totals']
    updates = updates.groupby(['id', 'bookmaker_key', 'market_key']).agg({'market_last_update': ['min', 'max']})
    updates.columns = updates.columns.droplevel(0)
    updates = updates.reset_index()
    updates = updates.rename(columns={'min': 'opening_datetime', 'max': 'closing_datetime'})

    # Totals Opening Lines
    # Obtain the lines and calculate the consensus
    totals_open = df_odds.merge(
        updates,
        left_on=['id', 'bookmaker_key', 'market_key', 'market_last_update'],
        right_on=['id', 'bookmaker_key', 'market_key', 'opening_datetime'],
    )

    totals_open_cons = (
        totals_open
        .groupby(['id', 'name', 'home_team', 'away_team'])
        .agg(
            totals_open_price=('price', 'median'),
            totals_open_point=('point', 'median'),
            commence_time=('commence_time', 'max'),
            totals_open_datetime=('opening_datetime', 'mean')
        )
    )

    totals_open_cons['totals_open_price'] = np.where(
        (totals_open_cons['totals_open_price'] >= -100) & (totals_open_cons['totals_open_price'] < 100),
        100,
        totals_open_cons['totals_open_price']
    )
    totals_open_cons['totals_open_price'] = totals_open_cons['totals_open_price'].apply(np.floor)
    totals_open_cons['totals_open_price'] = totals_open_cons['totals_open_price'].astype(int)

    totals_open_cons['totals_open_implied_probability'] = np.where(
        totals_open_cons['totals_open_price'] > 0,
        100 / (totals_open_cons['totals_open_price'] + 100),
        totals_open_cons['totals_open_price'] / (totals_open_cons['totals_open_price'] - 100)
    )
    totals_open_cons['totals_open_implied_probability'] = totals_open_cons['totals_open_implied_probability'].round(4)

    # Totals Closing Lines
    # Obtain the lines and calculate the consensus
    totals_close = df_odds.merge(
        updates,
        left_on=['id', 'bookmaker_key', 'market_key', 'market_last_update'],
        right_on=['id', 'bookmaker_key', 'market_key', 'closing_datetime'],
    )

    totals_close_cons = (
        totals_close
        .groupby(['id', 'name', 'home_team', 'away_team'])
        .agg(
            totals_close_price=('price', 'median'),
            totals_close_point=('point', 'median'),
            totals_close_datetime=('closing_datetime', 'mean')
        )
    )

    totals_close_cons['totals_close_price'] = np.where(
        (totals_close_cons['totals_close_price'] >= -100) & (totals_close_cons['totals_close_price'] < 100),
        100,
        totals_close_cons['totals_close_price']
    )
    totals_close_cons['totals_close_price'] = totals_close_cons['totals_close_price'].apply(np.floor)
    totals_close_cons['totals_close_price'] = totals_close_cons['totals_close_price'].astype(int)

    totals_close_cons['totals_close_implied_probability'] = np.where(
        totals_close_cons['totals_close_price'] > 0,
        100 / (totals_close_cons['totals_close_price'] + 100),
        totals_close_cons['totals_close_price'] / (totals_close_cons['totals_close_price'] - 100)
    )
    totals_close_cons['totals_close_implied_probability'] = totals_close_cons['totals_close_implied_probability'].round(4)

    # Combine Totals opening and closing, save file
    df_totals = totals_open_cons.join(totals_close_cons)

    directory = the_odds_api['data_directory']
    file = the_odds_api['totals_odds_file']
    if not use_s3:
        path = Path(directory)
        file_path = path / file
    else:
        file_path = f'{directory}/{file}'
    df_totals.to_parquet(file_path, filesystem=filesystem)

    return df_h2h, df_totals


def _create_combined_games_and_odds_dataset(df_games: pd.DataFrame, df_h2h: pd.DataFrame, df_totals: pd.DataFrame):
    """
    Combine all cleaned and prepared datasources into a single data file
    :param df_games:
    :param df_h2h:
    :param df_totals:
    :return:
    """
    df_games.reset_index(inplace=True)
    df_h2h.reset_index(inplace=True)
    df_totals.reset_index(inplace=True)

    # Double Headers
    df_h2h['game_date'] = df_h2h['commence_time'].dt.date
    df_h2h['game_date'] = df_h2h['game_date'].astype(str)
    df_h2h['commence_time_cst'] = df_h2h['commence_time'].dt.tz_convert('America/Chicago')
    df_h2h['game_date_cst'] = df_h2h['commence_time_cst'].dt.date
    df_h2h['game_date_cst'] = df_h2h['game_date_cst'].astype(str)
    df_h2h['game_num'] = df_h2h.groupby(['game_date_cst', 'home_team', 'away_team'])['commence_time'].rank('dense')
    df_h2h['game_num'] = df_h2h['game_num'].astype(int)

    df_totals['game_date'] = df_totals['commence_time'].dt.date
    df_totals['game_date'] = df_totals['game_date'].astype(str)
    df_totals['commence_time_cst'] = df_totals['commence_time'].dt.tz_convert('America/Chicago')
    df_totals['game_date_cst'] = df_totals['commence_time_cst'].dt.date
    df_totals['game_date_cst'] = df_totals['game_date_cst'].astype(str)
    df_totals['game_num'] = df_totals.groupby(['game_date_cst', 'home_team', 'away_team'])['commence_time'].rank('dense')
    df_totals['game_num'] = df_totals['game_num'].astype(int)

    df_games['game_date'] = df_games['game_date'].astype(str)

    # First add the home and away h2h odds to the game dataframe
    h2h_cols = [
        'home_team', 'away_team', 'game_date_cst', 'game_num', 'h2h_open_price', 'h2h_open_datetime',
        'h2h_open_implied_probability', 'h2h_close_price', 'h2h_close_datetime',
        'h2h_close_implied_probability'
    ]
    # Home team
    df = (
        df_games
        .merge(
            df_h2h[df_h2h['home_team'] == df_h2h['name']][h2h_cols],
            left_on=['game_date', 'game_num', 'home_name', 'away_name'],
            right_on=['game_date_cst', 'game_num', 'home_team', 'away_team']
        )
        .drop([c for c in h2h_cols if not c.startswith('h2h') and c not in df_games.columns], axis='columns')
        .rename(columns={c: f'{c}_home' for c in h2h_cols if c not in df_games.columns})
    )
    # Away team
    df = (
        df
        .merge(
            df_h2h[df_h2h['away_team'] == df_h2h['name']][h2h_cols],
            left_on=['game_date', 'game_num', 'home_name', 'away_name'],
            right_on=['game_date_cst', 'game_num', 'home_team', 'away_team']
        )
        .drop([c for c in h2h_cols if not c.startswith('h2h') and c not in df_games.columns], axis='columns')
        .rename(columns={c: f'{c}_away' for c in h2h_cols if c not in df_games.columns})
    )

    # Next add the over and under
    total_cols = [
        'home_team', 'away_team', 'game_date_cst', 'game_num', 'totals_open_point', 'totals_open_price',
        'totals_open_datetime', 'totals_open_implied_probability',
        'totals_close_point', 'totals_close_price', 'totals_close_datetime',
        'totals_close_implied_probability'
    ]

    # Over
    df = (
        df
        .merge(
            df_totals[df_totals['name'] == 'Over'][total_cols],
            left_on=['game_date', 'game_num', 'home_name', 'away_name'],
            right_on=['game_date_cst', 'game_num', 'home_team', 'away_team']
        )
        .drop([
            c for c in total_cols
            if not c.startswith('totals')
               and c not in df_games.columns
        ], axis='columns')
        .rename(columns={
            c: f'{c}_over'
            for c in total_cols
            if c not in df_games.columns
        })
    )

    # Under
    df = (
        df
        .merge(
            df_totals[df_totals['name'] == 'Under'][total_cols],
            left_on=['game_date', 'game_num', 'home_name', 'away_name'],
            right_on=['game_date_cst', 'game_num', 'home_team', 'away_team']
        )
        .drop([c for c in total_cols if not c.startswith('totals') and c not in df_games.columns], axis='columns')
        .drop(['totals_open_point', 'totals_close_point'], axis='columns')
        .rename(columns={c: f'{c}_under' for c in total_cols if c not in df_games.columns})
    )

    df = df.rename(columns={
        'totals_open_point_over': 'totals_open_point',
        'totals_close_point_over': 'totals_close_point'
    })

    df = df.sort_values(
        ['totals_close_datetime_over', 'totals_close_datetime_under',
         'h2h_close_datetime_home', 'h2h_close_datetime_away'
         ]).drop_duplicates('game_id', keep='last')

    df.to_parquet(games_and_odds_file, filesystem=filesystem)


# 2. Lineups
def _historic_lineup_data(game_data: pd.DataFrame) -> DataFrame:
    print('Loading historic lineups')
    df_games = game_data[~game_data.status.isin(['Postponed', 'Cancelled', 'Scheduled', 'Pre-Game'])]
    df_games = df_games.set_index('game_id')

    # Check if file exists or if new run
    data_directory = mlb_stats_api['data_directory']
    file = mlb_stats_api['historic_lineup_file']
    if not use_s3:
        path = Path(data_directory)
        historic_lineup_file = path / file
        new_run = not historic_lineup_file.exists()
    else:
        historic_lineup_file = f'{data_directory}/{file}'
        new_run = not s3.exists(historic_lineup_file)

    df_historic_lineups = pd.DataFrame()

    if new_run:
        games_to_load = df_games.index.tolist()
    else:
        df_historic_lineups = pd.read_parquet(historic_lineup_file, filesystem=filesystem)
        games_to_load = df_games[~df_games.index.isin(df_historic_lineups.index)].index.tolist()

    dfs = []

    for game_id in (pbar := tqdm(games_to_load)):
        pbar.set_description(str(game_id))

        # Game
        game = statsapi.get(
            "schedule",
            {"sportId": 1, "gamePk": game_id, "hydrate": "probablePitcher(note)"}
        )
        game_date = game['dates'][0]['date']

        # Away
        away_team = game['dates'][0]['games'][0]['teams']['away']['team']
        # Away Pitcher
        try:
            away_probable_pitcher = game['dates'][0]['games'][0]['teams']['away']['probablePitcher']
        except KeyError:
            continue
        # Away Batters
        away_batters = statsapi.get(
            "game",
            {"gamePk": game_id}
        )['liveData']['boxscore']['teams']['away']['batters']
        # print(len(away_batters))
        if len(away_batters) == 0:
            continue

        # Home
        home_team = game['dates'][0]['games'][0]['teams']['home']['team']
        # Home Pitcher
        try:
            home_probable_pitcher = game['dates'][0]['games'][0]['teams']['home']['probablePitcher']
        except KeyError:
            continue
        # Home Batters
        home_batters = statsapi.get(
            "game",
            {"gamePk": game_id}
        )['liveData']['boxscore']['teams']['home']['batters']
        # print(len(home_batters))
        if len(home_batters) == 0:
            continue

        df = pd.DataFrame(
            {
                'game_id': game_id,
                'game_date': game_date,
                'away_team_id': away_team['id'],
                'away_team_name': away_team['name'],
                'away_probable_pitcher_id': away_probable_pitcher['id'],
                'away_probable_pitcher_name': away_probable_pitcher['fullName'],
                'away_batter_1': away_batters[0],
                'away_batter_2': away_batters[1],
                'away_batter_3': away_batters[2],
                'away_batter_4': away_batters[3],
                'away_batter_5': away_batters[4],
                'away_batter_6': away_batters[5],
                'away_batter_7': away_batters[6],
                'away_batter_8': away_batters[7],
                'away_batter_9': away_batters[8],
                'home_team_id': home_team['id'],
                'home_team_name': home_team['name'],
                'home_probable_pitcher_id': home_probable_pitcher['id'],
                'home_probable_pitcher_name': home_probable_pitcher['fullName'],
                'home_batter_1': home_batters[0],
                'home_batter_2': home_batters[1],
                'home_batter_3': home_batters[2],
                'home_batter_4': home_batters[3],
                'home_batter_5': home_batters[4],
                'home_batter_6': home_batters[5],
                'home_batter_7': home_batters[6],
                'home_batter_8': home_batters[7],
                'home_batter_9': home_batters[8],
            },
            index=[game_id])

        dfs.append(df)

    load = len(dfs) > 0
    if load:
        df = pd.concat(dfs).set_index('game_id')
        if not new_run:
            df_historic_lineups.update(df)
            df = df[~df.index.isin(df_historic_lineups.index)]
            df = pd.concat([df_historic_lineups, df])

        df.to_parquet(historic_lineup_file, filesystem=filesystem)
        return df
    else:
        print('No historic starting lineups to load.')
        return df_historic_lineups


def _probable_lineup_data(game_data: pd.DataFrame) -> DataFrame:
    print('Loading probable lineups')
    today = str(datetime.date.today())
    tomorrow = str(datetime.date.today() + datetime.timedelta(days=1))

    df_games = game_data[~game_data.status.isin(['Postponed', 'Cancelled'])]
    df_games = df_games[(df_games.game_date >= today) & (df_games.game_date <= tomorrow)]
    df_games = df_games.set_index('game_id')

    # Check if file exists or if new run
    data_directory = mlb_stats_api['data_directory']
    file = mlb_stats_api['probable_lineup_file']
    if not use_s3:
        path = Path(data_directory)
        probable_lineup_file = path / file
        new_run = not probable_lineup_file.exists()
    else:
        probable_lineup_file = f'{data_directory}/{file}'
        new_run = not s3.exists(probable_lineup_file)

    df_probable_lineups = pd.DataFrame()

    if not new_run:
        df_probable_lineups = pd.read_parquet(probable_lineup_file, filesystem=filesystem)
        df_games = df_games[~df_games.index.isin(df_probable_lineups.index)]

    games_to_load = df_games.index.tolist()

    dfs = []

    for game_id in (pbar := tqdm(games_to_load)):
        pbar.set_description(str(game_id))

        # Game
        game = statsapi.get(
            "schedule",
            {"sportId": 1, "gamePk": game_id, "hydrate": "probablePitcher(note)"}
        )
        game_date = game['dates'][0]['date']

        # Away
        away_team = game['dates'][0]['games'][0]['teams']['away']['team']
        # Away Pitcher
        try:
            away_probable_pitcher = game['dates'][0]['games'][0]['teams']['away']['probablePitcher']
        except KeyError:
            continue
        # Away Batters
        away_batters = statsapi.get(
            "game",
            {"gamePk": game_id}
        )['liveData']['boxscore']['teams']['away']['batters']
        # print(len(away_batters))
        if len(away_batters) == 0:
            continue

        # Home
        home_team = game['dates'][0]['games'][0]['teams']['home']['team']
        # Home Pitcher
        try:
            home_probable_pitcher = game['dates'][0]['games'][0]['teams']['home']['probablePitcher']
        except KeyError:
            continue
        # Home Batters
        home_batters = statsapi.get(
            "game",
            {"gamePk": game_id}
        )['liveData']['boxscore']['teams']['home']['batters']
        # print(len(home_batters))
        if len(home_batters) == 0:
            continue

        df = pd.DataFrame(
            {
                'game_id': game_id,
                'game_date': game_date,
                'away_team_id': away_team['id'],
                'away_team_name': away_team['name'],
                'away_probable_pitcher_id': away_probable_pitcher['id'],
                'away_probable_pitcher_name': away_probable_pitcher['fullName'],
                'away_batter_1': away_batters[0],
                'away_batter_2': away_batters[1],
                'away_batter_3': away_batters[2],
                'away_batter_4': away_batters[3],
                'away_batter_5': away_batters[4],
                'away_batter_6': away_batters[5],
                'away_batter_7': away_batters[6],
                'away_batter_8': away_batters[7],
                'away_batter_9': away_batters[8],
                'home_team_id': home_team['id'],
                'home_team_name': home_team['name'],
                'home_probable_pitcher_id': home_probable_pitcher['id'],
                'home_probable_pitcher_name': home_probable_pitcher['fullName'],
                'home_batter_1': home_batters[0],
                'home_batter_2': home_batters[1],
                'home_batter_3': home_batters[2],
                'home_batter_4': home_batters[3],
                'home_batter_5': home_batters[4],
                'home_batter_6': home_batters[5],
                'home_batter_7': home_batters[6],
                'home_batter_8': home_batters[7],
                'home_batter_9': home_batters[8],
            },
            index=[game_id])

        dfs.append(df)

    load = len(dfs) > 0
    if load:
        df = pd.concat(dfs).set_index('game_id')
        if not new_run:
            df_probable_lineups.update(df)
            df = df[~df.index.isin(df_probable_lineups.index)]
            df = pd.concat([df_probable_lineups, df])
        df.to_parquet(probable_lineup_file, filesystem=filesystem)
        return df
    else:
        print('No probable starting lineups to load.')
        return df_probable_lineups


def _combine_lineup_data(df_historic: pd.DataFrame, df_probable: pd.DataFrame) -> pd.DataFrame:
    df_games_odds = pd.read_parquet(games_and_odds_file, filesystem=filesystem)

    df_combined = df_games_odds.set_index('game_id')
    games_odds_overlap_cols = ['home_probable_pitcher', 'away_probable_pitcher']
    df_combined = df_combined[[c for c in df_combined.columns if c not in games_odds_overlap_cols]]

    lineup_overlap_cols = ['game_date', 'home_team_id', 'away_team_id', 'home_team_name', 'away_team_name']
    df_historic_lineups = df_historic[[c for c in df_historic.columns if c not in lineup_overlap_cols]]
    df_combined = df_combined.join(df_historic_lineups)

    df_probable_lineups = df_probable[[c for c in df_probable.columns if c not in lineup_overlap_cols]]
    df_combined = df_combined.join(df_probable_lineups, rsuffix='_probable')

    for team in ['home', 'away']:
        df_combined[f'{team}_probable_pitcher_id'] = df_combined[f'{team}_probable_pitcher_id'].fillna(
            df_combined[f'{team}_probable_pitcher_id_probable']
        )
        df_combined[f'{team}_probable_pitcher_name'] = df_combined[f'{team}_probable_pitcher_name'].fillna(
            df_combined[f'{team}_probable_pitcher_name_probable']
        )
        df_combined = df_combined.drop([f'{team}_probable_pitcher_id_probable', f'{team}_probable_pitcher_name_probable'], axis='columns')

        for batter in range(9):
            batter = batter + 1
            df_combined[f'{team}_batter_{batter}'] = df_combined[f'{team}_batter_{batter}'].fillna(
                df_combined[f'{team}_batter_{batter}_probable']
            )
            df_combined = df_combined.drop(f'{team}_batter_{batter}_probable', axis='columns')

    df_combined.to_parquet(games_odds_lineups_file, filesystem=filesystem)
    return df_combined


def _pitching_stats_by_day(df_games: pd.DataFrame):
    print('Loading daily pitching stats.')
    # Check if file exists or if new run
    data_directory = baseball_ref['data_directory']
    file = baseball_ref['pitching_stats_file']
    if not use_s3:
        path = Path(data_directory)
        pitching_stats_file = path / file
        new_run = not pitching_stats_file.exists()
    else:
        pitching_stats_file = f'{data_directory}/{file}'
        new_run = not s3.exists(pitching_stats_file)

    df_pitching_stats = pd.DataFrame()

    today = str(datetime.date.today())
    df_games = df_games[df_games.status == 'Final']
    df_games = df_games[df_games.game_type == 'R']
    df_games = df_games[df_games.game_date < today]

    if new_run:
        dates_to_load = df_games.game_date.unique().tolist()
    else:
        df_pitching_stats = pd.read_parquet(pitching_stats_file, filesystem=filesystem)
        dates_to_load = df_games[~df_games.game_date.isin(df_pitching_stats.reset_index().game_date.unique())].game_date.unique().tolist()

    for date in (pbar := tqdm(dates_to_load)):
        pbar.set_description(date)
        try:
            df = pb.pitching_stats_range(date)
        except IndexError as e:
            print('Index Error with PyBaseball')
            print(str(e))
            continue
        df['game_date'] = date
        df = df.set_index(['mlbID', 'game_date'])

        if new_run:
            new_run = False
        else:
            df_pitching_stats.update(df)
            df = df[~df.index.isin(df_pitching_stats.index)]
            df = pd.concat([df_pitching_stats, df])

        df.to_parquet(pitching_stats_file, filesystem=filesystem)
        df_pitching_stats = df.copy()

        sleep(7)


def _batting_stats_by_day(df_games):
    print('Loading daily batting stats.')
    data_directory = baseball_ref['data_directory']
    file = baseball_ref['batting_stats_file']
    if not use_s3:
        path = Path(data_directory)
        batting_stats_file = path / file
        new_run = not batting_stats_file.exists()
    else:
        batting_stats_file = f'{data_directory}/{file}'
        new_run = not s3.exists(batting_stats_file)

    df_batting_stats = pd.DataFrame()

    today = str(datetime.date.today())
    df_games = df_games[df_games.status == 'Final']
    df_games = df_games[df_games.game_type == 'R']
    df_games = df_games[df_games.game_date < today]

    if new_run:
        dates_to_load = df_games.game_date.unique().tolist()
    else:
        df_batting_stats = pd.read_parquet(batting_stats_file, filesystem=filesystem)
        dates_to_load = df_games[~df_games.game_date.isin(df_batting_stats.reset_index().game_date.unique())].game_date.unique().tolist()

    for date in (pbar := tqdm(dates_to_load)):
        pbar.set_description(date)
        try:
            df = pb.batting_stats_range(date)
        except IndexError as e:
            print('Index Error with PyBaseball')
            print(str(e))
            continue
        df['game_date'] = date
        df = df.set_index(['mlbID', 'game_date'])

        if new_run:
            new_run = False
        else:
            df_batting_stats.update(df)
            df = df[~df.index.isin(df_batting_stats.index)]
            df = pd.concat([df_batting_stats, df])

        df.to_parquet(batting_stats_file, filesystem=filesystem)
        df_batting_stats = df.copy()

        sleep(7)


def _agg_and_append_pitching_stats(df_games):
    df_games['away_probable_pitcher_id'] = df_games['away_probable_pitcher_id'].astype('Int64').astype(str)
    df_games['home_probable_pitcher_id'] = df_games['home_probable_pitcher_id'].astype('Int64').astype(str)

    for games in pitching_stat_games:
        print(f'Calculating {games if games else "season"} {"game" if games else ""} pitching stats.')
        dfp = aggregated_pitching_stats(df_games, last_games=games)
        dfp = dfp[['mlbID', 'game_dates_prior_to'] + pitching_stats]

        # Away pitcher
        dfp_away = dfp.rename(columns={s: f'away_{s}_{games if games else "season"}' for s in pitching_stats})
        df_games = df_games.merge(
            dfp_away,
            how='left',
            left_on=['away_probable_pitcher_id', 'game_date'],
            right_on=['mlbID', 'game_dates_prior_to']
        )
        df_games = df_games.drop(['mlbID', 'game_dates_prior_to'], axis='columns')

        # Home pitcher
        dfp_home = dfp.rename(columns={s: f'home_{s}_{games if games else "season"}' for s in pitching_stats})
        df_games = df_games.merge(
            dfp_home,
            how='left',
            left_on=['home_probable_pitcher_id', 'game_date'],
            right_on=['mlbID', 'game_dates_prior_to']
        )
        df_games = df_games.drop(['mlbID', 'game_dates_prior_to'], axis='columns')

    return df_games


def _agg_and_append_batting_stats(df_games):
    home_batter_cols = [f'home_batter_{i}' for i in range(1, 10)]
    away_batter_cols = [f'away_batter_{i}' for i in range(1, 10)]
    batter_cols = away_batter_cols + home_batter_cols

    df_games[batter_cols] = df_games[batter_cols].astype('Int64')
    
    for games in batting_stat_games:
        print(f'Calculating {games if games else "season"} {"game" if games else ""} batting stats.')
        dfb = aggregated_batting_stats(df_games, last_games=games)
        dfb = dfb[['mlbID', 'game_dates_prior_to'] + batting_stats]
    
        for batter in batter_cols:
            df_batter = dfb.rename(columns={s: f'{batter}_{s}_{games if games else "season"}' for s in batting_stats})
            df_games = df_games.merge(
                df_batter,
                how='left',
                left_on=[batter, 'game_date'],
                right_on=['mlbID', 'game_dates_prior_to']
            )
            df_games = df_games.drop(['mlbID', 'game_dates_prior_to'], axis='columns')

    return df_games


def _agg_and_append_bullpen_stats(df_games):
    for games in pitching_stat_games:
        print(f'Calculating {games if games else "season"} {"game" if games else ""} bullpen stats.')
        dfp = aggregated_pitching_stats(df_games, last_games=games, bullpen=True)

        # Team names don't match up so we need to fetch that in order to join to games
        team_dfs = []
        # print('Mapping team info')
        # for season in (pbar := tqdm(dfp.season.unique())):
        for season in dfp.season.unique():
            # pbar.set_description(str(season))
            team_df = pd.DataFrame(
                statsapi.get('teams', params={'sportId': 1, 'season': season})['teams']
            )
            team_df = team_df[[
                'id', 'name', 'season', 'abbreviation', 'teamName', 'locationName',
                'shortName', 'franchiseName', 'clubName', 'league'
            ]]
            team_df['league'] = [d.get('name') for d in team_df['league']]
            team_dfs.append(team_df)

        df_team = pd.concat(team_dfs)

        dfp.loc[dfp['Lev'] == 'Maj-AL', 'league'] = 'American League'
        dfp.loc[dfp['Lev'] == 'Maj-NL', 'league'] = 'National League'

        dfp = dfp.merge(df_team, left_on=['Tm', 'league', 'season'], right_on=['franchiseName', 'league', 'season'])

        dfp = dfp[['id', 'game_dates_prior_to'] + pitching_stats]

        # Away bullpen
        dfp_away = dfp.rename(columns={s: f'away_bullpen_{s}_{games if games else "season"}' for s in pitching_stats})
        df_games = df_games.merge(
            dfp_away,
            how='left',
            left_on=['away_id', 'game_date'],
            right_on=['id', 'game_dates_prior_to']
        )
        df_games = df_games.drop(['id', 'game_dates_prior_to'], axis='columns')

        # Home bullpen
        dfp_home = dfp.rename(columns={s: f'home_bullpen_{s}_{games if games else "season"}' for s in pitching_stats})
        df_games = df_games.merge(
            dfp_home,
            how='left',
            left_on=['home_id', 'game_date'],
            right_on=['id', 'game_dates_prior_to']
        )
        df_games = df_games.drop(['id', 'game_dates_prior_to'], axis='columns')

    return df_games


def _calculate_home_and_away_win_pct(df):
    df_final = df[df.status == 'Final'].sort_values('game_datetime')

    df_final['home_win'] = np.where(df_final['home_name'] == df_final['winning_team'], 1, 0)
    df_final['home_win'] = df_final.groupby(['season', 'home_name']).home_win.shift(1)
    df_final['home_win'] = df_final['home_win'].fillna(0)
    df_final['home_wins'] = df_final.groupby(['season', 'home_name'])['home_win'].cumsum()
    df_final['home_games'] = df_final.groupby(['season', 'home_name']).game_id.cumcount()
    df_final['home_win_pct'] = df_final['home_wins'] / df_final['home_games']

    df_final['away_win'] = np.where(df_final['away_name'] == df_final['winning_team'], 1, 0)
    df_final['away_win'] = df_final.groupby(['season', 'away_name']).away_win.shift(1)
    df_final['away_win'] = df_final['away_win'].fillna(0)
    df_final['away_wins'] = df_final.groupby(['season', 'away_name'])['away_win'].cumsum()
    df_final['away_games'] = df_final.groupby(['season', 'away_name']).game_id.cumcount()
    df_final['away_win_pct'] = df_final['away_wins'] / df_final['away_games']

    df_final['home_run_diff'] = df_final['home_score'] - df_final['away_score']
    df_final['away_run_diff'] = df_final['away_score'] - df_final['home_score']

    df_runs = pd.concat([
        df_final[['game_id', 'season', 'home_name', 'game_datetime', 'home_score', 'away_score']].rename(columns={
            'home_name': 'team',
            'home_score': 'runs_for',
            'away_score': 'runs_against'
        }),
        df_final[['game_id', 'season', 'away_name', 'game_datetime', 'away_score', 'home_score']].rename(columns={
            'away_name': 'team',
            'away_score': 'runs_for',
            'home_score': 'runs_against'
        })
    ]).sort_values('game_datetime')

    df_runs['games'] = df_runs.groupby(['season', 'team'])['game_id'].cumcount()

    df_runs['prior_runs_for'] = df_runs.groupby(['season', 'team']).runs_for.shift(1)
    df_runs['prior_runs_for'] = df_runs['prior_runs_for'].fillna(0)
    df_runs['cumulative_runs_for'] = df_runs.groupby(['season', 'team'])['prior_runs_for'].cumsum()

    df_runs['prior_runs_against'] = df_runs.groupby(['season', 'team']).runs_against.shift(1)
    df_runs['prior_runs_against'] = df_runs['prior_runs_against'].fillna(0)
    df_runs['cumulative_runs_against'] = df_runs.groupby(['season', 'team'])['prior_runs_against'].cumsum()

    df_runs['pythagorean'] = df_runs['cumulative_runs_for']**1.83 / (df_runs['cumulative_runs_for']**1.83 + df_runs['cumulative_runs_against']**1.83)

    df_final = (
        df_final
        .merge(
            df_runs[['game_id', 'team', 'games', 'cumulative_runs_for', 'cumulative_runs_against', 'pythagorean']],
            left_on=['game_id', 'away_name'],
            right_on=['game_id', 'team']
        )
        .rename(columns={
            'games': 'home_total_games',
            'cumulative_runs_for': 'home_szn_runs_scored',
            'cumulative_runs_against': 'home_szn_runs_allowed',
            'pythagorean': 'home_pythag_win_pct'
        })
        .merge(
            df_runs[['game_id', 'team', 'games', 'cumulative_runs_for', 'cumulative_runs_against', 'pythagorean']],
            left_on=['game_id', 'home_name'],
            right_on=['game_id', 'team']
        )
        .rename(columns={
            'games': 'away_total_games',
            'cumulative_runs_for': 'away_szn_runs_scored',
            'cumulative_runs_against': 'away_szn_runs_allowed',
            'pythagorean': 'away_pythag_win_pct'
        })
    )

    df_full = (
        df
        .merge(
            df_final[[
                'game_id',
                'home_wins', 'home_games', 'home_win_pct', 'away_wins', 'away_games', 'away_win_pct',
                'home_run_diff', 'away_run_diff',
                'home_szn_runs_scored', 'home_szn_runs_allowed', 'home_pythag_win_pct',
                'away_szn_runs_scored', 'away_szn_runs_allowed', 'away_pythag_win_pct',
                'home_total_games', 'away_total_games'
            ]],
            on='game_id',
            how='left'
        )
    )

    cols = ['wins', 'games', 'win_pct', 'run_diff', 'szn_runs_scored', 'szn_runs_allowed', 'pythag_win_pct', 'total_games']
    df_full[[f'home_{c}' for c in cols]] = df_full.groupby('home_name')[[f'home_{c}' for c in cols]].ffill()
    df_full[[f'away_{c}' for c in cols]] = df_full.groupby('away_name')[[f'away_{c}' for c in cols]].ffill()

    return df_full


def _calculate_result_columns(df):
    df['total_score'] = df['home_score'] + df['away_score']
    df['home_win'] = df['home_score'] > df['away_score']
    df['away_win'] = df['away_score'] > df['home_score']
    df['over_open'] = np.where(df['total_score'] > df['totals_open_point'], 1, 0)
    df['under_open'] = np.where(df['total_score'] < df['totals_open_point'], 1, 0)
    return df


def _fetch_weather(game_data: pd.DataFrame):
    print('Loading weather data')
    df_games = game_data[~game_data.status.isin(['Postponed', 'Cancelled'])]
    df_games = df_games.set_index('game_id')

    # Check if file exists or if new run
    data_directory = mlb_stats_api['data_directory']
    file = mlb_stats_api['weather_file']
    if not use_s3:
        path = Path(data_directory)
        weather_file = path / file
        new_run = not weather_file.exists()
    else:
        weather_file = f'{data_directory}/{file}'
        new_run = not s3.exists(weather_file)

    df_weather = pd.DataFrame()

    if new_run:
        games_to_load = df_games.index.tolist()
    else:
        df_weather = pd.read_parquet(weather_file, filesystem=filesystem)
        games_to_load = df_games[~df_games.index.isin(df_weather.index)].index.tolist()

    for game_id in (pbar := tqdm(games_to_load)):
        pbar.set_description(str(game_id))

        # Game
        game = statsapi.boxscore_data(gamePk=game_id)

        box_info = game['gameBoxInfo']

        weather = [d['value'] for d in box_info if d['label'] == 'Weather']
        wind = [d['value'] for d in box_info if d['label'] == 'Wind']

        if len(weather) == 0 or len(wind) == 0:
            continue

        df = pd.DataFrame({
            'weather': weather,
            'wind': wind
        }, index=[game_id])

        load = not df.empty
        if load:
            df[['weather_temp', 'weather_cond']] = df['weather'].str.split(',', expand=True)
            df['weather_temp'] = df['weather_temp'].str.replace(' degrees', '')
            df['weather_cond'] = df['weather_cond'].str.replace('.', '')

            df[['wind_speed', 'wind_dir']] = df['wind'].str.split(',', expand=True)
            df['wind_speed'] = df['wind_speed'].str.replace(' mph', '')
            df['wind_dir'] = df['wind_dir'].str.replace('.', '')

        if new_run:
            new_run = False
        else:
            df_weather.update(df)
            df = df[~df.index.isin(df_weather.index)]
            df = pd.concat([df_weather, df])

        df.to_parquet(weather_file, filesystem=filesystem)
        df_weather = df.copy()

    return df_weather


def _combine_weather_data(df_games, df_weather):
    df_games.set_index('game_id', inplace=True)

    df = df_games.join(df_weather)

    return df.reset_index()


def _cleanup(df):
    # remove games with pitchers with few pitches
    # df = df[df['away_Pit_season'] >= minimum_pitches]
    # df = df[df['home_Pit_season'] >= minimum_pitches]

    # remove games with batters with few plate appearances
    # for team in ['away', 'home']:
    #     for i in range(1, 10):
    #         df = df[df[f'{team}_batter_{i}_PA_season'] >= minimum_plate_appearances]

    # drop pitcher notes. It is just a constant value.
    df.drop(['home_pitcher_note', 'away_pitcher_note'], axis='columns', inplace=True)

    # drop national broadcasts and summary
    df.drop(['national_broadcasts', 'summary'], axis='columns', inplace=True)

    return df


def _simple_feature_engineering(df):
    # df['total_over_open_points'] = np.where(df['total_score'] > df['totals_open_point'], 1, 0)
    # df['total_under_open_points'] = 1 - df['total_over_open_points']

    df['home_runs_per_game'] = df['home_szn_runs_scored'] / df['home_total_games']
    df['away_runs_per_game'] = df['away_szn_runs_scored'] / df['away_total_games']
    df['total_runs_per_game'] = df['home_runs_per_game'] + df['away_runs_per_game']

    df['total_pitches'] = df['away_Pit_season'] + df['home_Pit_season']
    df['total_weighted_ERA'] = ((df['away_ERA_season'] * df['away_Pit_season'] + df['home_ERA_season'] * df['home_Pit_season']) / df['total_pitches']) * 2

    # df['totals_point_movement'] = df['totals_close_point'] - df['totals_open_point']
    # df.loc[df['totals_point_movement'] < 0, 'totals_move_direction'] = -1
    # df.loc[df['totals_point_movement'] == 0, 'totals_move_direction'] = 0
    # df.loc[df['totals_point_movement'] > 0, 'totals_move_direction'] = 1

    return df


def _data_availability_flags(df):
    df['lineups_available'] = (
            df['away_probable_pitcher_id'].notnull() &
            df['away_batter_1'].notnull() &
            df['home_probable_pitcher_id'].notnull() &
            df['home_batter_1'].notnull()
    )

    df['weather_available'] = (
            df['weather'].notnull() &
            df['wind'].notnull()
    )

    return df


def load_and_create_dataset():
    # 1. Games & Odds
    # Check for and create the mlb data folder
    data_directory = mlb_stats_api['data_directory']
    path = Path(data_directory)
    _directory_check_and_create(path)

    # Fetch, save and return MLB game data from MLB Stats API
    game_data = _mlb_game_data()

    # Check for and create the odds api data folder
    data_directory = the_odds_api['data_directory']
    path = Path(data_directory)
    _directory_check_and_create(path)

    # Fetch historic odds data, save, and return all odds data
    _ = _historic_odds_data(game_data)
    # Fetch current odds data, save, and return all odds data
    _ = _current_odds_data()

    # Clean, prepare, save, and return odds datasets
    df_h2h, df_totals = _build_odds_datasets()

    # combine games and odds
    _create_combined_games_and_odds_dataset(game_data, df_h2h, df_totals)

    # 2. Lineups
    # Fetch and save historic lineups
    df_historic = _historic_lineup_data(game_data)

    # Fetch and save probable lineups
    df_probable = _probable_lineup_data(game_data)

    # add lineups into combined dataset
    df_combined = _combine_lineup_data(df_historic, df_probable)

    # 3. Pitching Stats
    # Check for and create the baseball ref data folder
    data_directory = baseball_ref['data_directory']
    path = Path(data_directory)
    _directory_check_and_create(path)

    _pitching_stats_by_day(df_combined)

    # 4. Batting Stats
    _batting_stats_by_day(df_combined)

    # Aggregate stats
    df_combined = df_combined.reset_index()
    # Pitching
    df_combined = _agg_and_append_pitching_stats(df_combined)
    # Hitting
    df_combined = _agg_and_append_batting_stats(df_combined)
    # Bullpen
    df_combined = _agg_and_append_bullpen_stats(df_combined)

    # 6. Weather
    df_weather = _fetch_weather(df_combined)
    df_combined = _combine_weather_data(df_combined, df_weather)

    # 8. Team Stats
    df_combined = _calculate_home_and_away_win_pct(df_combined)

    # Game results
    df_combined = _calculate_result_columns(df_combined)

    # Cleanup dataset
    df_clean = _cleanup(df_combined)

    # Simple feature engineering
    df_clean = _simple_feature_engineering(df_clean)

    # Data availability flags
    df_clean = _data_availability_flags(df_clean)

    # Model Over Under probs
    df_mod = model_ou_probability(df_clean, cv_iters=10)

    # Save complete dataset
    df_mod.to_parquet(complete_data_parquet, filesystem=filesystem)
    df_mod.to_csv(complete_data_csv)
