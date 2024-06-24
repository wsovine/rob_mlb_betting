import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from pathlib import Path
from config import baseball_ref
import warnings


def _calculate_era(df: pd.DataFrame):
    df['ERA'] = ((df['ER'] * 9) / df['IP']).round(2)
    return df


def _calculate_fip(df: pd.DataFrame, all_pitching_stats: pd.DataFrame):
    lg = all_pitching_stats[['ER', 'IP', 'HR', 'BB', 'HBP', 'SO']].sum()
    lg_era = (lg['ER'] * 9) / lg['IP']
    fip_constant = lg_era - (((13 * lg['HR']) + (3*(lg['BB'] + lg['HBP'])) - (2*lg['SO'])) / lg['IP'])

    df['FIP'] = (((df['HR']*13) + ( 3*(df['BB'] + df['HBP'])) - (2*df['SO'])) / df['IP'] + fip_constant).round(2)
    return df


def _calculate_kbb(df: pd.DataFrame):
    df['K/BB'] = (df['SO'] / df['BB']).round(2)
    return df


def aggregated_pitching_stats(df_games, last_games: int = None, bullpen: bool = False) -> pd.DataFrame:
    """
    Aggregate pitching stats from prior games
    :param bullpen: True will aggregate relief pitchers by team
    :param df_games: list of games that pitching stats are needed ahead of
    :param last_games: if None, then the full season. Provide int for last n games.
    :return: DataFrame
    """
    path = Path(baseball_ref['data_directory'])
    pitching_stats_file = path / baseball_ref['pitching_stats_file']

    df_full = pd.read_parquet(pitching_stats_file).reset_index()
    df_full = df_full if not bullpen else df_full[df_full['GS'] == 0]
    group_col = 'mlbID' if not bullpen else ['Tm', 'Lev']

    dfs = []

    for season in df_games.season.unique():
        print(f'Aggregating pitches for {season}')
        df_season = df_games[df_games.season == season]
        start_date = df_season.game_date.min()
        for date in (pbar := tqdm(df_season.game_date.unique())):
            pbar.set_description(date)
            df_date = df_full[(df_full['game_date'] >= start_date) & (df_full['game_date'] < date)]
            if last_games is not None:
                df = df_date.sort_values('game_date').groupby(group_col).tail(last_games)
            else:
                df = df_date.copy()

            df = df.groupby(group_col).agg({
                'G': 'sum',
                'GS': 'sum',
                'W': 'sum',
                'L': 'sum',
                'SV': 'sum',
                'IP': 'sum',
                'H': 'sum',
                'R': 'sum',
                'ER': 'sum',
                'BB': 'sum',
                'SO': 'sum',
                'HR': 'sum',
                'HBP': 'sum',
                'AB': 'sum',
                '2B': 'sum',
                '3B': 'sum',
                'IBB': 'sum',
                'GDP': 'sum',
                'SF': 'sum',
                'SB': 'sum',
                'CS': 'sum',
                'PO': 'sum',
                'BF': 'sum',
                'Pit': 'sum',
            }).reset_index()

            df['season'] = season
            df['game_dates_prior_to'] = date
            
            # stat calculations
            warnings.filterwarnings('ignore')
            # ERA
            df = _calculate_era(df)
            # FIP
            df = _calculate_fip(df, df_date)
            # K/BB
            df = _calculate_kbb(df)
            warnings.filterwarnings('default')

            df = df.replace([np.inf, -np.inf], np.nan)

            dfs.append(df)

    return pd.concat(dfs)


def _calculate_ba(df):
    df['BA'] = (df['H'] / df['AB']).round(3)
    return df


def _calculate_obp(df):
    df['OBP'] = ((df['H'] + df['BB'] + df['HBP']) / 
                 (df['AB'] + df['BB'] + df['HBP'] + df['SF'])).round(3)
    return df


def _calculate_slg(df):
    df['1B'] = df['H'] - df['2B'] - df['3B'] - df['HR']

    df['SLG'] = ((df['1B'] + 2*df['2B'] + 3*df['3B'] + 4*df['HR']) / 
                 df['AB']).round(3)
    
    return df


def _calculate_ops(df):
    df['OPS'] = df['OBP'] + df['SLG']
    return df


def aggregated_batting_stats(df_games, last_games: int = None) -> pd.DataFrame:
    """
    Aggregate batting stats from prior games
    :param df_games: list of games that batting stats are needed ahead of
    :param last_games: if None, then the full season. Provide int for last n games.
    :return: DataFrame
    """
    path = Path(baseball_ref['data_directory'])
    batting_stats_file = path / baseball_ref['batting_stats_file']
    dfs = []

    for season in df_games.season.unique():
        print(f'Aggregating batting stats for {season}')
        df_season = df_games[df_games.season == season]
        start_date = df_season.game_date.min()
        for date in (pbar := tqdm(df_season.game_date.unique())):
            pbar.set_description(date)
            df_full = pd.read_parquet(batting_stats_file).reset_index()
            df_full = df_full[(df_full['game_date'] >= start_date) & (df_full['game_date'] < date)]
            if last_games is not None:
                df = df_full.sort_values('game_date').groupby('mlbID').tail(last_games)
            else:
                df = df_full.copy()
            df = df.groupby('mlbID').agg({
                'G': 'sum',
                'PA': 'sum',
                'AB': 'sum',
                'R': 'sum',
                'H': 'sum',
                '2B': 'sum',
                '3B': 'sum',
                'HR': 'sum',
                'RBI': 'sum',
                'BB': 'sum',
                'IBB': 'sum',
                'SO': 'sum',
                'HBP': 'sum',
                'SH': 'sum',
                'SF': 'sum',
                'GDP': 'sum',
                'SB': 'sum',
                'CS': 'sum'
            }).reset_index()

            df['season'] = season
            df['game_dates_prior_to'] = date

            # stat calculations
            warnings.filterwarnings('ignore')
            # BA
            df = _calculate_ba(df)
            # OBP
            df = _calculate_obp(df)
            # SLG
            df = _calculate_slg(df)
            # OPS
            df = _calculate_ops(df)
            warnings.filterwarnings('default')

            df = df.replace([np.inf, -np.inf], np.nan)

            dfs.append(df)

    return pd.concat(dfs)


def ou_correct(row):
    if row['total_score'] == row['totals_open_point']:
        return 0
    elif row['over_bet'] + row['under_bet'] == 0:
        return 0
    elif row['over_bet'] & (row['total_score'] > row['totals_open_point']):
        return 1
    elif row['under_bet'] & (row['total_score'] < row['totals_open_point']):
        return 1
    return -1


def odds_to_profit(odds, u = 1):
    if odds < 0:
        profit = -100 / odds
    else:
        profit = odds / 100
    return profit * u


def ou_win_payout(row):
    if row['total_score'] > row['totals_open_point']:
        return odds_to_profit(row['totals_open_price_over'])
    elif row['total_score'] < row['totals_open_point']:
        return odds_to_profit(row['totals_open_price_under'])
    return 0
