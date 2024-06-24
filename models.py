import xgboost as xgb
import pandas as pd

from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split


def model_ou_probability(df, target: str = 'over_open', cv_iters: int = 100):
    mod = xgb.XGBClassifier
    model_params = dict(
        enable_categorical=True,
        eta=0.01,
        max_depth=3,
        early_stopping_rounds=50,
        n_estimators=2000,
        subsample=.7,
        alpha=10,
        objective='binary:logistic',
        eval_metric='logloss',
        tree_method='hist',
        scale_pos_weight=1.15
    )
    model_params['lambda'] = 100

    pitch_stats = ['ERA', 'FIP', 'K/BB']
    pitch_stats = [f'away_{s}' for s in pitch_stats] + [f'home_{s}' for s in pitch_stats]
    pitch_stats = [c for c in df.columns if c.startswith(tuple(pitch_stats))]
    
    bat_stats = ['BA', 'OBP', 'SLG', 'OPS']
    bat_stats = (
            [f'home_batter_{i}_{s}' for i in range(1, 10) for s in bat_stats] +
            [f'away_batter_{i}_{s}' for i in range(1, 10) for s in bat_stats]
    )
    bat_stats = [c for c in df.columns if c.startswith(tuple(bat_stats))]
    
    bullpen_stats = ['ERA', 'FIP', 'K/BB']
    bullpen_stats = [f'away_bullpen_{s}' for s in bullpen_stats] + [f'home_bullpen_{s}' for s in bullpen_stats]
    bullpen_stats = [c for c in df.columns if c.startswith(tuple(bullpen_stats))]

    df['venue_name'] = df['venue_name'].astype('category')
    df['season'] = df['season'].astype('category')
    df['weather_cond'] = df['weather_cond'].astype('category')
    df['wind_dir'] = df['wind_dir'].astype('category')
    category_cols = ['venue_name', 'season', 'weather_cond', 'wind_dir']
    
    other_numeric = ['weather_temp', 'wind_speed', 'totals_open_point']
    for c in other_numeric:
        if df[c].dtype == 'object':
            df[c] = df[c].astype('Int64')
    
    X_cols = (pitch_stats + bat_stats + bullpen_stats + category_cols + other_numeric)
    y_col = target
    
    train_ratio = 0.70

    predictions = []

    df_train = df[df['status'] == 'Final']

    for _ in tqdm(range(cv_iters)):
        X_train, X_val, y_train, y_val = train_test_split(df_train[X_cols], df_train[y_col], test_size=1 - train_ratio)

        eval_set = [(X_train, y_train), (X_val, y_val)]

        m = mod(**model_params)
        m = m.fit(X_train, y_train, eval_set=eval_set, verbose=0)

        y_pred = m.predict(df[X_cols])
        y_prob = m.predict_proba(df[X_cols])[:, 1]

        # df['over_pred'] = y_pred
        # df['under_pred'] = 1 - y_pred
        df['over_prob'] = y_prob
        df['under_prob'] = 1 - y_prob

        predictions.append(df)

    df_pred = pd.concat(predictions).groupby('game_id').agg({
        # 'over_pred': 'mean',
        # 'under_pred': 'mean',
        'over_prob': 'mean',
        'under_prob': 'mean'
    }).reset_index()

    df = df.drop([
        # 'over_pred', 'under_pred',
        'over_prob', 'under_prob'
    ], axis='columns').merge(df_pred, on='game_id')

    df.to_parquet('data/complete_dataset.parquet')

    return df
