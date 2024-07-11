import streamlit as st
import pandas as pd
import time
from tasks import load_and_create_dataset
import config
from s3fs.core import S3FileSystem

use_s3 = config.s3
filesystem = None
if use_s3:
    s3 = S3FileSystem(anon=False, key=config.AWS_ACCESS_KEY_ID, secret=config.AWS_SECRET_ACCESS_KEY)
    filesystem = s3


def time_convert(sec):
    mins = sec // 60
    sec = sec % 60
    mins = mins % 60
    return int(mins), int(sec)


def style_color_bool(val):
    color = 'green' if val else 'red'
    return f'background-color: {color}'


st.title('Rob MLB Betting Data Download')

st.subheader('Latest 30 Records')
df = pd.read_parquet(config.complete_data_parquet, filesystem=filesystem)

df['game_datetime'] = df['game_datetime'].dt.tz_convert('America/Chicago')
df['game_datetime_str'] = df['game_datetime'].dt.strftime('%m-%d-%Y %-I:%M')

st.dataframe(
    df[pd.to_datetime(df.game_date) <= pd.to_datetime('today')]
    .sort_values('game_datetime', ascending=True)
    [[
        'game_datetime_str', 'status', 'away_name', 'home_name',
        'lineups_available', 'weather_available',
        'totals_open_point',
        'over_prob', 'under_prob', 'total_score',
        'away_win_prob', 'min_away_odds', 'away_score',
        'home_win_prob', 'min_home_odds', 'home_score'
    ]]
    .rename(columns={
        'game_datetime_str': 'Game Time',
        'status': 'Status',
        'away_name': 'Away',
        'home_name': 'Home',
        'lineups_available': 'Lineups',
        'weather_available': 'Weather',
        'totals_open_point': 'Total (Open)',
        'over_prob': 'Over %',
        'under_prob': 'Under %',
        'total_score': 'Total Score',
        'away_win_prob': 'Away Win %',
        'min_away_odds': 'Away Min. Odds',
        'away_score': 'Away Score',
        'home_win_prob': 'Home Win %',
        'min_home_odds': 'Home Min. Odds',
        'home_score': 'Home Score'
    })
    .tail(30)
    .style
    .map(style_color_bool, subset=['Lineups', 'Weather'])
    .format({
        'Over %': '{:.1%}'.format,
        'Under %': '{:.1%}'.format,
        'Total (Open)': '{:.1f}'.format,
        'Away Win %': '{:.1%}'.format,
        'Home Win %': '{:.1%}'.format,
        # 'Away Score': '{:.1f}'.format,
        # 'Home Score': '{:.1f}'.format,
    })
)

st.download_button(
    'Download Full CSV',
    df.to_csv(index=False).encode('utf-8'),
    file_name='complete_dataset.csv',
    mime='text/csv'
)
st.subheader('Totals')
st.text('Rule of thumb for totals - \nbet 55%+ for profitability and 56%+ for optimal ROI based on backtesting. ')
st.subheader('H2H')
st.text('Rule of thumb for moneyline - \nbet based on the minimum odds. \nBacktesting showed profitability when'
        ' implied odds were off by 4%+ \nand optimal ROI at 6%+. \nMinimum odds are calibrated for 4%.')


# col1, col2 = st.columns(2)
# with col1:
# with col2:



