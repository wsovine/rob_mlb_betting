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


def refresh_dataset():
    start_time = time.time()
    with st.spinner('Loading latest data and running models.'):
        load_and_create_dataset()

    end_time = time.time()
    mins, secs = time_convert(end_time - start_time)
    st.success(f'Refresh Complete | {mins}:{secs}')


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
        'game_datetime_str', 'status', 'away_name', 'home_name', 'totals_open_point',
        'lineups_available', 'weather_available',
        'over_prob', 'under_prob', 'total_score'
    ]]
    .tail(30)
    .style
    .map(style_color_bool, subset=['lineups_available', 'weather_available'])
    .format({
        'over_prob': '{:.1%}'.format,
        'under_prob': '{:.1%}'.format,
        'totals_open_point': '{:.1f}'.format
    })
)

col1, col2 = st.columns(2)
with col1:
    st.button(
        'Refresh Data',
        on_click=refresh_dataset
    )
    st.download_button(
        'Download Full CSV',
        df.to_csv(index=False).encode('utf-8'),
        file_name='complete_dataset.csv',
        mime='text/csv'
    )

with col2:
    st.text('May take up to 15 minutes to refresh.')


