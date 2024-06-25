import streamlit as st
import pandas as pd
import time
from tasks import load_and_create_dataset
from models import model_ou_probability


def time_convert(sec):
    mins = sec // 60
    sec = sec % 60
    mins = mins % 60
    return int(mins), int(sec)


def refresh_dataset():
    start_time = time.time()
    with st.spinner('Loading latest data.'):
        load_and_create_dataset()
    with st.spinner('Modeling Over/Under probabilities.'):
        data_df = pd.read_parquet('data/complete_dataset.parquet')
        model_ou_probability(df=data_df, target='over_open', cv_iters=10)

    end_time = time.time()
    mins, secs = time_convert(end_time - start_time)
    st.success(f'Refresh Complete | {mins}:{secs}')


def color_bool(val):
    color = 'green' if val else 'red'
    return f'background-color: {color}'


st.title('Rob MLB Betting Data Download')

st.subheader('Latest 20 Records')
df = pd.read_parquet('data/complete_dataset.parquet')

st.dataframe(
    df[[
        'game_date', 'status', 'away_name', 'home_name', 'totals_open_point',
        'lineups_available', 'weather_available',
        'over_prob', 'under_prob', 'total_score'
    ]]
    .tail(20)
    .style.map(color_bool, subset=['lineups_available', 'weather_available'])
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


