import streamlit as st
import pandas as pd
from tasks import load_and_create_dataset


def refresh_dataset():
    with st.spinner('Refresh in progress...'):
        load_and_create_dataset()
    st.success('Refresh Complete')


st.title('Rob MLB Betting Data Download')

st.subheader('Latest 20 Records')
df = pd.read_parquet('data/complete_dataset.parquet')

st.write(df.tail(20))

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
