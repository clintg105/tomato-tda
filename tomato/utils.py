import pandas as pd
from pathlib import Path

def tomato_repo_path():
    return Path(__file__).resolve().parents[1]

def tomato_data_path():
    return tomato_repo_path() / 'data'

def load_critic_review_df():
    return pd.read_csv(
        tomato_data_path() / 'rotten_tomatoes_critic_reviews.csv')