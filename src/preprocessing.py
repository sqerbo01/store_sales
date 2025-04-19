import pandas as pd
import numpy as np
import os


def load_data(data_path):
    train = pd.read_csv(os.path.join(data_path, 'train.csv'), parse_dates=['date'])
    stores = pd.read_csv(os.path.join(data_path, 'stores.csv'))
    oil = pd.read_csv(os.path.join(data_path, 'oil.csv'), parse_dates=['date'])
    holidays = pd.read_csv(os.path.join(data_path, 'holidays_events.csv'), parse_dates=['date'])
    transactions = pd.read_csv(os.path.join(data_path, 'transactions.csv'), parse_dates=['date'])
    return train, stores, oil, holidays, transactions


def merge_data(train, stores, oil, holidays, transactions):
    df = train.merge(stores, on='store_nbr', how='left')
    df = df.merge(transactions, on=['date', 'store_nbr'], how='left')
    df = df.merge(oil, on='date', how='left')
    df = df.merge(
        holidays[['date']].drop_duplicates().assign(is_holiday=1),
        on='date', how='left'
    )
    df['is_holiday'] = df['is_holiday'].fillna(0)
    return df


def add_time_features(df):
    df["dayofweek"] = df["date"].dt.dayofweek
    df["weekday_name"] = df["date"].dt.day_name()
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["dayofmonth"] = df["date"].dt.day
    df["is_weekend"] = df["dayofweek"].isin([5, 6])  # Saturday, Sunday
    df["is_month_start"] = df["date"].dt.is_month_start
    df["is_month_end"] = df["date"].dt.is_month_end
    df["is_quarter_start"] = df["date"].dt.is_quarter_start
    df["is_quarter_end"] = df["date"].dt.is_quarter_end
    df["n_days_from_start"] = (df["date"] - df["date"].min()).dt.days
    df['day'] = df['date'].dt.day
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    return df


def create_lag_features(df, lags=[1], windows=[7]):
    for lag in lags:
        df[f'lag_{lag}'] = df.groupby(['store_nbr', 'family'])['sales'].shift(lag)
    for window in windows:
        df[f'rolling_mean_{window}'] = df.groupby(['store_nbr', 'family'])['sales'].shift(1).rolling(window).mean().reset_index(0, drop=True)
        df[f'rolling_std_{window}'] = df.groupby(['store_nbr', 'family'])['sales'].shift(1).rolling(window).std().reset_index(0, drop=True)
    return df


def encode_categoricals(df):
    df['family'] = df['family'].astype('category').cat.codes
    df['store_nbr'] = df['store_nbr'].astype('category')
    return df


def preprocess_pipeline(data_path):
    train, stores, oil, holidays, transactions = load_data(data_path)
    df = merge_data(train, stores, oil, holidays, transactions)
    df = add_time_features(df)
    df = create_lag_features(df)
    df = encode_categoricals(df)
    # df['log_sales'] = np.log1p(df['sales'])
    df.fillna(method='bfill', inplace=True)
    return df


def save_processed_data(df, output_path):
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    data_path = 'data/store-sales-time-series-forecasting/'
    output_path = 'data/processed_train.csv'
    df = preprocess_pipeline(data_path)
    save_processed_data(df, output_path)
    print("Preprocessing complete. File saved to:", output_path)
