import pandas as pd


def load_prices(csv_path, price_column="close", date_column="date"):
    df = pd.read_csv(csv_path)
    df[date_column] = pd.to_datetime(df[date_column].values)
    df.sort_values(by=date_column, inplace=True)
    prices = df[price_column].values
    return prices.astype(float)
