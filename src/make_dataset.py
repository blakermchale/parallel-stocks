
import pandas as pd

if __name__ == '__main__':

    df = pd.read_csv("../data/raw/bitstampUSD.csv", index_col="Timestamp")
    df['Volume_(BTC)'].fillna(value=0, inplace=True)
    df['Volume_(Currency)'].fillna(value=0, inplace=True)
    df['Weighted_Price'].fillna(value=0, inplace=True)

    df['Open'].fillna(method='ffill', inplace=True)
    df['High'].fillna(method='ffill', inplace=True)
    df['Low'].fillna(method='ffill', inplace=True)
    df['Close'].fillna(method='ffill', inplace=True)

    df.to_csv("../data/processed/bitstampUSD.csv")
    print(df.head())
    print("Done")

