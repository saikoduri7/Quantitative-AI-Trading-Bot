import yfinance as yf


def download_data(
    ticker,
    start_date,
    end_date
):
    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=True
    )

    return df
