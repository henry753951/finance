from FinMind.data import DataLoader
def update_stock_info():
    api = DataLoader()
    # api.login_by_token(api_token='token')
    df = api.taiwan_stock_info()
    print(df)
    df.to_csv('data/taiwan_stock_info.csv', index=False)

if __name__ == '__main__':
    update_stock_info()