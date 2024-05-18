import numpy as np
import pandas as pd
import joblib

def main():
    train=pd.read_csv('./store-sales-time-series-forecasting/train.csv')
    #test=pd.read_csv('./store-sales-time-series-forecasting/test.csv')

    # merge train and test features
    #df=pd.concat([train, test]).reset_index(drop=True)
    df=pd.concat([train]).reset_index(drop=True)

    # read dataframes
    holidays = pd.read_csv('./store-sales-time-series-forecasting/holidays_events.csv')
    oil = pd.read_csv('./store-sales-time-series-forecasting/oil.csv')

    # transfer date to datetime type
    holidays.isna().sum()
    # get rid of unnecessary columns
    holidays.drop(columns=['description','locale_name'], inplace=True)

    oil.isna().sum()
    oil[oil['dcoilwtico'].isna()]
    oil['dcoilwtico'].fillna(method='ffill', inplace=True)
    oil['dcoilwtico'].fillna(method='bfill', inplace=True)

    # merge holidays
    df = df.merge(holidays,how='left',on='date')
    df = df.merge(oil,how='left',on='date')
    df = pd.get_dummies(df, columns=['family'], dummy_na=False, prefix='family')
    df = pd.get_dummies(df, columns=['type'], dummy_na=False, prefix='holidayType')
    df = pd.get_dummies(df, columns=['locale'], dummy_na=False, prefix='holidayLocale')
    df = pd.get_dummies(df, columns=['transferred'], dummy_na=False, prefix='holidayTransferred')

    new_oil=df[['date','dcoilwtico']].drop_duplicates(subset='date', keep="first").reset_index(drop=True)
    # fill with adjcent values
    new_oil['dcoilwtico'].fillna(method='ffill', inplace=True)
    new_oil['dcoilwtico'].fillna(method='bfill', inplace=True)

    # remove the old column from df before merging with the new_oil
    df=df.drop(columns=['dcoilwtico']) 
    df = df.merge(new_oil,how='left',on='date')

    # remove duplicates
    boolean_columns = [
        'holidayType_Additional',
        'holidayType_Bridge',
        'holidayType_Event',
        'holidayType_Holiday',
        'holidayType_Transfer',
        'holidayType_Work Day',
        'holidayLocale_Local',
        'holidayLocale_National',
        'holidayLocale_Regional',
        'holidayTransferred_False',
        'holidayTransferred_True'
    ]

    # Group by 'id' and take the union of the boolean columns
    df_grouped = df.groupby('id')[boolean_columns].max().reset_index()

    # Take the first occurrence of non-boolean columns
    non_boolean_columns = [col for col in df.columns if col not in boolean_columns]
    non_boolean_columns.remove('id')
    df_non_boolean = df.groupby('id')[non_boolean_columns].first().reset_index()

    # Merge the non-boolean and boolean DataFrames
    df = pd.merge(df_non_boolean, df_grouped, on='id')

    print("Saving cleaned and combined data to store_data.csv")
    df.to_csv("store_data.csv",  sep=',', encoding='utf-8', index=False)

if __name__ == "__main__":
    main()
