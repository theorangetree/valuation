# Average industry data by sector and industry from line-by-line company data csv

import pandas as pd

# Set display options to show more rows and columns than default
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 200)

def clean_industry_data(df):
    # Replace low-saample size industry and sector values with similar industries
    df = df[df.country == 'United States']

    # Remove companies with multiple share classes and tickers
    secondary_tickers = ['BATRK','FWONA','LSXMK','CENTA','FOX','GOOG','LBRDA','LILAK','NWS','RUSHB','UA','VIAC','ZG']
    df = df[~df.index.isin(secondary_tickers)]
    
    # Combine similar sectors
    df = df.replace({'sector':{'Financial':'Financial Services'}})

    # Combine similar industries
    df = df.replace({'industry':{r'.*Banks.*':'Banks',
                                 r'^Beverages.*':'Beverages',
                                 r'^Insurance.*':'Insurance',
                                 r'^Drug Manufacturers.*':'Drug Manufacturers',
                                 r'^REIT.*':'REIT',
                                 r'^Utilities.*':'Utilities'
                                 }},
                    regex=True)
    
    #df = df.replace({'industry':
    #df.industry.replace()
    return df

def industry_averages(csv_path):
    df = pd.read_csv(csv_path, index_col='Unnamed: 0')
    df = clean_industry_data(df)
    df_avg_industry = df.groupby('industry').agg(['count', 'mean', 'median'])
    df_avg_sector   = df.groupby('sector').agg(['count', 'mean', 'median'])
    print(df_avg_industry)
    #print(df_avg_sector)
    df_avg_industry.to_csv('industry_avgs.csv')
    df_avg_sector.to_csv('sector_avgs.csv')
    

industry_averages('industry_data.csv')
