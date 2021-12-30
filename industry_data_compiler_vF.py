# Main function:
# industry_averages(csv_path: str, ticker='----', write=False)

# Purpose:
# Average industry data by sector and industry from line-by-line stock data csv file, excluding specified ticker
# [Optional] If write=True, outputs averages as csv file

import pandas as pd

def filter_companies(df, ticker='----'):
    # [Optional ticker argument] Remove the valuation company so that it is excluded from the averages
    df = df[df.index != ticker]

    # Filter for U.S. companies
    df = df[df.country == 'United States']

    # Remove non-primary tickers for companies with multiple share classes and tickers
    secondary_tickers = ['BATRK','FWONA','LSXMK','CENTA','DISCK','FOX','GOOG','LBRDA','LILAK','NWS','RUSHB','UA','VIAC','ZG']
    df = df[~df.index.isin(secondary_tickers)]

    return df

def clean_industry_data(df):
    # This function replaces low sample-size industry and sector values with similar industries
    # Combine similar sectors
    df = df.replace({'sector'  :{'Financial':'Financial Services'}})
    
    # Combine similar industries
    df = df.replace({'industry':{r'.*Banks.*'               :'Banks',
                                 r'^Beverages.*'            :'Beverages',
                                 r'^Drug Manufacturers.*'   :'Drug Manufacturers',
                                 r'^Insurance.*'            :'Insurance',
                                 r'^Real Estate.*'          :'Real Estate Services',
                                 r'^REIT.*'                 :'REIT',
                                 r'^Utilities.*'            :'Utilities'
                                 r'^Oil & Gas (?!Equipment & Services).*$':'Oil & Gas',
                                 }},
                    regex=True)
    
    df = df.replace({'industry':{'Airports & Air Services'          :'Rental & Leasing Services',
                                 'Lumber & Wood Production'         :'Building Materials',
                                 'Aluminum'                         :'Industrial Metals & Mining',
                                 'Copper'                           :'Industrial Metals & Mining',
                                 'Steel'                            :'Industrial Metals & Mining',
                                 'Other Industrial Metals & Mining' :'Industrial Metals & Mining',
                                 'Coking Coal'                      :'Industrial Metals & Mining',
                                 'Thermal Coal'                     :'Industrial Metals & Mining',
                                 'Uranium'                          :'Industrial Metals & Mining',
                                 'Gold'                             :'Precious Metals & Mining',
                                 'Other Precious Metals & Mining'   :'Precious Metals & Mining',
                                 'Pharmaceutical Retailers'         :'Grocery Stores',
                                 'Department Stores'                :'Apparel Retail',
                                 'Footwear & Accessories'           :'Apparel Retail',
                                 'Textile Manufacturing'            :'Apparel Manufacturing',
                                 'Confectioners'                    :'Packaged Foods',
                                 'Infrastructure Operations'        :'Engineering & Construction',
                                 'Pollution & Treatment Controls'   :'Building Products & Equipment',
                                 'Financial Conglomerates'          :'Asset Management',
                                 'Electronics & Computer Distribution':'Information Technology Services'
                                 }})
    return df

def industry_averages(csv_path, ticker='----', write=False):
    df = pd.read_csv(csv_path, index_col=0)
    avgs_date = df.index.name

    df = filter_companies(df)
    df = clean_industry_data(df)

    df_avg_industry = df.groupby('industry').agg(['count', 'mean', 'median'])
    df_avg_sector   = df.groupby('sector').agg(['count', 'mean', 'median'])

    df_avg_industry.index.name = avgs_date
    df_avg_sector.index.name   = avgs_date

    # Output sector and industry averages as csv files
    if write == True:
        df_avg_industry.to_csv('industry_avgs.csv')
        df_avg_sector.to_csv('sector_avgs.csv')

    return df_avg_sector, df_avg_industry, avgs_date

#industry_averages('industry_data.csv', write=True)
