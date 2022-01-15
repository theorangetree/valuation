# Main function:
    # industry_aggregates(csv_path: str, ex_ticker='----', write=False)

# Purpose:
    # Aggregate stock data by sector and industry from specified line-by-line market data csv file
    # [Optional] Exclude specified ticker from aggregates
    # [Optional] If write=True, outputs aggregates as csv file

import pandas as pd

def filter_companies(df, ex_ticker):
    # [Optional ex_ticker argument] Exclude the valuation company ticker so that it is excluded from the averages
    df = df[df.index != ex_ticker]

    # Filter for U.S. companies
    df = df[df.country == 'United States']

    # Remove non-primary tickers for companies with multiple share classes and tickers
    secondary_tickers = ['BATRK','FWONA','LSXMK','CENTA','DISCK','FOX','GOOG','LBRDA','LILAK','NWS','RUSHB','UA','VIAC','ZG'] # List needs to be manually updated
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
                                 r'^Utilities.*'            :'Utilities',
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

def percentile(n):
    # Create percentile function usable with Pandas groupby().agg()
    def percentile_(x):
        return x.quantile(n)
    percentile_.__name__ = 'percentile_{:2.0f}'.format(n*100)
    return percentile_

def industry_aggregates(csv_path, ex_ticker='----', limited=True, write=False):
    df = pd.read_csv(csv_path, index_col=0)
    aggs_date = df.index.name

    df = filter_companies(df, ex_ticker=ex_ticker)
    df = clean_industry_data(df)
    
    if limited == True: # Only aggregate data needed for valuation
        col_subset = ['sector','industry','revenueGrowth','operatingMargins']
        if 'unleveredBeta' in df.columns:
            col_subset.append('unleveredBeta')
        else:
            print('Must run Company_Info().calculated_fields() method to calculate unlevered betas in market data')

        # Aggregate data
        df_agg_sector   = df.loc[:,col_subset].groupby('sector'  ).agg(['count', 'mean', percentile(0.25), 'median', percentile(0.75)])
        df_agg_industry = df.loc[:,col_subset].groupby('industry').agg(['count', 'mean', percentile(0.25), 'median', percentile(0.75)])     
    else:
        df_agg_sector   = df.groupby('sector'  ).agg(['count', 'mean', percentile(0.25), 'median', percentile(0.75)])
        df_agg_industry = df.groupby('industry').agg(['count', 'mean', percentile(0.25), 'median', percentile(0.75)])

    if 'investedCapital' in df.columns: # Requires Invested Capital column to be calculated first
        # Identify rows where both invested capital and revenue are available (i.e. not null)
        df['totalRevenueMask']    = df.totalRevenue   .mask(df.investedCapital.isnull())
        df['investedCapitalMask'] = df.investedCapital.mask(df.totalRevenue   .isnull())

        # Sum invested capital and revenue across sector and industry
        sector_stc   = df.loc[:,['totalRevenueMask', 'investedCapitalMask', 'sector'  ]].groupby('sector'  ).agg(['sum'])
        industry_stc = df.loc[:,['totalRevenueMask', 'investedCapitalMask', 'industry']].groupby('industry').agg(['sum'])

        # Use the sums to calculate sales-to-capital ratios
        sector_stc  [('salesToCapitalAggregate','aggregate')] = sector_stc.totalRevenueMask   / sector_stc.investedCapitalMask
        industry_stc[('salesToCapitalAggregate','aggregate')] = industry_stc.totalRevenueMask / industry_stc.investedCapitalMask

        # Join to main aggregates DataFrame
        df_agg_sector   = df_agg_sector  .join(sector_stc, rsuffix='_R')
        df_agg_industry = df_agg_industry.join(industry_stc, rsuffix='_R')
    else:
        print('Must run Company_Info().calculated_fields() method for Invested Capital before aggregate sales-to-capital ratio can be calculated.')
    
    # Record date
    df_agg_sector.index.name   = aggs_date
    df_agg_industry.index.name = aggs_date   

    # Output sector and industry aggregates as csv files
    if write == True:
        df_agg_sector.to_csv('sector_aggs.csv')
        df_agg_industry.to_csv('industry_aggs.csv')
    
    return df_agg_sector, df_agg_industry, aggs_date
