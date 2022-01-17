"""Calculate market data statistics (percentile and mean) by sector and industry

Main function:
industry_aggregates() -- Clean market data and return Data Frames with percentile and mean statistics by sector and industry
    # [Optional] Exclude specified ticker from aggregates (e.g. ticker of company being valued)
    # [Optional] If write=True, also outputs statistics to CSV files

Script is meant to be run AFTER calculated_fields.py

Other functions:
filter_companies()    -- Filter out companies from market data
clean_industry_data() -- Recategorize company industries and sectors when sample size is low
percentile()          -- Percentile function usable with Pandas groupby().agg()
"""
import pandas as pd

def filter_companies(df, ex_ticker):
    """Remove tickers and return filtered Data Frame:
        # Remove non-U.S. companies
        # Remove non-primary tickers for companies with multiple share classes and tickers
        # Remove valuation company ticker <ex_ticker> so it is excluded from aggregate statistics
    """
    # Filter for U.S. companies
    df = df[df.country == 'United States']

    # Identify duplicate company names (with same revenue) and tickers to remove
    dup_rows    = df      [df.duplicated      (subset=['shortName','totalRevenue'], keep=False  )].sort_values(by='regularMarketPrice')
    dup_tickers = dup_rows[dup_rows.duplicated(subset=['shortName','totalRevenue'], keep='first')].index.to_list() # Keep ticker with  highest stock price

    # Identify non-primary tickers missed by the duplicate filter
    other_secondary_tickers = ['DISCK'] # List needs to be manually updated

    # Remove tickers
    tickers_to_remove = [ex_ticker] + dup_tickers + other_secondary_tickers
    df = df[~df.index.isin(tickers_to_remove)]

    return df

def clean_industry_data(df):
    """Replace low sample-size industry/sector values with similar industries in market data file"""
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
    """Percentile function usable with Pandas groupby().agg()"""
    def percentile_(x):
        return x.quantile(n)
    percentile_.__name__ = 'percentile_{:2.0f}'.format(n*100)
    return percentile_

def industry_aggregates(file_path, ex_ticker='----', limited=True, write=False):
    """Clean market data and return Data Frames with percentile and mean statistics by sector and industry

    Keyword arguments:
    file_path -- file path to market data CSV file
    ex_ticker -- exclude ticker from aggregate statistics (default placeholder '----')
    limited   -- calculate only columns needed for valuation model (default True)
    write     -- output statistics to CSV files

    This function is meant to be run AFTER calculated_fields.py
    """
    # Load, filter and clean market data
    df = pd.read_csv(file_path, index_col=0)
    aggs_date = df.index.name
    df = filter_companies(df, ex_ticker=ex_ticker)
    df = clean_industry_data(df)

    if limited is True: # Only aggregate data needed for valuation
        col_subset = ['sector','industry','revenueGrowth','operatingMargins']
        if 'unleveredBeta' in df.columns: # Unlevered Beta column already calculated
            col_subset.append('unleveredBeta')
        else:
            print('Must run Company_Info().calculated_fields() method to calculate unlevered betas in market data')

        # Aggregate data
        df_agg_sector   = df.loc[:,col_subset].groupby('sector'  ).agg(['count', 'mean', percentile(0.25), 'median', percentile(0.75)])
        df_agg_industry = df.loc[:,col_subset].groupby('industry').agg(['count', 'mean', percentile(0.25), 'median', percentile(0.75)])
    else:
        df_agg_sector   = df.groupby('sector'  ).agg(['count', 'mean', percentile(0.25), 'median', percentile(0.75)])
        df_agg_industry = df.groupby('industry').agg(['count', 'mean', percentile(0.25), 'median', percentile(0.75)])

    if 'investedCapital' in df.columns: # Invested Capital column already calculated
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
    if write is True:
        df_agg_sector.to_csv('sector_aggs.csv')
        df_agg_industry.to_csv('industry_aggs.csv')

    return df_agg_sector, df_agg_industry, aggs_date

# Uncomment to export market data aggregate statistics to CSV files
#industry_aggregates(file_path='market_data.csv', limited=False, write=True)
