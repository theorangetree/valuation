""" Calculate unlevered betas and sales-to-capital ratio and update in line-by-line market data file

Functions:
unlevered_betas()  -- Calculate unlevered betas and update market data file
sales_to_capital() -- Calculate sales-to-capital ratios and update market data file
"""
import pandas as pd

def unlevered_betas(file_path, tax, force_update=False):
    """Calculate unlevered betas and update market data file

    Keyword arguments:
    file_path    -- file path to market data file
    tax          -- marginal tax rate
    force_update -- recalculate unlevered beta even if already in file
    """
    # Load market data
    df = pd.read_csv(file_path, index_col=0)

    update = True
    if force_update is not True and 'betaTax' in df.columns:
        if df.betaTax.iloc[0] == tax:
            # Unlevered betas already calculated for current tax rate
            update = False

    if update is True:
        # Calculate unlevered betas
        df['marketDebtToEquity'] = df.totalDebt / df.marketCap
        df['betaTax']            = tax
        df['unleveredBeta']      = df.beta / (1 + ((1-df.betaTax) * df.marketDebtToEquity))

        # Exclude outliers; i.e. remove the top x/2 % and bottom x/2 % from unlevered beta
        percent_excluded = 0.02
        low_cutoff  = df.unleveredBeta.quantile(percent_excluded/2)
        high_cutoff = df.unleveredBeta.quantile(1 - percent_excluded/2)
        df['unleveredBeta'] = df.unleveredBeta.where((df.unleveredBeta >= low_cutoff) & (df.unleveredBeta <= high_cutoff))
        print(f'Unlevered betas calculated assuming tax rate of {tax}')
        print(f'Top and bottom {percent_excluded*100/2:.1f}% of unlevered betas excluded as outliers (cutoffs: [{low_cutoff:.3f}, {high_cutoff:.3f}])')

        df.to_csv(file_path)
        print(f'Updated {file_path} with unlevered betas')
    else:
        print(f'Unlevered betas have already been calculated so {file_path} was not updated. Set force_update=True to force an update.')

    return df

def sales_to_capital(file_path, force_update=False):
    """Calculate sales-to-capital ratios and update market data file

    Keyword arguments:
    file_path    -- file path to market data CSV file
    force_update -- recalculate sales-to-capital ratios even if already in file
    """
    # Load market data
    df = pd.read_csv(file_path, index_col=0)

    update = True
    if force_update is not True and 'salesToCapitalIndividual' in df.columns:
        update = False

    if update is True:
        # Calculate sales-to-capital ratios
        df['bookEquity']               = (df.totalDebt / df.debtToEquity) * 100
        df['investedCapital']          = df.bookEquity + df.totalDebt - df.totalCash
        df['salesToCapitalIndividual'] = df.totalRevenue / df.investedCapital

        df.to_csv(file_path)
        print(f'Updated {file_path} for sales-to-capital ratio')
    else:
        print(f'Sales-to-capital ratios have already been calculated so {file_path} was not updated. Set force_update=True to force an update.')

    return df

# Uncomment to test
#print(unlevered_betas(file_path='market_data_synchronous.csv', tax=0.24))
#print(sales_to_capital(file_path='market_data_synchronous.csv'))
