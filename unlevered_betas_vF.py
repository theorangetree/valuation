# Purpose:
    # Calculate unlevered betas and update the main line-by-line market data file with them

import pandas as pd

def unlevered_betas(file_path, tax, force_update=False):
    df = pd.read_csv(file_path, index_col=0)

    update = True
    if force_update != True and 'betaTax' in df.columns:
        if df.betaTax.iloc[0] == tax:
            # Unlevered betas already calculated for current tax rate
            update = False
        
    if update == True:
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
    else:
        print(f'Unlevered beta has already been calculated so {file_path} was not updated. Set force_update=True to force an update.')
    
    return df

# Uncomment to test
#print(unlevered_betas(file_path='market_data_synchronous.csv', tax=0.24))

