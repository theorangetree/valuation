# Value a company based on its historical financial information as well as historical industry benchmarks
import locale
import time
import datetime
import pandas as pd
import numpy as np
import yahoo_fin.stock_info as si
from yahoo_fin.stock_info import _parse_json
from dateutil.relativedelta import relativedelta
from IPython.display import display
import asyncio
import yf_scraper_asyncio_vF       as scraper
import unlevered_betas_vF          as betas
import industry_data_aggregator_vF as aggregator

#import nest_asyncio
#nest_asyncio.apply()

# Set display options to show more rows and columns than default
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 100)
#locale.setlocale(locale.LC_ALL,'en_US.UTF-8') # change number formatting for locale.atof() to convert strings back into floats

# Identify the stock ticker of interest
#ticker_1 = input("Type a US stock ticker: ")
#ticker_1 = 'FIVN'

#industry = pd.read_csv("2021 Damodaran Data.csv")
#print(industry.columns)

t0 = time.time()
tp0 = time.process_time()

ticker_list = ['FIVN'] # List of companies we want to value



# This asyncio function returns a dictionary with nested dictionaries for each ticker:
# {*ticker*: {*webpage or financial statement*: {key: value}}}
companies = asyncio.run(scraper.company_data(ticker_list))

def ttm_income_statement(is_quarterly):
    # This function organizes necessary TTM (trailing twelve month) income statement items for equity valuation
    
    ttm_output = is_quarterly                 # Quarterly income statements (four latest quarters)
    ttm_date   = ttm_output.columns[0]        # TTM end date
    ttm_output = ttm_output.sum(axis=1)       # Sum four latest quarters to calculate TTM data
    ttm_output = ttm_output.rename(ttm_date)
    
    # Calculate operating margin
    ttm_output.loc['operatingMargin'] = ttm_output.loc['operatingIncome'] / ttm_output.loc['totalRevenue']
    
    # Select and order relevant data fields to output
    ttm_output = ttm_output.reindex(['totalRevenue',
                                     'operatingMargin',
                                     'operatingIncome',
                                     'researchDevelopment',
                                     'interestExpense'])
    # Note: Operating Income in yahoo_fin and google finance removes Contingent Consideration Expense, which is not typical for GAAP
    return ttm_output

def latest_balance_sheet(bs_quarterly):
    # This function organizes necessary latest quarterly balance sheet items for equity valuation
    
    bs_output = bs_quarterly        # Quarterly balance sheets (four latest quarters)
    bs_output = bs_output.iloc[:,0] # Keep only the latest quarter
    
    # Select and order relevant data fields to output
    bs_output = bs_output.reindex(['totalStockholderEquity'])
    return bs_output

def income_statement_trends(is_yearly):
    # This function identifies revenue and operating margin trends over time (mainly for reference)
    
    trends_output = is_yearly # Annual income statements (four latest fiscal years)
   
    # Calculate annual revenue growth and 3-year compounded annual growth rate (CAGR)
    trends_output.loc['totalRevenueShift']  = trends_output.loc['totalRevenue'].shift(-1,axis=0)
    trends_output.loc['revenue1YGrowth']    = trends_output.loc['totalRevenue'] / trends_output.loc['totalRevenueShift'] - 1
    trends_output.loc['totalRevenueShift3'] = trends_output.loc['totalRevenue'].shift(-3,axis=0)
    trends_output.loc['revenue3YCAGR']      = (trends_output.loc['totalRevenue'] / trends_output.loc['totalRevenueShift3']) ** (1/3) - 1
        
    # Calculate operating margin
    trends_output.loc['operatingMargin'] = trends_output.loc['operatingIncome'] / trends_output.loc['totalRevenue']
    
    # Select and order relevant data fields to output
    trends_output = trends_output.reindex(['totalRevenue',
                                           'revenue1YGrowth',
                                           'revenue3YCAGR',
                                           'operatingMargin',
                                           'researchDevelopment'])
    return trends_output

class Company_Info:
    # Stores company specific data
    def __init__(self, ticker: str, market_data_path='market_data_synchronous.csv'):
        self.ticker             = ticker
        self.market_data_path   = market_data_path
        self.income_statement   = ttm_income_statement(companies[ticker]['is_quarterly'])
        self.balance_sheet      = latest_balance_sheet(companies[ticker]['bs_quarterly'])
        self.trend_summary      = income_statement_trends(companies[ticker]['is_yearly'])
        self.quote_summary      = companies[ticker]['quote_summary']
        self.name               = self.quote_summary['shortName']
        self.sector             = self.quote_summary['sector']
        self.industry           = self.quote_summary['industry']
        self.latest_quarter_end = self.income_statement.name
        self.latest_year_end    = self.trend_summary.columns[0]
        self.model_inputs       = {}

        # Add risk-free rate
        self.model_inputs['rf_rate'] = companies['rf_rate']

        # Replace low sample-size sectors/industries with similar ones
        df = pd.DataFrame({'sector'  :[self.sector],
                           'industry':[self.industry]})
        df = aggregator.clean_industry_data(df)
        self.sector   = df.loc[0,'sector']
        self.industry = df.loc[0,'industry']
        print(f'---\nPreparing valuation information for {self.name} ({self.ticker}) (Sector: {self.sector}, Industry: {self.industry})\n---')
        print(f'{self.quote_summary["longBusinessSummary"]}\n')
    
    ### Add betas to the market data .csv file
    
    def tax_rate(self, tax=0.24):
        # Define tax rate
        # Jan-2022 corporate tax rate is 21%, but there is a proposal in congress to increase this to 26.5%
        # As a result, I am assuming a roughly in-between tax rate of 24%
        self.model_inputs['tax_rate'] = tax

    def calculate_betas(self, force_update=False):
        betas.unlevered_betas(file_path=self.market_data_path, tax=self.model_inputs['tax_rate'], force_update=force_update)

    ### Calculate market aggregate statistics (while excluding self from averages)

    def get_market_aggs(self, read=False):
        if read == True:
            # Read sector and industry averages from .csv files
            self.sector_aggs   = pd.read_csv('sector_aggs.csv', index_col=0, header=[0,1])   # header argument recognizes multi-index columns
            self.industry_aggs = pd.read_csv('industry_aggs.csv', index_col=0, header=[0,1])
            aggs_date          = datetime.date.fromisoformat(sector_aggs.index.name)
        else:
            # Aggregate sector and industry averages from line-by-line market data .csv file
            self.sector_aggs, self.industry_aggs, aggs_date = aggregator.industry_aggregates(self.market_data_path, ex_ticker=self.ticker)
            aggs_date = datetime.date.fromisoformat(aggs_date)

        # Calculate days since market data was updated; if too old (e.g. >3 months), recommend rerunning market data webscrape
        today = datetime.date.today()
        print(f'Sector and industry averages loaded as of {aggs_date} ({(today - aggs_date).days} days since last update)')

    ### Calculate DCF assumptions for growth, margins, sales-to-capital ratio, and cost of capital
    
    def growth(self, manual=False, years=5): # years: int >= 1
        # High-growth rate
        if type(manual) == float or type(manual) == int: # Manual input
            self.model_inputs['growth_rate'] = manual
        
        else: # Market-based estimate
            print('Auto-calculating growth_rate...')
            self_TTM = self.quote_summary['revenueGrowth']
            self_3yr = self.trend_summary.loc['revenue3YCAGR',self.latest_year_end]
            sect_TTM = self.sector_aggs.loc[self.sector,('revenueGrowth','median')]     # Use median instead of mean to exclude outliers
            indu_TTM = self.industry_aggs.loc[self.industry,('revenueGrowth','median')]
            
            if pd.isnull(self_3yr):
                growth_rates = [self_TTM, sect_TTM, indu_TTM]
                weights      = [0.65, 0.2, 0.15]
            else:
                growth_rates = [self_TTM, self_3yr, sect_TTM, indu_TTM]
                weights      = [0.45, 0.2, 0.2, 0.15]

            print(f'Growth rates: {growth_rates}')
            self.model_inputs['growth_rate'] = np.average(growth_rates, weights=weights)

        # Years of high-growth
        self.model_inputs['growth_duration'] = years



    def target_margin(self, manual=False, years=False, max_years=10): # years: int >= 1
        # Target operating income margin
        if type(manual) == float or type(manual) == int: # Manual input
            self.model_inputs['target_margin'] = manual

        else: # Market-based estimate
            print('Auto-calculating target_margin...')
            self_TTM = self.income_statement.loc['operatingMargin']
            self_4yr = self.trend_summary.loc['operatingMargin',:].mean() # Yahoo Finance returns up to 4 years of data
            sect_TTM = self.sector_aggs  .loc[self.sector,  ('operatingMargins',['percentile_25','median','percentile_75'])].xs('operatingMargins')
            indu_TTM = self.industry_aggs.loc[self.industry,('operatingMargins',['percentile_25','median','percentile_75'])].xs('operatingMargins')
            
            operating_margins = [self_TTM, self_4yr, sect_TTM.loc['percentile_75'], indu_TTM.loc['percentile_75']] # Use 75th percentile since I'm assuming target margin is aspirational and higher than median
            weights = [0.125, 0.125, 0.375, 0.375]

            print(f'Operating margins: {operating_margins}')
            self.model_inputs['target_margin'] =  np.average(operating_margins, weights=weights)

        # Years to achieve target operating margin
        if type(years) is int:
            self.model_inputs['years_to_target_margin'] = years

        else:        
            # Adjust years to reach target margin based on current margin positioning vs. market percentiles
                # Above 50th %ile     : growth years + 0 years
                # Closer to 50th %ile : growth years + 1 year
                # Closer to 25th %ile : growth years + 2 years
                # Below 25th %ile     : growth years + 3 years

            print('Auto-calculating years to achieve target_margin based on number of growth years and market margins...')
            self_margin    = np.average([self_TTM, self_4yr])
            market_margins = pd.concat([sect_TTM, indu_TTM], axis='columns').mean(axis='columns')
            midpoint_25_50 = market_margins.loc[['percentile_25','median']].mean(axis='index')

            cutoff_0, cutoff_1, cutoff_2 = market_margins.loc['median'], midpoint_25_50, market_margins.loc['percentile_25']
            
            try:
                growth_duration = self.model_inputs['growth_duration']
            except:
                print('Assuming growth_duration and baseline target_margin achievement of 5 years')
                growth_duration = 5

            years = growth_duration
            
            if   self_margin >= cutoff_0:
                None
            elif self_margin >= cutoff_1:
                years += 1
            elif self_margin >= cutoff_2:
                years += 2
            elif self_margin <  cutoff_2:
                years += 3
            else:
                print('Error occurred while calculating target margin achievement year')

            years = min(years, max_years)
            
            self.model_inputs['years_to_target_margin'] = years
    
    def sales_to_capital(self):
        # Calculate sales to capital ratio
        invested_capital = self.balance_sheet['totalStockholderEquity'] + \
                           self.quote_summary['totalDebt'] - \
                           self.quote_summary['totalCash']
        
        self.model_inputs['sales_to_capital'] = self.income_statement['totalRevenue'] / invested_capital
        print('Current sales to capital ratio: {}'.format(self.model_inputs['sales_to_capital']))
    
    def cost_of_capital(self, wacc = False, cod = False, coe = False, erp = False): # Cost of debt is pre-tax
        # Calculate weighted-average cost of capital (WACC)
        if type(wacc) == float:
            self.model_inputs['cost_of_capital'] = wacc

        else:
            print('Auto-calculating weighting average cost of capital')
            market_cap = self.quote_summary['marketCap']
            total_debt = self.quote_summary['totalDebt']
            d_e_ratio  = total_debt / (market_cap + total_debt)
            print(f'Debt-to-equity ratio: {d_e_ratio}')
            
            # Cost of debt
            if type(cod) == float:
                None
            else:
                print('Auto-calculating cost of debt... (I recommend manually inputing cost of debt from sec filings as estimates can be inaccurate)')
                interest_expense   = -self.income_statement.loc['interestExpense']
                operating_income   = self.income_statement.loc['operatingIncome']
                interest_coverage  = interest_expense / operating_income
                effective_int_rate = interest_expense / total_debt
                print(f'Effective interest rate: {effective_int_rate}')

                credit_rating_df = pd.read_csv('synthetic_credit_rating.csv')

                # Small firm interest rate spread (based on interest coverage ratio)
                for i, threshold in enumerate(credit_rating_df.loc[:,'Interest Coverage-Small Firms']):
                    if interest_coverage < threshold:
                        higher_spread = credit_rating_df.loc[:,'Spread (2021)'].iloc[i]
                        break

                # Large firm interest rate spread (based on interest coverage ratio)
                for i, threshold in enumerate(credit_rating_df.loc[:,'Interest Coverage-Large Firms']):
                    if interest_coverage < threshold:
                        lower_spread = credit_rating_df.loc[:,'Spread (2021)'].iloc[i]
                        break

                # Determine if small or large firm based on market cap
                lower = 5000000000
                upper = 10000000000
                diff  = upper - lower
                large_weighting = min(max((market_cap - lower), 0), diff) / diff
                small_weighting = 1 - large_weighting                            

                # Calculate interest rate based on synthetic rating
                synthetic_int_rate = self.model_inputs['rf_rate'] + np.average([higher_spread, lower_spread], weights=[small_weighting, large_weighting])
                print(f'Synthetic interest rate: {synthetic_int_rate}')

                cod = np.average([effective_int_rate, synthetic_int_rate])

            after_tax_cod = cod * (1-self.model_inputs['tax_rate'])
            print(f'After-tax cost of debt: {after_tax_cod}')
                
            # Cost of equity
            if type(coe) == float:
                None
            else:
                print('Auto-calculating cost of equity...')
                # Retrieve implied equity risk premium
                if type(erp) == float:
                    None
                else:
                    erp = companies['implied_erp']

                # Retrieve unlevered beta for sector and industry
                sector_u_beta   = self.sector_aggs  .loc[self.sector,  ('unleveredBeta','mean')] # Mean can be used since outliers have been excluded
                industry_u_beta = self.industry_aggs.loc[self.industry,('unleveredBeta','mean')]

                # Re-lever sector and industry beta and calculate cost of equity
                sector_beta   = sector_u_beta   * (1 + (1-self.model_inputs['tax_rate']) * d_e_ratio)
                industry_beta = industry_u_beta * (1 + (1-self.model_inputs['tax_rate']) * d_e_ratio)
                sector_coe    = self.model_inputs['rf_rate'] + erp * sector_beta
                industry_coe  = self.model_inputs['rf_rate'] + erp * industry_beta

                # Retrieve company's levered beta
                self_beta    = self.quote_summary['beta']

                # Not all companies have enough history for a regression (levered) beta on Yahoo Finance
                if type(self_beta) == float:
                    self_coe = self.model_inputs['rf_rate'] + erp * self_beta
                    coe     = [self_coe, sector_coe, industry_coe]
                    weights = [0.05, 0.475, 0.475] # It's best to mostly rely on industry and sector beta to reduce variance
                else:
                    coe     = [sector_coe, industry_coe]
                    weights = [0.5,0.5]                    

                print(f'Levered betas: {[self_beta, sector_beta, industry_beta]}')
                print(f'Costs of equity: {coe}')

                # Calculate cost of equity
                coe = np.average(coe, weights=weights)

            print(f'Cost of equity: {coe}')

            # Weighted average cost of capital
            wacc = (coe * (1 - d_e_ratio)) + (after_tax_cod * d_e_ratio)
            print(f'Weighted-average cost of capital: {wacc}')
            
        self.model_inputs['cost_of_capital'] = wacc

    def research_development(self):
        None
        # In future, I may decide to add a method to capitalize R&D
        # For now, I've decided not to for consistency with sector/industry data
    
    ### Prepare data for model
    def gather_dcf_data(self):
        # Gather historical company data needed for DCF model
        self.model_inputs['base_date']          = self.latest_quarter_end
        self.model_inputs['base_revenue']       = self.income_statement['totalRevenue']
        self.model_inputs['base_margin']        = self.income_statement['operatingMargin']
        self.model_inputs['debt']               = self.quote_summary['totalDebt']
        self.model_inputs['cash']               = self.quote_summary['totalCash']
        self.model_inputs['shares_outstanding'] = self.quote_summary['sharesOutstanding']
    
    def auto_generate_dcf_assumptions(self):
        # Auto-generate assumptions for growth, margins, sales-to-capital ratio, and cost of capital
        self.growth()
        self.target_margin()
        self.sales_to_capital()
        self.tax_rate()
        self.cost_of_capital()

    def set_manual_dcf_assumptions(prior_NOL = 0, non_operating_assets = 0, minority_interests = 0):
        self.model_inputs['prior_NOL']            = prior_NOL
        self.model_inputs['non_operating_assets'] = non_operating_assets
        self.model_inputs['minority_interests']   = minority_interests
    
    ### Printing methods
    
    
    ### User-guide methods
    def available_individual_data(self):
        # Show all data available to be retreived with retreive_individual_data() method
        print("Here are all 1st level keys and their 2nd level keys (where applicable):")
        first_level_keys = []
        for k1, v1 in self.json_data.items():
            if isinstance(v1, dict):
                second_level_keys = []
                for k2, v2 in v1.items():
                    second_level_keys.append(k2)
                print('\n{}--- {}'.format(k1, ' | '.join(second_level_keys)))
            else:
                first_level_keys.append(k1)
        print('\nAll keys with no second level dictionary: {}'.format(' | '.join(first_level_keys)))

FIVN = Company_Info(ticker_list[0])
import pprint as pp
#pp.pprint(FIVN.__dict__)
FIVN.tax_rate()
FIVN.calculate_betas()
FIVN.get_market_aggs()
FIVN.growth()
FIVN.target_margin()
FIVN.sales_to_capital()
FIVN.cost_of_capital(cod=0.0576)

print(FIVN.model_inputs)

   

def valuation_model(Company):
    # Valuation model
    
    # Company related data and inputs
    ttm_data = ttm_financials(ticker)#.apply(lambda x: locale.atof(x))   # convert numbers in string format back into floats
    bs_data = balance_sheet_items(ticker)#.apply(lambda x: locale.atof(x))   # convert numbers in string format back into floats

    # All model inputs in a dictionary
    """
    inputs = {'model_settings': {'years_before_terminal': 10, # settings specific to the model - typically independent of company
                                 'num_time_periods'     : 12,
                                 },
              'global_params': {'rf_rate'          : risk_free_rate(), # data parameters that is universal for all companies
                                'mature_market_erp': 0.0523
                                },
              'dcf_params': {'high_g'                : 0.3, # data parameters that can be tweaked to match model assumptions
                             'high_g_duration'       : 5,
                             'target_margin'         : 0.2,
                             'years_to_target_margin': 8,
                             'prior_NOL'        : 0, # NOL is not available on Yahoo Finance, as it is an off-balance sheet item
                             'tax_rate'              : 0.25,
                             'sales_to_capital_ratio': 0.75,
                             'growth_wacc'           : 0.07543,
                             'growth_wacc_duration'  : 5,
                             'non_operating_assets'  : 0, # Parameter that probably won't be used
                             'minority_interests'    : 0  # Parameter that probably won't be used
                             },
              'company_data': {'base_year'         : latest_period_end_date(ticker),
                               'base_revenue'      : ttm_data['totalRevenue'],
                               'base_margin'       : ttm_data['operatingMargin'],
                               'debt'              : ind_data['totalDebt'],
                               'cash'              : bs_data['cash'],
                               'shares_outstanding': bs_data['commonStock']
                               'beta'              : ind_data['beta']
                               }
              }
    """
    
    dcf_cols = ['Revenue_Growth','Revenue','Operating_Margin','Operating_Income','Prior_Net_Operating_Loss',
                'Taxable_Operating_Income','Tax_Rate','After_Tax_Operating_Income','Sales_to_Capital',
                'Reinvestment','FCFF','Cost_of_Capital','Discount_Factor','PV(FCFF)']

    years_before_terminal = 10 # Set the number of full years before the terminal year calculation
    num_time_periods = years_before_terminal + 2 # Add two years for the base year and terminal year
    
    dcf_dict = dict()
    # Create empty values for each column
    for line_item in dcf_cols:
        dcf_dict[line_item] = [0] * num_time_periods
     
    # Create a list of dates that represent the period end dates of each year
    base_year = latest_period_end_date(ticker)
    all_years = [] 
    for i in range(num_time_periods):
        all_years.append(base_year + relativedelta(years=i))
    
    dcf_dict['Year_Ended'] = all_years
    
    # Construct the empty DataFrame for the DCF    
    blank_dcf = pd.DataFrame(dcf_dict).set_index('Year_Ended')
    dcf = blank_dcf
    
    
    # Global data
    rf_rate = risk_free_rate()
    
    # Subjective parameters that are tweakable
    
    ### Create the three stages of revenue growth (high, mature, terminal)
    # Inputs
    high_g = 0.3          # high-growth stage growth rate
    high_g_duration = 5   # length of high-growth stage (years)
    mature_g_duration = years_before_terminal - high_g_duration  # length of mature-growth stage
    terminal_g = rf_rate  # terminal growth rate
    
    col_revenue_growth = [0] # base year
    col_revenue_growth += [high_g]*(high_g_duration) # high-growth stage
    col_revenue_growth += np.linspace(high_g, terminal_g, mature_g_duration + 2).tolist()[1:-1] # mature-growth stage (indexing excludes one year of high-growth and one year of terminal growth rate)
    col_revenue_growth += [terminal_g] # terminal growth rate
    
    dcf.Revenue_Growth = col_revenue_growth
    
    ### Create revenue line
    # Inputs
    base_revenue = ttm_data['totalRevenue']
    
    col_revenue = [base_revenue] # base year
    # Calculate revenue through the terminal year
    for i in range(1, num_time_periods):
        col_revenue.append(col_revenue[i-1] * (1 + dcf.Revenue_Growth[i]))  # previous year's revenue multiplied by current year's growth rate
    
    dcf.Revenue = col_revenue
    
    ### Create operating margin line
    # Inputs
    base_margin = ttm_data['operatingMargin']
    target_margin = 0.2
    years_to_target_margin = 8
    
    col_operating_margin = np.linspace(base_margin, target_margin, num=years_to_target_margin + 1).tolist() # base year through first target margin year
    col_operating_margin += [target_margin] * (num_time_periods - years_to_target_margin - 1) # remaing years, including terminal year
    
    dcf.Operating_Margin = col_operating_margin
    
    ### Create operating income line
    # No inputs needed
    dcf.Operating_Income = dcf.Revenue * dcf.Operating_Margin
    
    ### Create net operating loss line
    # Inputs
    prior_NOL = 0
    
    col_prior_NOL = [prior_NOL] # base year
    for i in range(1, num_time_periods):
        if col_prior_NOL[i-1] <= dcf.Operating_Income[i-1]:
            col_prior_NOL.append(0) # cumulative net operating losses all used up
        else:
            col_prior_NOL.append(col_prior_NOL[i-1] - dcf.Operating_Income[i-1]) # carry over prior net operating losses
    
    dcf.Prior_Net_Operating_Loss = col_prior_NOL
    
    ### Create taxable operating income line:
    # No inputs needed
    col_taxable_operating_income = []
    for i in range(num_time_periods):
        if dcf.Prior_Net_Operating_Loss[i] >= dcf.Operating_Income[i]:
            col_taxable_operating_income.append(0) # no taxable income
        else:
            col_taxable_operating_income.append(dcf.Operating_Income[i] - dcf.Prior_Net_Operating_Loss[i])
    
    dcf.Taxable_Operating_Income = col_taxable_operating_income
    
    ### Create tax rate line
    # Inputs
    tax_rate = 0.25 # average effective corporate tax rate is around 25%
    
    dcf.Tax_Rate = tax_rate
    
    ### Create after-tax operating income line
    # No inputs needed
    dcf['After_Tax_Operating_Income'] = dcf.Operating_Income - (dcf.Taxable_Operating_Income * dcf.Tax_Rate)

    ### Create sales to capital ratio line
    # Inputs
    sales_to_capital_ratio = 0.75
    
    dcf.Sales_to_Capital = sales_to_capital_ratio
      
    ### Create reinvestment rate line
    # No inputs needed
    col_reinvestment = [0] # base year
    for i in range(1, num_time_periods):
        col_reinvestment.append((dcf.Revenue[i] - dcf.Revenue[i-1]) / dcf.Sales_to_Capital[i]) # all remaining years
    
    dcf.Reinvestment = col_reinvestment
    
    ### Create free cash flow to firm (FCFF) line
    # No inputs needed  
    dcf.FCFF = dcf['After_Tax_Operating_Income'] - dcf.Reinvestment
    dcf.FCFF.iat[0] = 0 # Set base year FCFF to zero, since those cash flows have already been recognized
    
    ### Create cost of capital line
    # Inputs
    growth_wacc = 0.07543
    growth_wacc_duration = 5 # years
    mature_market_erp = 0.0523
    terminal_cost_of_capital = rf_rate + mature_market_erp
    
    col_cost_of_capital = [0] # base year
    col_cost_of_capital += [growth_wacc] * growth_wacc_duration # growth years
    col_cost_of_capital += np.linspace(growth_wacc,terminal_cost_of_capital, num = years_before_terminal - growth_wacc_duration + 2).tolist()[1:] # maturity through terminal year
    
    dcf.Cost_of_Capital = col_cost_of_capital
    
    ### Create discount factor line
    # No inputs needed
    col_discount_factor = [1]
    for i in range(1, num_time_periods):
        col_discount_factor.append(col_discount_factor[i-1] / (1 + dcf.Cost_of_Capital[i]))
    
    dcf.Discount_Factor = col_discount_factor
    
    ### Create present value of free cash flow line
    # No inputs needed
    dcf['PV(FCFF)'] = dcf.FCFF * dcf.Discount_Factor
    
    
    ### Show the completed dcf (without terminal value)
    print(dcf.transpose())
    
    ############# Finishing the Valuation
    # Calculate cumulative PV(FCFF) for pre-terminal years
    pv_before_terminal = dcf['PV(FCFF)'].iloc[:-1].sum()
    print(f'Present value of {years_before_terminal} years cash flows before terminal year: {pv_before_terminal}')
    # Calcualte PV(terminal value)
    terminal_value = (dcf.iloc[-1].loc['FCFF'] / (dcf.iloc[-1].loc['Cost_of_Capital'] - rf_rate)) 
    pv_terminal_value = terminal_value * dcf.iloc[-2].loc['Discount_Factor'] # Discount terminal value back to present
    print(f'Present value of terminal cash flows in perpetuity: {pv_terminal_value}')
    # Calculate total present value
    total_pv = pv_before_terminal + pv_terminal_value
    print(f'Total present value of cash flows: {total_pv}')
    
    # Inputs
    debt = bs_data['totalDebt']
    cash = bs_data['cash']
    non_operating_assets = 0
    minority_interests = 0
    shares_outstanding = bs_data['commonStock'] * 1000
    # Calculate final estimated value
    equity_value = total_pv + cash - debt + non_operating_assets + minority_interests
    print(f'Total present value of cash flows to shareholders = {equity_value}')
    print(f'Common shares outstanding = {shares_outstanding}')    
    estimate_value_per_share = equity_value / shares_outstanding
    print(f'\nEstimated value per share = ${estimate_value_per_share.round(2)}')
    price_to_value = current_price(ticker) / estimate_value_per_share
    print(f'\nPrice to value ratio: {price_to_value.round(3)}')
    
    return

"""
class DCF_Model:
    # Create DCF models
    def __init__(self, company):
        self.company = company
        None
"""




def latest_period_end_date(ticker):
    # Identify the most recent period end date of filed financial statements
    
    # The get_income_statement method provides the four most recent quarters of income statements, when yearly=False
    quarterly_output = si.get_income_statement(ticker, yearly=False)
    
    # Identify latest quarter end date
    end_date = quarterly_output.columns[0]
    
    return end_date

def valuation_info(ticker):
    
    # Company name
    name = si.get_quote_data(ticker)['shortName']
    
    # Print heading, comapny name and business description
    print("--- Equity valuation information for {} ---\n".format(name))
    print(si.get_company_info(ticker).loc['longBusinessSummary','Value'] + "\n")
    print("{} is in the industry: {}\n---".format(name, si.get_company_info(ticker).loc['industry','Value']))
    
    # Print key current financial statement information
    print(ttm_financials(ticker))
    print(balance_sheet_items(ticker))
    
    # Identify trends in financials
    print('\n---\n')
    print(trends_over_time(ticker))
    
    # Print current price
    current_price(ticker)
    
    return None


### Create invested capital line
    # Inputs
    base_invested_capital = bs_data['totalStockholderEquity'] + bs_data['totalDebt'] - bs_data['cash']

    col_invested_capital = [base_invested_capital] # base year
    for i in range(1, num_time_periods):
        col_invested_capital.append(col_invested_capital[i-1] + dcf.Reinvestment[i])
    print(col_invested_capital)
    
    dcf.Invested_Capital = col_invested_capital




t1 = time.time()
tp1 = time.process_time()

print(f'Normal time: {np.round(t1-t0, 3)}s, Process time: {np.round(tp1-tp0, 3)}s')

