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
import yf_scraper_asyncio_vF as scraper
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

def get_market_aggs(read=False, ex_ticker="----"):
    if read == True:
        # Read sector and industry averages from .csv files
        sector_aggs   = pd.read_csv('sector_aggs.csv', index_col=0, header=[0,1]) # header argument helps recognize multi-index column names
        industry_aggs = pd.read_csv('industry_aggs.csv', index_col=0, header=[0,1])
        aggs_date = datetime.date.fromisoformat(sector_aggs.index.name)
    else:
        # Aggregate sector and industry averages from line-by-line stock data .csv file
        sector_aggs, industry_aggs, aggs_date = aggregator.industry_aggregates('market_data.csv', ex_ticker=ex_ticker)
        aggs_date = datetime.date.fromisoformat(aggs_date)

    # Calculate days since market data was updated; if too old (e.g. >3 months), recommend rerunning market data webscrape
    today = datetime.date.today()
    print(f'Sector and industry averages loaded as of {aggs_date} ({(today - aggs_date).days} days since last update)')

    return sector_aggs, industry_aggs, aggs_date

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
    bs_output = bs_output.reindex(['totalStockholderEquity',
                                   'cash',
                                   'commonStock'])
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
    def __init__(self, ticker: str, rnd_adjustment = True):
        self.ticker             = ticker
        self.income_statement   = ttm_income_statement(companies[ticker]['is_quarterly'])
        self.balance_sheet      = latest_balance_sheet(companies[ticker]['bs_quarterly'])
        self.trend_summary      = income_statement_trends(companies[ticker]['is_yearly'])
        self.quote_summary      = companies[ticker]['quote_summary']
        self.name               = self.quote_summary['shortName']
        self.sector             = self.quote_summary['sector']
        self.industry           = self.quote_summary['industry']
        self.latest_quarter_end = self.income_statement.name
        self.latest_year_end    = self.trend_summary.columns[0]
        self.rnd_adjustment     = rnd_adjustment
        self.model_inputs       = {}

        # Replace low sample-size sectors/industries with similar ones
        df = pd.DataFrame({'sector'  :[self.sector],
                           'industry':[self.industry]})
        df = aggregator.clean_industry_data(df)
        self.sector   = df.loc[0,'sector']
        self.industry = df.loc[0,'industry']

        # Calculate market aggregate statistics while excluding self-data
        self.sector_aggs, self.industry_aggs, aggs_date = get_market_aggs(ex_ticker=self.ticker)
    
    ### Calculate DCF assumptions for growth, margins, sales-to-capital ratio, and cost of capital
    
    def growth(self, manual=False, duration=5): # duration: int >= 1
        
        if manual != False: # Manual input growth
            self.model_inputs['growth_rate'] = manual
        
        else: # Auto estimate growth
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

        self.model_inputs['growth_duration'] = duration

    def margin(self):
        self.model_inputs['target_margin'] = np.mean(industry_data.loc[self.industry,'operatingMargins'],
                                                     industry_data.loc[self.sector,'operatingMargins'])
    
    def sales_to_capital(self):
        # Calculate sales to capital ratio
        invested_capital = self.balance_sheet['totalStockholderEquity'] + \
                           self.quote_summary['totalDebt'] - \
                           self.balance_sheet['cash']
        
        self.model_inputs['sales_to_capital'] = self.income_statement['totalRevenue'] / invested_capital
    
    def beta(self):
        None

    def cost_of_capital(self):
        market_cost_of_capital = companies['rf_rate'] # + market ERP
        self.model_inputs['cost_of_capital'] = np.mean(industry_data.loc[self.industry,'cost_of_capital'])

    def research_development(self):
        self.model_inputs['rnd_amortization_years'] = 3

        if self.rnd_adjustment == True:
            rnd         = [self.income_statement.loc['researchDevelopment']]+ self.trend_summary.loc['researchDevelopment'].to_list()
            weights     = [1]
            years       = self.model_inputs['rnd_amortization_years']
            time_offset = (self.latest_quarter_end.month - self.latest_year_end.month)/12 # figure out how to do months
            if len(rnd) <= years:
                rnd.append([rnd[-1]*(years-len(rnd)+1)])

            for i in range(years):
                if i == 0:
                    weights.append(time_offset)
                elif i == years - 1:
                    weights.append(1 - time_offset)
                else:
                    weights.append(1)

            new_rnd_expense  = rnd * weights * 1/years
            rnd_differential = None
    
    ### Prepare data for model
    def company_dcf_data(self):
        # Gather historical company data needed for DCF model
        None
    
    def company_dcf_assumptions(self):
        # Gather assumptions for growth, margins, sales-to-capital ratio, and cost of capital
        None
    
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
pp.pprint   (FIVN.__dict__)
FIVN.growth()
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
                             'base_prior_NOL'        : 0, # NOL is not available on Yahoo Finance, as it is an off-balance sheet item
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
    base_prior_NOL = 0
    
    col_prior_NOL = [base_prior_NOL] # base year
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

