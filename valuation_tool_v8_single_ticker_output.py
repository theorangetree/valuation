# Value a company based on its historical financial information as well as historical industry benchmarks
import locale
import time
import datetime
import pandas as pd
import numpy as np
import yahoo_fin.stock_info as si
from dateutil import relativedelta
from IPython.display import display
import asyncio
import yf_scraper_asyncio_vF       as scraper
import calculated_fields_vF        as fields
import industry_data_aggregator_vF as aggregator

# Set display options to show more rows and columns than default
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 100)
#locale.setlocale(locale.LC_ALL,'en_US.UTF-8') # change number formatting for locale.atof() to convert strings back into floats

# Identify the stock ticker of interest
#tickers = input("Type a US stock ticker/s (comma-separated): ")
#ticker_list = [tickers]

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
        self.fields_calculated  = False
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
    
    ### Add calculated fields to the market data .csv file
    
    def tax_rate(self, tax=0.24):
        # Define tax rate for calculating unlevered beta
        # Jan-2022 corporate tax rate is 21%, but there is a proposal in congress to increase this to 26.5%
        # As a result, I am assuming a roughly in-between tax rate of 24%
        self.model_inputs['tax_rate'] = tax

    def calculated_fields(self, force_update=False):
        # Calculate unlevered betas and market debt-to-equity
        fields.unlevered_betas(file_path=self.market_data_path, tax=self.model_inputs['tax_rate'], force_update=force_update)
        
        # Calculate sales-to-capital ratios and invested capital
        fields.sales_to_capital(file_path=self.market_data_path, force_update=force_update)

        self.fields_calculated = True

    ### Calculate market aggregate statistics by sector and industry (while excluding self from averages)

    def get_market_aggs(self, read=False):
        if read == True:
            # Read sector and industry averages from .csv files
            self.sector_aggs   = pd.read_csv('sector_aggs.csv', index_col=0, header=[0,1])   # header argument recognizes multi-index columns
            self.industry_aggs = pd.read_csv('industry_aggs.csv', index_col=0, header=[0,1])
            aggs_date          = datetime.date.fromisoformat(sector_aggs.index.name)
        else:
            # Aggregate sector and industry averages from line-by-line market data .csv file
            self.sector_aggs, self.industry_aggs, aggs_date = aggregator.industry_aggregates(self.market_data_path, ex_ticker=self.ticker, sales_to_capital=self.fields_calculated)
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
            sect_TTM = self.sector_aggs  .loc[self.sector  ,('revenueGrowth',['percentile_25','median','percentile_75'])].xs('revenueGrowth')
            indu_TTM = self.industry_aggs.loc[self.industry,('revenueGrowth',['percentile_25','median','percentile_75'])].xs('revenueGrowth')
            print(f'Growth rates: {[self_TTM, self_3yr, sect_TTM.to_list(), indu_TTM.to_list()]}')

            # Average of self and market growth rates
            if pd.isnull(self_3yr):
                self_growth = self_TTM
            else:
                self_growth = np.average([self_TTM, self_3yr], weights=[0.7, 0.3]) # Weighted average
            market_growth = pd.concat([sect_TTM, indu_TTM], axis='columns').mean(axis='columns')

            # Weight higher value of self or market more heavily
            if   self_growth >= market_growth.loc['percentile_75']:
                weights = [0.7, 0.3]
            elif self_growth >= market_growth.loc['median']:
                weights = [0.7, 0.3]
            elif self_growth <  market_growth.loc['median']:
                weights = [0.5, 0.5]

            self.model_inputs['growth_rate'] = np.average([self_growth, market_growth.loc['median']], weights=weights)

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

            # Average self and market margins
            self_margin    = np.average([self_TTM, self_4yr])
            market_margins = pd.concat([sect_TTM, indu_TTM], axis='columns').mean(axis='columns')

            # Weight higher value out of self margin vs. market margin more heavily
            if self_margin >= market_margins.loc['percentile_75']:
                weights = [0.3, 0.3, 0.2, 0.2]
            elif self_margin < market_margins.loc['percentile_75']:
                weights = [0, 0, 0.4, 0.6]

            print(f'Operating margins: {[self_TTM, self_4yr, sect_TTM.to_list(), indu_TTM.to_list()]}')
            self.model_inputs['target_margin'] = max([np.average(operating_margins, weights=weights), market_margins.loc['median']])

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
            midpoint_50_75 = market_margins.loc[['median','percentile_75']].mean(axis='index')
            midpoint_25_50 = market_margins.loc[['percentile_25','median']].mean(axis='index')

            cutoff_0, cutoff_1, cutoff_2, cutoff_3 = midpoint_50_75, market_margins.loc['median'], midpoint_25_50, market_margins.loc['percentile_25']
            
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
            elif self_margin >= cutoff_2:
                years += 3
            elif self_margin <  cutoff_3:
                years += 4
            else:
                print('Error occurred while calculating target margin achievement year')

            years = min(years, max_years)
            
            self.model_inputs['years_to_target_margin'] = years
            print(f'Years to achieve target margin: {years}') 
    
    def sales_to_capital(self, manual=False):
        # Sales to invested capital ratio
        if type(manual) == float or type(manual) == int:
            self.model_inputs['sales_to_capital'] = manual

        else: # Market-based estimate
            print('Auto-calculating sales-to-capital ratio...')
            # Calculate current invested capital and sales-to-capital ratio
            self_inv_cap = self.balance_sheet['totalStockholderEquity'] + \
                           self.quote_summary['totalDebt'] - \
                           self.quote_summary['totalCash']
            self_stc     = self.income_statement['totalRevenue'] / self_inv_cap

            # Retrieve market sales-to-capital ratioe
            sector_stc   = self.sector_aggs  .loc[self.sector  ,('salesToCapitalAggregate','aggregate')]
            industry_stc = self.industry_aggs.loc[self.industry,('salesToCapitalAggregate','aggregate')]

            sales_to_capital_ratios = [self_stc, sector_stc, industry_stc]
            print(f'Sales-to-capital ratios: {sales_to_capital_ratios}')
            
            if self.quote_summary['totalCash'] >= self_inv_cap:
                print('Warning: review current sales-to-capital ratio as a large portion of cash was subtracted from invested capital')
                weights = [0.175, 0.55, 0.275]
            else:
                weights = [0.4, 0.4, 0.2]

            # Drop negative ratios from the list
            clean_ratios  = []
            clean_weights = []
            for ratio, weight in zip(sales_to_capital_ratios, weights):
                if ratio > 0:
                    clean_ratios .append(ratio)
                    clean_weights.append(weight)
            self.model_inputs['sales_to_capital'] = np.average(clean_ratios, weights=clean_weights)
    
    def cost_of_capital(self, wacc = False, cod = False, coe = False, erp = False, mature_wacc = False): # Cost of debt is pre-tax
        ### Weighted-average cost of capital (WACC)
        if type(wacc) == float:
            self.model_inputs['cost_of_capital'] = wacc

        else:
            print('Auto-calculating weighted-average cost of capital')
            market_cap = self.quote_summary['marketCap']
            total_debt = self.quote_summary['totalDebt']
            d_e_ratio  = total_debt / market_cap
            debt_ratio = total_debt / (market_cap + total_debt)
            print(f'Market debt-to-equity ratio: {d_e_ratio}')
            
            ## Cost of debt
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
                
            ## Cost of equity
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

                # Not all companies have a regression (levered) beta on Yahoo Finance
                if type(self_beta) == float:
                    self_coe = self.model_inputs['rf_rate'] + erp * self_beta
                    coe     = [self_coe, sector_coe, industry_coe]
                    weights = [0.05, 0.475, 0.475] # It's best to rely on sector and industry beta
                else:
                    coe     = [sector_coe, industry_coe]
                    weights = [0.5,0.5]                    

                print(f'Levered betas: {[self_beta, sector_beta, industry_beta]}')
                print(f'Costs of equity: {coe}')

                # Calculate cost of equity
                coe = np.average(coe, weights=weights)

            print(f'Cost of equity: {coe}')

            # Weighted average cost of capital
            wacc = (coe * (1 - debt_ratio)) + (after_tax_cod * debt_ratio)
            self.model_inputs['cost_of_capital'] = wacc

        print(f'Weighted-average cost of capital: {wacc}')

        ### Mature (i.e. terminal) WACC
        if type(mature_wacc) == float:
            self.model_inputs['mature_wacc'] = mature_wacc
        else:
            print('Auto-estimating mature WACC...')
            mature_wacc = self.model_inputs['rf_rate'] + 0.045 # This is an approximation of market average cost of capital, which mature companies tend to approach
            self.model_inputs['mature_wacc'] = min([wacc, np.average([wacc, mature_wacc])])
        print(f'Mature WACC: {self.model_inputs["mature_wacc"]}')

    def research_development(self):
        None
        # In future, I may decide to add a method to capitalize R&D
        # For now, I've decided not to for consistency with sector/industry data
    
    ### Prepare data for model
    def prepare_market_data(self, tax): # Recommend marginal tax rate (i.e. corporate tax rate)
        self.tax_rate(tax=tax)
        self.calculated_fields()
        self.get_market_aggs()
    
    def auto_generate_dcf_assumptions(self, growth=False, growth_years=5, margin=False, margin_years=False, sales_to_capital=False,
                                      wacc=False, cod=False, coe=False, erp=False, mature_wacc=False):
        # Auto-generate assumptions for growth, margins, sales-to-capital ratio, and cost of capital
        self.growth          (manual = growth, years = growth_years)
        self.target_margin   (manual = margin, years = margin_years)
        self.sales_to_capital(manual = sales_to_capital)
        self.cost_of_capital (wacc = wacc, cod = cod, coe = coe, erp = erp, mature_wacc = mature_wacc)
        
    def gather_dcf_data(self):
        # Gather historical company data needed for DCF model
        self.model_inputs['name']               = self.name
        self.model_inputs['ticker']             = self.ticker
        self.model_inputs['base_date']          = self.latest_quarter_end.date()
        self.model_inputs['base_growth']        = self.quote_summary['revenueGrowth']
        self.model_inputs['base_revenue']       = self.income_statement['totalRevenue']
        self.model_inputs['base_margin']        = self.income_statement['operatingMargin']
        self.model_inputs['debt']               = self.quote_summary['totalDebt']
        self.model_inputs['cash']               = self.quote_summary['totalCash']
        self.model_inputs['shares_outstanding'] = self.quote_summary['sharesOutstanding']
        self.model_inputs['current_price']      = self.quote_summary['regularMarketPrice']

    def set_manual_dcf_data(self, prior_NOL = 0, non_operating_assets = 0, minority_interests = 0):
        self.model_inputs['prior_NOL']            = prior_NOL
        self.model_inputs['non_operating_assets'] = non_operating_assets
        self.model_inputs['minority_interests']   = minority_interests

############# --- End of Company Class --- #############





def valuation_model(model_inputs: dict, terminal_year = 11, write=False):
    # Discounted Cash Flow (DCF) valuation model to value a single stock
    
    ### Construct and empty data frame of size (no. of time periods x no. of DCF items)
    dcf_dict = {}

    # Default is 12 time periods: i.e. Base Year + 10 Years + Terminal Year   
    num_time_periods = terminal_year + 1
    # Create a list of dates containing the end date of each time period (year)
    year_ends = []
    for i in range(num_time_periods):
        year_ends.append(model_inputs['base_date'] + relativedelta.relativedelta(years=i))
    year_ends[0]  = year_ends[0].strftime('%Y-%m-%d')  + ' (Base)'
    year_ends[-1] = year_ends[-1].strftime('%Y-%m-%d') + ' (Terminal)' 

    
    dcf_dict['Year_Ended'] = year_ends

    # Add empty placeholder values for each DCF item
    dcf_items = ['Revenue_Growth','Revenue','Operating_Margin','Operating_Income','Prior_Net_Operating_Loss',
                 'Taxable_Operating_Income','Tax_Rate','After_Tax_Operating_Income','Sales_to_Capital',
                 'Reinvestment','FCFF','Cost_of_Capital','Discount_Factor','PV(FCFF)']
    for item in dcf_items:
        dcf_dict[item] = [0] * num_time_periods

    # Construct placeholder DataFrame
    dcf = pd.DataFrame(dcf_dict).set_index('Year_Ended')
   
    ### Create the three stages of revenue growth (high-growth, mature-growth, terminal growth)
    # Inputs
    high_g            = model_inputs['growth_rate']         # High-growth stage growth rate
    high_g_duration   = model_inputs['growth_duration']     # Length of high-growth stage (years)
    mature_g_duration = terminal_year - high_g_duration - 1 # Length of mature-growth stage
    terminal_g        = model_inputs['rf_rate']             # Terminal growth rate

    # Revenue growth
    col_revenue_growth =  [model_inputs['base_growth']] # Base year (for reference only)
    col_revenue_growth += [high_g]*(high_g_duration) # High-growth stage
    col_revenue_growth += np.linspace(high_g, terminal_g, num = mature_g_duration + 2).tolist()[1:-1] # Mature-growth stage (indexing excludes one year of high-growth and one year of terminal growth rate)
    col_revenue_growth += [terminal_g] # Terminal growth rate
    
    dcf.Revenue_Growth = col_revenue_growth

    ### Create revenue line
    # Base year
    col_revenue = [model_inputs['base_revenue']]
    # Calculate revenue through the terminal year using revenue growth rates
    for i in range(1, num_time_periods):
        col_revenue.append(col_revenue[i-1] * (1 + dcf.Revenue_Growth[i]))
    
    dcf.Revenue = col_revenue
    
    ### Create operating margin line
    # Inputs
    base_margin            = model_inputs['base_margin']
    target_margin          = model_inputs['target_margin']
    years_to_target_margin = model_inputs['years_to_target_margin']

    # Operating margins
    col_operating_margin = np.linspace(base_margin, target_margin, num = years_to_target_margin + 1).tolist() # Base year through first target margin year
    col_operating_margin += [target_margin] * (num_time_periods - years_to_target_margin - 1) # Remaing years, including terminal year
    
    dcf.Operating_Margin = col_operating_margin
    
    ### Create operating income line
    # No inputs needed
    dcf.Operating_Income = dcf.Revenue * dcf.Operating_Margin
    
    ### Create net operating loss line
    # Base year
    col_prior_NOL = [model_inputs['prior_NOL']]
    # Calculate net operating loss through the terminal year
    for i in range(1, num_time_periods):
        if col_prior_NOL[i-1] <= dcf.Operating_Income[i-1]:
            col_prior_NOL.append(0) # cumulative net operating losses all used up
        else:
            col_prior_NOL.append(col_prior_NOL[i-1] - dcf.Operating_Income[i-1]) # carry over net operating losses
    
    dcf.Prior_Net_Operating_Loss = col_prior_NOL
    
    ### Create taxable operating income line:
    # No inputs needed
    col_taxable_operating_income = []
    # Calculate taxable operating income through the terminal year
    for i in range(num_time_periods):
        if dcf.Prior_Net_Operating_Loss[i] >= dcf.Operating_Income[i]:
            col_taxable_operating_income.append(0) # No taxable income
        else:
            col_taxable_operating_income.append(dcf.Operating_Income[i] - dcf.Prior_Net_Operating_Loss[i])
    
    dcf.Taxable_Operating_Income = col_taxable_operating_income
    
    ### Create tax rate line
    dcf.Tax_Rate = model_inputs['tax_rate']
    
    ### Calculate after-tax operating income line
    dcf['After_Tax_Operating_Income'] = dcf.Operating_Income - (dcf.Taxable_Operating_Income * dcf.Tax_Rate)

    ### Create sales-to-capital ratio line
    dcf.Sales_to_Capital = model_inputs['sales_to_capital']
      
    ### Create reinvestment line
    # No inputs needed
    col_reinvestment = [0] # Base year
    # Calculate capital reinvestment amounts (needed for growth) through the terminal year
    for i in range(1, num_time_periods):
        col_reinvestment.append((dcf.Revenue[i] - dcf.Revenue[i-1]) / dcf.Sales_to_Capital[i])
    
    dcf.Reinvestment = col_reinvestment
    
    ### Create free cash flow to firm (FCFF) line
    dcf.FCFF = dcf['After_Tax_Operating_Income'] - dcf.Reinvestment
    dcf.FCFF.iat[0] = 0 # Set base year FCFF to zero, since those cash flows have already been recognized
    
    ### Create cost of capital line
    # Inputs
    growth_wacc              = model_inputs['cost_of_capital']
    growth_wacc_duration     = model_inputs['growth_duration']
    terminal_cost_of_capital = model_inputs['mature_wacc']

    # Weighted-average cost of capital
    col_cost_of_capital =  [0] # Base year
    col_cost_of_capital += [growth_wacc] * growth_wacc_duration # Growth years
    col_cost_of_capital += np.linspace(growth_wacc, terminal_cost_of_capital, num = terminal_year - growth_wacc_duration + 1).tolist()[1:] # Mature years through terminal year. Indexing excludes one year of growth-stage WACC.
    
    dcf.Cost_of_Capital = col_cost_of_capital
    
    ### Create discount factor line
    # No inputs needed
    col_discount_factor = [1] # Base year
    # Calculate discount factor through the terminal year using WACC
    for i in range(1, num_time_periods):
        col_discount_factor.append(col_discount_factor[i-1] / (1 + dcf.Cost_of_Capital[i]))
    
    dcf.Discount_Factor         = col_discount_factor
    dcf.Discount_Factor.iat[-1] = np.nan # Terminal year discount factor is not relevant to terminal value calculation
    
    ### Create present value of free cash flow line
    dcf['PV(FCFF)']         = dcf.FCFF * dcf.Discount_Factor
    dcf['PV(FCFF)'].iat[-1] = np.nan # Terminal year cash flow is included in terminal value calculation

    
    ############# Calculate Terminal Value and Finish the Valuation #############
    # Calculate cumulative PV(FCFF) for pre-terminal years
    pv_before_terminal = dcf['PV(FCFF)'].iloc[:-1].sum()
    print(f'\nPresent value of {terminal_year-1} years of cash flows before terminal year: {pv_before_terminal}')
    # Calcualte PV(terminal value)
    terminal_value    = dcf.iloc[-1].loc['FCFF'] / (dcf.iloc[-1].loc['Cost_of_Capital'] - model_inputs['rf_rate'])
    pv_terminal_value = terminal_value * dcf.iloc[-2].loc['Discount_Factor'] # Discount terminal value back to present
    print(f'Present value of terminal cash flows in perpetuity: {pv_terminal_value}')
    # Calculate total present value of operations
    total_pv = pv_before_terminal + pv_terminal_value
    print(f'Total present value of cash flows from operations: {total_pv}')
    
    # Inputs for final valuation
    debt                 = model_inputs['debt']
    cash                 = model_inputs['cash']
    non_operating_assets = model_inputs['non_operating_assets']
    minority_interests   = model_inputs['minority_interests']
    shares_outstanding   = model_inputs['shares_outstanding']
    current_price        = model_inputs['current_price']
    # Calculate final estimated value
    equity_value    = total_pv + cash - debt + non_operating_assets + minority_interests
    value_per_share = equity_value / shares_outstanding
    value_to_price  = value_per_share / current_price
    
    print(f'Total present value of cash flows to shareholders = {equity_value}')
    print(f'Common shares outstanding = {shares_outstanding}')    
    print(f'\nEstimated value per share = ${value_per_share:.2f}\n---')
    print(f'\nCurrent price per share = ${current_price:.2f}\n---')
    print(f'\nPrice to value ratio: {value_to_price:.3f}\n---')

    ############# Output completed DCF and terminal value #############
    if write == True:
        dcf_output = dcf.transpose()
        dcf_output.index.set_names(f'{model_inputs["name"]} ({model_inputs["ticker"]})', inplace=True)

        final_dict      = {'PV_DCF_Cash_Flows'      : pv_before_terminal,
                           'PV_Terminal_Value'      : pv_terminal_value,
                           'Total_Present_Value'    : total_pv,
                           '- Debt'                 : debt,
                           '- Minority_Interests'   : minority_interests,
                           '+ Cash'                 : cash,
                           '+ Non_Operating_Assets' : non_operating_assets,
                           '---'                    : np.nan,
                           'Value_of_Equity'        : equity_value,
                           'Shares_Outstanding'     : shares_outstanding,
                           'Value_per_Share'        : value_per_share,
                           'Current_Stock_Price'    : current_price,
                           'Value_to_Price_Ratio'   : value_to_price
                           }
        final_output         = pd.DataFrame.from_dict(final_dict, orient='index')
        final_output         = pd.concat([dcf_output.iloc[:,-1], final_output], axis = 'rows')
        final_output.columns = [dcf_output.columns[-1]]
        final_output.index.set_names(f'{model_inputs["name"]} ({model_inputs["ticker"]})', inplace=True)

        def format_excel_rows(worksheet: object, row_indices: list, cell_format: object):
            for row in row_indices:
                worksheet.set_row(row, None, cell_format)
        
        with pd.ExcelWriter(f'DCF valuation.xlsx') as writer:
            dcf_output  .to_excel(writer, sheet_name='DCF')
            final_output.to_excel(writer, sheet_name='Final Value')

            # Assign workbook and worksheet objects to variables
            workbook          = writer.book
            dcf_sheet         = writer.sheets['DCF']
            final_value_sheet = writer.sheets['Final Value']
            
            # Add Excel format types to workbook
            number_format  = workbook.add_format({'num_format': '#,##0'})
            percent_format = workbook.add_format({'num_format': '0.00%'})
            dollar_format  = workbook.add_format({'num_format': '$#,##0.00'})

            # Set row formats for DCF sheet
            number_rows  = [2,4,5,6,8,10,11,14]
            percent_rows = [1,3,7,9,12,13]
            format_excel_rows(worksheet = dcf_sheet, row_indices = number_rows , cell_format = number_format)
            format_excel_rows(worksheet = dcf_sheet, row_indices = percent_rows, cell_format = percent_format)
            
            # Set row formats for Final Value sheet
            number_rows  = [2,4,5,6,8,10,11,14,16,17,18,19,20,21,22,23,24]
            percent_rows = [1,3,7,9,12,13,27]
            dollar_rows  = [25,26]
            format_excel_rows(worksheet = final_value_sheet, row_indices = number_rows , cell_format = number_format)
            format_excel_rows(worksheet = final_value_sheet, row_indices = percent_rows, cell_format = percent_format)
            format_excel_rows(worksheet = final_value_sheet, row_indices = dollar_rows , cell_format = dollar_format)
            
            # Set column formats for DCF sheet
            dcf_sheet.set_column('A:A',27)
            dcf_sheet.set_column('B:B',16)
            dcf_sheet.set_column('C:L',14)
            dcf_sheet.set_column('M:M',20)
            
            # Set column formats for Final Value sheet
            final_value_sheet.set_column('A:A',27)
            final_value_sheet.set_column('B:B',20)
            
    return


FIVN = Company_Info(ticker_list[0])
import pprint as pp
#pp.pprint(FIVN.__dict__)
FIVN.prepare_market_data(tax=0.24)
FIVN.auto_generate_dcf_assumptions()
FIVN.gather_dcf_data()
FIVN.set_manual_dcf_data(prior_NOL=0)


print(FIVN.model_inputs)

valuation_model(FIVN.model_inputs, write=True)


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
