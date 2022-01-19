""" Value companies based on historical financials and industry benchmarks, as well as manually set assumptions

!! This is the main script !!

Usage Guideline:
    # Update TICKER_LIST to contain a python list of tickers for companies you want to value
    # (Optional) Update ASSUMPTION_LIST by adding dictionaries to the list
    # Run script -> review Excel output -> adjust assumptions -> re-run script
    # If market data is older than 3 months, consider running yf_market_data_scraper.py to update it (runtime ~2.5hrs)

Main function:
main() -- Export an Excel workbook containing a DCF valuation in each worksheet per ticker in TICKER_LIST

Class and main methods:
CompanyInfo()               -- Store valuation-related data
    .prepare_model_inputs() -- Generate model inputs and assumptions automatically or set based on arguments passed
    .valuation_model()      -- Create DCF valuation model using the model inputs above
Outputs tuple containing: dcf (data frame), final valuation (data frame), ticker, name, value-to-price ratio, model inputs (dictionary)

Other class methods:
.tax_rate()          -- Set tax rate assumption
.calculated_fields() -- Run calculated_fields.py to calculate unlevered beta and sales-to-capital ratios for market data
.get_market_aggs()   -- Run industry_aggregator.py to aggregate market data by sector and industry
.growth()            -- Set or auto-estimate growth rate and number of growth years
.target_margin()     -- Set or auto-estimate target margin and years to achieve target margin
.sales_to_capital()  -- Set or auto-estimate future sales-to-capital ratio
.cost_of_capital()   -- Set or auto-estimate cost of capital (same for its components: cost of debt and cost of equity)

Other functions:
ttm_income_statement    -- Clean and return latest quarterly income statement items for valuation model inputs
latest_balance_sheet    -- Clean and return latest quarterly balance sheet items for valuation model
income_statement_trends -- Calculate and return revenue and operating margin numbers and changes over recent years
"""
import time
import datetime
import asyncio
import pandas as pd
import numpy as np
from dateutil import relativedelta
# Local imports
import yf_scraper          as scraper
import calculated_fields   as fields
import industry_aggregator as aggregator
import excel_formatter

t0, tp0 = time.time(), time.process_time()

# Set display options to show more rows and columns than default
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 100)

# Identify the stock ticker/s of interest
TICKER_LIST = ['ASAN','CRM','GOOG','WRBY'] # List of companies to value

# USER-INPUT ASSUMPTIONS (OPTIONAL)
# Use a list of dictionaries, where first dictionary contains assumptions for first ticker and so on...
ASSUMPTIONS_LIST = [{}]

""" Here areall available keyword arguments for the assumptions
tax                  -- Marginal tax rate if the company is profitable
terminal_year        -- Final year of model when the company is mature and growing on pace with the economy
growth_years         -- Number of years of high growth without any decline in growth rate
prior_NOL            -- Carry forward "Net Operating Loss" that can be used as future tax offset
non_operating_assets -- Value of non-operating assets (non-cash assets with no impact on business operations)
minority_interests   -- Value of minority interests in the company being valued, to be subtracted from valuation
growth               -- High-growth period growth rate
margin               -- Target operating income margin
margin_years         -- Years until target operating income margin is achieved
sales_to_capital     -- Future sales-to-capital ratio, i.e. growth in revenue dollars per dollar of capital invested
wacc                 -- Weighted average cost of capital (i.e. average cost for company to raise capital)
cod                  -- Cost of debt (i.e. rate demanded by lenders to lend money to the company)
coe                  -- Cost of equity (i.e. return demanded by company shareholders for owning its stock)
erp                  -- Equity risk premium (i.e. excess return over the risk-free rate investors want for owning equity)
mature_wacc          -- Expected cost of capital for the average mature company
"""

# Concurrently scrape and return a dictionary with dictionaries for each ticker:
    # {*ticker*: {*webpage or financial statement*: {key: value}}}
COMPANY_DATA = asyncio.run(scraper.company_data(TICKER_LIST))

def ttm_income_statement(is_quarterly):
    """Clean and return latest quarterly TTM (trailing twelve month) income statement items for valuation model inputs"""
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
    """Clean and return latest quarterly balance sheet items for valuation model input"""

    bs_output = bs_quarterly        # Quarterly balance sheets (four latest quarters)
    bs_output = bs_output.iloc[:,0] # Keep only the latest quarter

    # Select and order relevant data fields to output
    bs_output = bs_output.reindex(['totalStockholderEquity'])
    return bs_output

def income_statement_trends(is_yearly):
    """Calculate and return revenue and operating margin numbers and changes over recent years"""

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

class CompanyInfo:
    """Pull, estimate and store valuation information specific to one company
    Uses information to create DCF valuation model
    """
    def __init__(self, ticker: str, market_data_path='market_data.csv'):
        self.ticker             = ticker
        self.market_data_path   = market_data_path
        self.income_statement   = ttm_income_statement(COMPANY_DATA[ticker]['is_quarterly'])
        self.balance_sheet      = latest_balance_sheet(COMPANY_DATA[ticker]['bs_quarterly'])
        self.trend_summary      = income_statement_trends(COMPANY_DATA[ticker]['is_yearly'])
        self.quote_summary      = COMPANY_DATA[ticker]['quote_summary']
        self.latest_quarter_end = self.income_statement.name
        self.latest_year_end    = self.trend_summary.columns[0]
        self.name               = self.quote_summary['shortName']
        self.sector             = self.quote_summary['sector']
        self.industry           = self.quote_summary['industry']
        self.sector_aggs        = None
        self.industry_aggs      = None
        self.model_inputs       = {}

        # Add risk-free rate
        self.model_inputs['rf_rate'] = COMPANY_DATA['rf_rate']

        # Replace low sample-size sectors/industries with similar ones
        temp_df = pd.DataFrame({'sector'  :[self.sector],
                           'industry':[self.industry]})
        temp_df = aggregator.clean_industry_data(temp_df)
        self.sector   = temp_df.loc[0,'sector']
        self.industry = temp_df.loc[0,'industry']
        print(f'---\nPreparing valuation information for {self.name} ({self.ticker}) (Sector: {self.sector}, Industry: {self.industry})\n---')
        print(f'{self.quote_summary["longBusinessSummary"]}\n')

    ### Add calculated fields to the market data .csv file

    def tax_rate(self, tax):
        # Define tax rate for calculating unlevered beta
        self.model_inputs['tax_rate'] = tax

    def calculated_fields(self, force_update=False):
        # Calculate unlevered betas and market debt-to-equity
        fields.unlevered_betas(file_path=self.market_data_path, tax=self.model_inputs['tax_rate'], force_update=force_update)

        # Calculate sales-to-capital ratios and invested capital
        fields.sales_to_capital(file_path=self.market_data_path, force_update=force_update)

    ### Calculate market aggregate statistics by sector and industry (while excluding self from averages)

    def get_market_aggs(self, read=False):
        if read is True:
            # Read sector and industry averages from .csv files
            self.sector_aggs   = pd.read_csv('sector_aggs.csv', index_col=0, header=[0,1])   # header argument recognizes multi-index columns
            self.industry_aggs = pd.read_csv('industry_aggs.csv', index_col=0, header=[0,1])
            aggs_date          = datetime.date.fromisoformat(self.sector_aggs.index.name)
        else:
            # Aggregate sector and industry averages from line-by-line market data .csv file
            self.sector_aggs, self.industry_aggs, aggs_date = aggregator.industry_aggregates(self.market_data_path, ex_ticker=self.ticker)
            aggs_date = datetime.date.fromisoformat(aggs_date)

        # Calculate days since market data was updated; if too old (e.g. >3 months), recommend rerunning market data webscrape
        today = datetime.date.today()
        print(f'Sector and industry averages loaded as of {aggs_date} ({(today - aggs_date).days} days since last update)')

    ### Calculate DCF assumptions for growth, margins, sales-to-capital ratio, and cost of capital

    def growth(self, manual=False):
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

    def target_margin(self, manual=False, years=False, max_years=10):
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
            except KeyError:
                print('Recommend running .prepare_model_inputs() method first. Assuming growth_duration of 5 years')
                growth_duration = 5

            years = growth_duration

            if   self_margin >= cutoff_0:
                pass
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
        ### Weighted-average cost of capital (WACC) ###
        if type(wacc) == float:
            self.model_inputs['cost_of_capital'] = wacc

        else:
            print('Auto-calculating weighted-average cost of capital')
            market_cap = self.quote_summary['marketCap']
            total_debt = self.quote_summary['totalDebt']
            d_e_ratio  = total_debt / market_cap
            debt_ratio = total_debt / (market_cap + total_debt)
            print(f'Market debt-to-equity ratio: {d_e_ratio}')
            print(total_debt)

            ## Cost of debt ##
            if type(cod) == float:
                pass
            else:
                print('Auto-calculating cost of debt... (I recommend manually inputing cost of debt from sec filings as estimates can be inaccurate)')
                interest_expense   = -self.income_statement.loc['interestExpense']
                operating_income   = self.income_statement.loc['operatingIncome']
                if operating_income == 0:
                    operating_income = 1
                interest_coverage  = interest_expense / operating_income
                if total_debt == 0:
                    effective_int_rate = 0
                else:
                    effective_int_rate = interest_expense / total_debt

                print(f'Effective interest rate: {effective_int_rate}')

                credit_rating_df = pd.read_csv('synthetic_credit_rating.csv') # File needs to be manually updated (albeit very infrequently) using Damodaran's info (e.g. template models)

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

            ## Cost of equity ##
            if type(coe) == float:
                pass
            else:
                print('Auto-calculating cost of equity...')
                # Retrieve implied equity risk premium
                if type(erp) == float:
                    pass
                else:
                    erp = COMPANY_DATA['implied_erp']

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

            ## Weighted average cost of capital ##
            wacc = (coe * (1 - debt_ratio)) + (after_tax_cod * debt_ratio)
            self.model_inputs['cost_of_capital'] = wacc

        print(f'Weighted-average cost of capital: {wacc}')

        ### Mature (i.e. terminal) WACC ###
        if type(mature_wacc) == float:
            self.model_inputs['mature_wacc'] = mature_wacc
        else:
            print('Auto-estimating mature WACC...')
            mature_wacc = self.model_inputs['rf_rate'] + 0.045 # This is an approximation of market average cost of capital, which mature companies tend to approach
            self.model_inputs['mature_wacc'] = min([wacc, np.average([wacc, mature_wacc])])
        print(f'Mature WACC: {self.model_inputs["mature_wacc"]}')

    ### Prepare data for model
    def prepare_model_inputs(self, tax=0.24,
                             terminal_year=11, growth_years=5, prior_NOL=0, non_operating_assets=0, minority_interests=0,
                             growth=False, margin=False, margin_years=False, sales_to_capital=False,
                             wacc=False, cod=False, coe=False, erp=False, mature_wacc=False):
        """ Run other class methods to set and/or automatically estimate model assumptions and inputs

        Keyword arguments are optional and used to override default model assumptions:
        tax                  -- Marginal tax rate if the company is profitable
        terminal_year        -- Final year of model when the company is mature and growing on pace with the economy
        growth_years         -- Number of years of high growth without any decline in growth rate
        prior_NOL            -- Carry forward "Net Operating Loss" that can be used as future tax offset
        non_operating_assets -- Value of non-operating assets (non-cash assets with no impact on business operations)
        minority_interests   -- Value of minority interests in the company being valued, to be subtracted from valuation
        growth               -- High-growth period growth rate
        margin               -- Target operating income margin
        margin_years         -- Years until target operating income margin is achieved
        sales_to_capital     -- Future sales-to-capital ratio, i.e. growth in revenue dollars per dollar of capital invested
        wacc                 -- Weighted average cost of capital (i.e. average cost for company to raise capital)
        cod                  -- Cost of debt (i.e. rate demanded by lenders to lend money to the company)
        coe                  -- Cost of equity (i.e. return demanded by company shareholders for owning its stock)
        erp                  -- Equity risk premium (i.e. excess return over the risk-free rate investors want for owning equity)
        mature_wacc          -- Expected cost of capital for the average mature company
        """
        # Set tax rate and prepare market data by sector and industry
        self.tax_rate(tax=tax)   # default 24%
        self.calculated_fields()
        self.get_market_aggs()

        # Set model assumptions that cannot be automatically estimated
        self.model_inputs['terminal_year']        = terminal_year        # default 11
        self.model_inputs['growth_duration']      = growth_years         # default 5
        self.model_inputs['prior_NOL']            = prior_NOL            # default 0
        self.model_inputs['non_operating_assets'] = non_operating_assets # default 0
        self.model_inputs['minority_interests']   = minority_interests   # default 0

        # Set or generate model assumptions that can be automatically estimated
        self.growth          (manual = growth)
        self.target_margin   (manual = margin, years = margin_years, max_years = terminal_year)
        self.sales_to_capital(manual = sales_to_capital)
        self.cost_of_capital (wacc = wacc, cod = cod, coe = coe, erp = erp, mature_wacc = mature_wacc)

        # Gather most recent company data needed for DCF model baseline year
        self.model_inputs['base_growth']        = self.quote_summary['revenueGrowth']
        self.model_inputs['base_revenue']       = self.income_statement['totalRevenue']
        self.model_inputs['base_margin']        = self.income_statement['operatingMargin']
        self.model_inputs['debt']               = self.quote_summary['totalDebt']
        self.model_inputs['cash']               = self.quote_summary['totalCash']
        self.model_inputs['shares_outstanding'] = self.quote_summary['sharesOutstanding']
        self.model_inputs['current_price']      = self.quote_summary['regularMarketPrice']

    def valuation_model(self, model_inputs = None, terminal_year = 11):
        # Discounted Cash Flow (DCF) valuation model to value a single stock
        if type(model_inputs) == dict:
            pass
        else:
            model_inputs = self.model_inputs

        ### Construct and empty data frame of size (no. of time periods x no. of DCF items)
        dcf_dict = {}

        # Default is 12 time periods: i.e. Base Year + 10 Years + Terminal Year
        terminal_year    = model_inputs['terminal_year']
        num_time_periods = terminal_year + 1
        # Create a list of dates containing the end date of each time period (year)
        year_ends = []
        for i in range(num_time_periods):
            year_ends.append(self.latest_quarter_end.date() + relativedelta.relativedelta(years=i))

        dcf_dict['Year Ended'] = year_ends

        # Add empty placeholder values for each DCF item
        dcf_items = ['Revenue Growth','Revenue','Operating Margin','Operating Income','Prior Net Operating Loss',
                     'Taxable Operating Income','Tax Rate','After-Tax Operating Income','Sales-to-Capital Ratio',
                     'Reinvestment','Free Cash Flow to Firm','Cost of Capital','Discount Factor','PV (Free Cash Flow to Firm)']
        for item in dcf_items:
            dcf_dict[item] = [0] * num_time_periods

        # Construct placeholder DataFrame
        dcf = pd.DataFrame(dcf_dict).set_index('Year Ended')

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

        dcf['Revenue Growth'] = col_revenue_growth

        ### Create revenue line
        # Base year
        col_revenue = [model_inputs['base_revenue']]
        # Calculate revenue through the terminal year using revenue growth rates
        for i in range(1, num_time_periods):
            col_revenue.append(col_revenue[i-1] * (1 + dcf['Revenue Growth'][i]))

        dcf['Revenue'] = col_revenue

        ### Create operating margin line
        # Inputs
        base_margin            = model_inputs['base_margin']
        target_margin          = model_inputs['target_margin']
        years_to_target_margin = model_inputs['years_to_target_margin']

        # Operating margins
        col_operating_margin = np.linspace(base_margin, target_margin, num = years_to_target_margin + 1).tolist() # Base year through first target margin year
        col_operating_margin += [target_margin] * (num_time_periods - years_to_target_margin - 1) # Remaing years, including terminal year

        dcf['Operating Margin'] = col_operating_margin

        ### Create operating income line
        # No inputs needed
        dcf['Operating Income'] = dcf['Revenue'] * dcf['Operating Margin']

        ### Create net operating loss line
        # Base year
        col_prior_NOL = [model_inputs['prior_NOL']]
        # Calculate net operating loss through the terminal year
        for i in range(1, num_time_periods):
            if col_prior_NOL[i-1] <= dcf['Operating Income'][i-1]:
                col_prior_NOL.append(0) # cumulative net operating losses all used up
            else:
                col_prior_NOL.append(col_prior_NOL[i-1] - dcf['Operating Income'][i-1]) # carry over net operating losses

        dcf['Prior Net Operating Loss'] = col_prior_NOL

        ### Create taxable operating income line:
        # No inputs needed
        col_taxable_operating_income = []
        # Calculate taxable operating income through the terminal year
        for i in range(num_time_periods):
            if dcf['Prior Net Operating Loss'][i] >= dcf['Operating Income'][i]:
                col_taxable_operating_income.append(0) # No taxable income
            else:
                col_taxable_operating_income.append(dcf['Operating Income'][i] - dcf['Prior Net Operating Loss'][i])

        dcf['Taxable Operating Income'] = col_taxable_operating_income

        ### Create tax rate line
        dcf['Tax Rate'] = model_inputs['tax_rate']

        ### Calculate after-tax operating income line
        dcf['After-Tax Operating Income'] = dcf['Operating Income'] - (dcf['Taxable Operating Income'] * dcf['Tax Rate'])

        ### Create sales-to-capital ratio line
        dcf['Sales-to-Capital Ratio'] = model_inputs['sales_to_capital']

        ### Create reinvestment line
        # No inputs needed
        col_reinvestment = [0] # Base year
        # Calculate capital reinvestment amounts (needed for growth) through the terminal year
        for i in range(1, num_time_periods):
            col_reinvestment.append((dcf['Revenue'][i] - dcf['Revenue'][i-1]) / dcf['Sales-to-Capital Ratio'][i])

        dcf['Reinvestment'] = col_reinvestment

        ### Create free cash flow to firm (FCFF) line
        dcf['Free Cash Flow to Firm'] = dcf['After-Tax Operating Income'] - dcf['Reinvestment']
        dcf['Free Cash Flow to Firm'].iat[0] = np.nan # Set base year FCFF to zero, since those cash flows have already been recognized

        ### Create cost of capital line
        # Inputs
        growth_wacc              = model_inputs['cost_of_capital']
        growth_wacc_duration     = model_inputs['growth_duration']
        terminal_cost_of_capital = model_inputs['mature_wacc']

        # Weighted-average cost of capital
        col_cost_of_capital =  [0] # Base year
        col_cost_of_capital += [growth_wacc] * growth_wacc_duration # Growth years
        col_cost_of_capital += np.linspace(growth_wacc, terminal_cost_of_capital, num = terminal_year - growth_wacc_duration + 1).tolist()[1:] # Mature years through terminal year. Indexing excludes one year of growth-stage WACC.

        dcf['Cost of Capital'] = col_cost_of_capital

        ### Create discount factor line
        # No inputs needed
        col_discount_factor = [1] # Base year
        # Calculate discount factor through the terminal year using WACC
        for i in range(1, num_time_periods):
            col_discount_factor.append(col_discount_factor[i-1] / (1 + dcf['Cost of Capital'][i]))

        dcf['Discount Factor']         = col_discount_factor
        dcf['Discount Factor'].iat[-1] = np.nan # Terminal year discount factor is not relevant to terminal value calculation

        ### Create present value of free cash flow line
        dcf['PV (Free Cash Flow to Firm)']         = dcf['Free Cash Flow to Firm'] * dcf['Discount Factor']
        dcf['PV (Free Cash Flow to Firm)'].iat[-1] = np.nan # Terminal year cash flow is included in terminal value calculation


        ############# Calculate Terminal Value and Finish the Valuation #############
        # Calculate cumulative PV(FCFF) for pre-terminal years
        pv_before_terminal = dcf['PV (Free Cash Flow to Firm)'].iloc[:-1].sum()
        print(f'\nPresent value of {terminal_year-1} years of cash flows before terminal year: {pv_before_terminal}')
        # Calcualte PV(terminal value)
        terminal_value    = dcf.iloc[-1].loc['Free Cash Flow to Firm'] / (dcf.iloc[-1].loc['Cost of Capital'] - model_inputs['rf_rate'])
        pv_terminal_value = terminal_value * dcf.iloc[-2].loc['Discount Factor'] # Discount terminal value back to present
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
        equity_value    = total_pv + cash - debt + non_operating_assets - minority_interests
        value_per_share = equity_value / shares_outstanding
        value_to_price  = value_per_share / current_price

        print(f'Total present value of cash flows to shareholders = {equity_value}')
        print(f'Common shares outstanding = {shares_outstanding}')
        print(f'\nEstimated value per share = ${value_per_share:.2f}\n---')
        print(f'\nCurrent price per share = ${current_price:.2f}\n---')
        print(f'\nValue-to-price ratio: {value_to_price:.3f}\n---')

        ############# Output completed DCF and terminal value #############

        dcf_output = dcf.transpose()
        dcf_output.iat[-1, -1] = pv_terminal_value

        final_dict = {'PV (DCF Cash Flows)'    : pv_before_terminal,
                      'PV (Terminal Value)'    : pv_terminal_value,
                      'Total Present Value'    : total_pv,
                      '- Debt'                 : debt,
                      '- Minority Interests'   : minority_interests,
                      '+ Cash'                 : cash,
                      '+ Non Operating Assets' : non_operating_assets,
                      'Value of Equity'        : equity_value,
                      'Market Capitalization'  : self.quote_summary['marketCap'],
                      'Shares Outstanding'     : shares_outstanding,
                      'Value per Share'        : value_per_share,
                      'Current Stock Price'    : current_price,
                      'Value-to-Price Ratio'   : value_to_price
                      }
        final_output = pd.DataFrame.from_dict(final_dict, orient='index')
        final_output.columns = ['Final Valuation']

        return dcf_output, final_output, self.ticker, self.name, value_to_price, self.model_inputs

############# --- End of CompanyInfo class --- #############

def main():
    """Export an Excel workbook containing a DCF valuation in each worksheet per ticker in TICKER_LIST"""
    if __name__ == '__main__':
        result_list = []

        for index, ticker in enumerate(TICKER_LIST):
            company = CompanyInfo(ticker)

            # Check for user-inputted assumptions
            assumptions = {}
            try:
                assumptions = ASSUMPTIONS_LIST[index]
            except IndexError:
                pass

            # Set model assumptions and inputs
            company.prepare_model_inputs(**assumptions)

            # Run valuation model
            symbol, name, dcf_output, final_output, ratio, assumptions = company.valuation_model() # terminal_year=11
            result_list.append((symbol, name, dcf_output, final_output, ratio, assumptions))

        excel_formatter.export_dcf(tuple_list=result_list)

main()

t1, tp1 = time.time(), time.process_time()
print(f'Normal time: {np.round(t1-t0, 3)}s, Process time: {np.round(tp1-tp0, 3)}s')
