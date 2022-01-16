# Auto Stock Valuation Tool (DCF methodology)
### Disclaimer
Made this for fun! Please do not consider this tool to be financial advice!

Any individual stock is unique and cannot be accurately intrinsically valued using an automated tool. Furthermore, every publically-traded stock on the stock market has it's price for a reason. If the intrinsic valuation differs from the stock price, there's always a reason (and usually a good one too).

## About
I wrote this code to help myself create fast, simple, bottoms-up DCF stock valuations. DCF stands for Discounted Cash Flow and is a methodology for valuing companies (or any other cash-generating asset) intrinsically, meaning the value is calculated 100% independently of actual market value.

In the simplest terms, a DCF model contructs projection of cash flows in perpertuity, based on company revenue and profitability assumptions.

## How to use
There are two scripts that should be run:
1. yf_market_data_scraper.py
2. valuation_tool.py

#### Script 1: Market Data Scraper (Setup)
The `yf_market_data_scraper.py` script scrapes data from Yahoo Finance for every stock listed on [MSCI's Russell 3000 ETF](https://www.ishares.com/us/products/239714/ishares-russell-3000-etf), which contains around ~2,700 stock tickers as of January, 2022.

Originally, the webscraping was performed synchronously (concurrently), however Yahoo Finance has a rate limit (exact limit unknown), after which data is returned blank or incorrect. Thus, the scraper is set to run synchronously and includes a ~1.5s delay, averaging to around one request every 3-3.5 seconds and totaling ~2.5 hours to complete the full market data scrape.

The data is exported to `market_data.csv` with the first cell containing the effective date of the data.

#### Script 2: Valuation Tool
The `valuation_tool.py` script constructs a DCF model for one or multiple specified stock tickers. To do this, it scrapes and gathers the necessary financial data and DCF model assumptions. Some model assumptions can be automatically estimated if a number is not manually provided. The other model assumptions require manual input, although they also have default values. See below for a list of DCF model assumptions and inputs.

The data is exported to `DCF_valuations.xlxs', which is a formatted Excel file with one stock and DCF model per worksheet.

## Valuation Tool Process Diagram
![Process Diagram](/README_images/valuation_tool_process_diagram.png)

## Sample Excel Output
![Excel Output](/README_images/sample_excel_output.png)

### Output Interpretation
The value to price ratio determines whether the model believes a stock is overpriced or undervalued (NOT financial advice).

Despite this code running automatically, the most important thing is to manually validate and tweak assumptions, so that the valuation fits a logical story. Without assumptions that can be explained and justified, the DCF model is unreliable.

When checking assumptions, one first step is to research the total size of an industry and estimate a realistic market share in order to sense-check the revenue growth rate and check your year-10 revenue. For example, let's say you're valuing a pizza chain stock:
- In the U.S., pizza shops had total sales revenue of around $46bn in 2020
- Let's say this is projected to grow at 5% CAGR over the next 10-years (i.e. $75bn in sales by 2030)
- Let's say you expect your pizza chain to achieve a best-case scenario of 33% market share in 10 years
- If your year-10 revenue is above $25bn, it's a sign your revenue growth assumptions may be too high

Overall, there's no right or wrong model; it's all about setting the assumptions. Just remember, if the DCF model valuation differs from the market price, which it often will, there's usually a good reason that's being missed by the model.

## Complete List of DCF Model Assumptions and Inputs
Note that model assumptions and inputs can change and vary over the years in the model.
#### Assumptions that can be manually input or automatically estimated (based on industry, sector and historical data)
- Growth rate
- Operating margin
- Number of years to achieve target operating margin
- Sales to capital ratio (i.e. revenue growth per dollar of capital invested)
- Weighted-average cost of capital
  - Cost of debt
  - Cost of equity
    - Equity risk premium
- Mature company weighted-average cost of capital

#### Assumptions that require manual input
- Number of high-growth years               (default=5)
- Number of DCF years before terminal year  (default=10)
- Marginal tax rate                         (default=24%)
- Current net operating loss carryover      (default=0)
- Value of non-operating assets             (default=0)
- Value of minority interests               (default=0)

#### Model inputs that are not assumptions
- Risk free rate (10-year U.S. treasury yield)
- Latest quarterly financials date
- Trailing 12-month revenue
- Trailing 12-month operating margin
- Total debt (interest bearing loans)
- Cash, cash-equivalents and marketable short-term assets
- Shares outstanding

#### Acknowledgements
This script is possible thanks to everything I have learnt from valuation legend Prof. Aswath Damodaran ([YouTube Channel](https://www.youtube.com/c/AswathDamodaranonValuation)) (Note: this was NOT made with the Prof's consultation). You can learn more about finance and valuation from his YouTube channel and website.