# Auto Stock Valuation Tool (DCF methodology)
### !! Disclaimer !!
Made this for fun! Please do not consider this tool to be financial advice!

Every stock is unique and cannot be accurately intrinsically valued using an automated tool. If the intrinsic valuation differs from the stock price, there's always a reason (and usually a good one).

## About
The purpose of this tool is to create Discounted Cash Flow (DCF) valuation models. For instant use, this tool can export a DCF valuation model with automatically estimated assumptions. For customized use, you can select the model assumptions to mannually adjust (*see instructions*).

What is a DCF valuation model? In simple terms, a DCF model forecasts company cash flows in perpertuity based on revenue and profitability assumptions, and then evaluates those cash flows based on assumed risk and time-value of money.

## Usage instructions
There are two scripts that can be run:
| Script name                        | Runtime (approx.)               | Frequency              |
| ---------------------------------- | ------------------------------- | ---------------------- |
| **1)** `yf_market_data_scraper.py` | 2.5 to 3 hours (rate limited)   | Run every ~3 months    |
| **2)** `valuation_tool.py`         | 3 seconds (+1s per extra stock) | Run for each valuation |

#### Script 1) Yahoo Finance Market Data Scraper
`yf_market_data_scraper.py` scrapes data from Yahoo Finance for every company listed on [MSCI's Russell 3000 ETF](https://www.ishares.com/us/products/239714/ishares-russell-3000-etf), which contains around ~2,700 stock tickers as of January, 2022. The web scraping is done asynchronously with 1.5s extra delay between requests, because if Yahoo Finance's rate limit is exceeded, data is returned blank or incorrect.

The data is exported to `market_data.csv` with cell "A1" containing the effective date of the data.

#### Script 2) DCF Valuation Tool
`valuation_tool.py` constructs a DCF model for each specified stock ticker. To do this, it web scrapes financial data and sets DCF model assumptions. Some model assumptions are automatically estimated if no input is given. The other model assumptions require manual input, although they also have default values.

The data is exported to `DCF_valuations.xlxs`, which is a formatted Excel file with one DCF model per stock.

##### Inputing Model Assumptions
Assumptions can be individually adjusted in the script by passing arguments to the `.prepare_model_inputs()` method. This is done by adding to a dictionary in the `ASSUMPTIONS_LIST` variable. There are plans to make this a much more accessible process in the next couple months.

## Relationship between Main Scripts and Imported Local Modules (Process Flow)
### Script 1) Yahoo Finance Market Data Scraper
<img alt="Market Data Scraper" src="/README_images/process_diagram_script_1.png" width="480"> <img alt="Legend" src="/README_images/legend.png" width="240">

### Script 2) DCF Valuation Tool
<img alt="Valuation Tool" src="/README_images/process_diagram_script_2.png">

## Sample Excel Output
<img alt="Excel Output" src="/README_images/excel_output.png">

### Output Interpretation
The value-to-price ratio determines whether the model believes a stock is overpriced or undervalued.

Despite this code running automatically, it's important to manually validate and tweak assumptions so that the valuation fits a logical story. Without assumptions that can be explained and justified, the DCF model is unreliable.
#### Reviewing Assumptions Example 1: Hypothetical Pizza Shop Revenues
One first step is to review revenue assumptions by researching the total market size and estimating a realistic market share for your company. For example, let's say you're valuing a pizza chain:
- In the U.S., pizza shops had total sales revenue of around $46bn in 2020
- Let's say this is projected to grow at 5% CAGR over the next 10-years (i.e. $75bn in sales by 2030)
- Let's say you estimate your pizza chain to achieve a best-case scenario of 33% market share in 10 years
- If your 2030 model revenue is above $25bn (33% share), it's a sign the revenue growth assumptions may be too high

#### Reviewing Assumptions Example 2: Asana Inc.'s Revenues
Let's look at the real case of Asana, Inc. (task management software company) shown in the image above:
- The global task management software market was $2.4bn in 2020 and projected to reach $4.7bn by 2026 (~12% CAGR)
  - The North American market was $578m in 2018 and projected to reach $1.4bn by 2026 (~12% CAGR)
- By comparison, Asana's 12-month 2021-10-30 revenue was $335m, resulting in ~10-15% global market share
  - Asana's revenue segmentation is 58% U.S. and 42% global, resulting in ~25-30% local market share
- The model estimates Asana's revenue to be over $3bn by 2026, which is projected to be over **60% global market share (too high)**
- This suggests the revenue growth assumptions are too generous, or that Asana's market is actually larger than being considered
- Despite the revenue growth assumptions, the model calculates a $45.37 Value-Per-Share, 25% lower than the $60.18 Stock Price
- However, before we can interpret this, the other assumptions also need to be reviewed (*full list of assumptions below*)

Overall, there's no right or wrong model; it's all about the assumptions. Remember, if the DCF valuation differs from the market price, which it often will, there's usually a good reason that's being overlooked by the model.

## Complete List of DCF Model Assumptions and Inputs
#### Assumptions that can be manually inputted or automatically estimated:
| Model Input                      | Automatic estimation is primarily based on the following... |
| -------------------------------- | ------------------------------------------------------- |
| Growth rate                      | Recent company growth rates                             |
| Target operating margin          | 75th percentile of industry and sector margins          |
| Years to achieve target margin   | Current margin compared to industry and sector margins  |
| Sales-to-capital ratio           | Sector ratios and company ratios                        |
| Weighted-average cost of capital | Calculated based on cost of debt and equity assumptions |
| - Cost of debt                   | Interest to debt ratio and interest coverage ratio      |
| - Cost of equity                 | Industry and sector betas                               |
|   - Equity risk premium          | U.S. market implied equity risk premium                 |
| Mature firm cost of capital      | Risk-free rate + 4.5%                                   |

#### Assumptions that require manual input:
| Model Input                              | Default value |
| ---------------------------------------- | ------------- |
| Number of high-growth years              | 5             |
| Number of DCF years before terminal year | 10            |
| Tax rate                                 | 24%           |
| Current net operating loss carryover     | 0             |
| Value of non-operating assets            | 0             |
| Value of minority interests              | 0             |

#### Model inputs that are facts, not assumptions:
- Risk free rate (10-year U.S. treasury yield)
- Date of latest quarterly financials
- Trailing 12-month revenue
- Trailing 12-month operating margin
- Total debt (interest bearing loans)
- Cash, cash-equivalents and marketable short-term assets
- Shares outstanding

*Note: some model assumptions and inputs will vary throughout the model*
### Acknowledgements
This script is possible thanks to everything I have learnt from Prof. Aswath Damodaran ([YouTube Channel](https://www.youtube.com/c/AswathDamodaranonValuation)) (*Note: this was NOT made with the Professor's consultation*). You can learn more about finance and valuation on his YouTube channel and website.
