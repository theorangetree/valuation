import asyncio
import aiohttp  # asynchronous version of 'requests' module
import pandas as pd
import json
import re
import time
import numpy as np
import yahoo_fin.stock_info as si
import math
from bs4 import BeautifulSoup
import dateutil.parser

# Set display options to show more rows and columns than default
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 200)

income_statement_url = 'https://finance.yahoo.com/quote/{}/financials'
balance_sheet_url    = 'https://finance.yahoo.com/quote/{}/balance-sheet'
quote_summary_url    = 'https://finance.yahoo.com/quote/{}'
damodaran_url        = 'http://people.stern.nyu.edu/adamodar/New_Home_Page/home.htm'

t0 = time.time()
tp0 = time.process_time()

async def get_html(session, url):
    # Web scrape and return html text
    async with session.get(url, ssl=False, headers={'User-agent': 'Mozilla/5.0'}) as resp:
        html = await resp.text()
        return html

async def scrape_tickers(session, url, ticker_list):
    # Asyncronously send web requests and return data
    print('begun scraping')
    tasks = [get_html(session, url.format(ticker)) for ticker in ticker_list]
    html_list = await asyncio.gather(*tasks)
    print('received')
    return html_list

async def risk_free_rate(session, url):
    # Ayncronously scrape risk-free rate, i.e. 10-yr US treasury bond yield
    print('begun rf_rate')
    task = get_html(session, url.format('%5ETNX'))
    rf_rate = await asyncio.gather(task)
    print('received rf_rate')
    return rf_rate

async def implied_erp(session, url):
    # Asyncronously scrape implied equity risk premium from Prof. Damodaran's website
    print('begun implied_erp')
    try:
        task = get_html(session, url)
        home_page = await asyncio.gather(task)
        soup = BeautifulSoup(home_page[0], 'html.parser')

        # Parse html for implied equity risk premium and the date of the result
        start = soup.find(string=re.compile('Implied ERP on'))
        date  = dateutil.parser.parse(start, fuzzy=True).date()
        regex = re.compile(r'\d\.\d{2}') # identify x.xx float
        line  = start.find_next(string=regex)

        # Convert implied ERP from percentage to float
        implied_erp = float(re.match(regex, line).group()) / 100
        print(f'received implied_erp from Prof. Damodaran\'s website of {implied_erp:.4f} on {date}')
    except:
        print('error retrieving implied_erp from Prof. Damodara\'s website')
        implied_erp = 0.05 # base assumption of 5% ERP

    return implied_erp

def parse_json(html):
    # Convert html into json
    try:
        json_str = html.split('root.App.main =')[1].split('(this)')[0].split(';\n}')[0].strip()
    except Exception as e:
        print(e)
        print(f'Could not parse_json:\n{html}')
    try:
        data = json.loads(json_str)['context']['dispatcher']['stores']['QuoteSummaryStore']
    except:
        return '{}'
        print(f'Nothing returns:\n{html}')
    else:
        new_data = json.dumps(data).replace('{}', 'null')
        new_data = re.sub(r'\{[\'|\"]raw[\'|\"]:(.*?),(.*?)\}', r'\1', new_data)
        json_info = json.loads(new_data)
        return json_info

def parse_table(json_info):
    # Convert json into pandas DataFrame
    df = pd.DataFrame(json_info)
    if df.empty:
        return df
    del df["maxAge"]
    df.set_index("endDate", inplace=True)
    df.index = pd.to_datetime(df.index, unit="s")

    df = df.transpose()
    df.index.name = "Line Item"
    df.columns.name = "End Date"

    return df

async def company_data(ticker_list):
    # This function outputs a dictionary: key = stock ticker, value = dictionary containing:
        # income statement
        # balance sheet
        # other relevant stock information
    
    # Scrape data
    async with aiohttp.ClientSession() as session:
        tasks_main = [scrape_tickers(session, income_statement_url, ticker_list),
                      scrape_tickers(session, balance_sheet_url, ticker_list),
                      scrape_tickers(session, quote_summary_url, ticker_list),
                      risk_free_rate(session, quote_summary_url),
                      implied_erp(session, damodaran_url)]
        html_list_is, html_list_bs, html_list_qs, rf_rate, erp = await asyncio.gather(*tasks_main)

    output = {}
    
    # Income statement data - quarterly, yearly
    for html in html_list_is:
        json_info = parse_json(html)
        output[json_info['symbol']] = {}
        output[json_info['symbol']]['is_quarterly'] = parse_table(json_info["incomeStatementHistoryQuarterly"]["incomeStatementHistory"])
        output[json_info['symbol']]['is_yearly'] = parse_table(json_info["incomeStatementHistory"]["incomeStatementHistory"])

    # Balance sheet data - quarterly
    for html in html_list_bs:
        json_info = parse_json(html)
        output[json_info['symbol']]['bs_quarterly'] = parse_table(json_info["balanceSheetHistoryQuarterly"]["balanceSheetStatements"])
    
    # Other stock information
    for html in html_list_qs:
        try:
            json_info = parse_json(html)
            # Alter this list to identify stock information of interest
            json_info_keys = [('price','shortName'),('financialData','totalDebt'),('financialData','totalCash'),('financialData','revenueGrowth'),
                              ('defaultKeyStatistics','beta'),('price','currency'),('price','regularMarketPrice'),
                              ('defaultKeyStatistics','sharesOutstanding'),('price','marketCap'),('summaryProfile','industry'),
                              ('summaryProfile','sector'),('summaryProfile','country'),('summaryProfile','longBusinessSummary')]
            quote_summary_dict = {b: json_info[a][b] for a, b in json_info_keys}
            output[json_info['symbol']]['quote_summary'] = quote_summary_dict
        except Exception as e:
            print(e)
            print('Error finding a key for {}'.format(json_info['symbol']))
    
    # Risk-free rate
    json_info = parse_json(rf_rate[0])
    output['rf_rate'] = json_info['price']['regularMarketPreviousClose'] / 100

    # Implied equity risk premium
    output['implied_erp'] = erp

    return output

# Uncomment to test
#print(asyncio.run(company_data(['CZR'])))

t1 = time.time()
tp1 = time.process_time()
print(f'Normal time: {np.round(t1-t0, 3)}s, Process time: {np.round(tp1-tp0, 3)}s')
