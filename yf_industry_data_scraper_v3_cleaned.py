import asyncio
import aiohttp  # asynchronous version of 'requests' module
import numpy as np
import pandas as pd
import json
import re
import time
import math
import yahoo_fin.stock_info as si
from russell3000_tickers import Russell3000_Tickers # Class with methods .download() and .to_csv() or .to_list()

#from yf_scraper_asyncio_vF import get_html

r3000 = Russell3000_Tickers()
r3000.download()
tickers = r3000.to_list()

# Set display options to show more rows and columns than default
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 200)

income_statement_url = 'https://finance.yahoo.com/quote/{}/financials'
balance_sheet_url    = 'https://finance.yahoo.com/quote/{}/balance-sheet'
quote_summary_url    = 'https://finance.yahoo.com/quote/{}'
sp500 = si.tickers_sp500()
russell3000 = ['A', 'AAL', 'AAP', 'AAPL', 'ABBV', 'ABC', 'ABMD', 'ABT', 'ACN', 'ADBE']
#ticker_list = ['A', 'AAL', 'AAP', 'AAPL', 'ABBV', 'ABC', 'ABMD', 'ABT', 'ACN', 'ADBE']
#ticker_list = ['FIVN', 'ASAN']
ticker_list = ['FIVN']

segmented_ticker_list = []

print(f'Number of tickers: {len(tickers)}')

t0 = time.time()
tp0 = time.process_time()


async def get_html(session, url):
    # Web scrape and return html text
    async with session.get(url, ssl=False) as resp:
        html = await resp.text()
        return html

async def bound_fetch(semaphore, session, url):
    # Fetch function with semaphore
    async with semaphore:
        html = await get_html(session, url)
        return html

async def scrape_tickers(semaphore, session, url, ticker_list):
    # Asyncronously send web requests and return data
    print('begun')
    tasks = [bound_fetch(semaphore, session, url.format(ticker)) for ticker in ticker_list]
    html_list = await asyncio.gather(*tasks)
    print('received')
    return html_list

def parse_json(html):
    # Convert html into json
    try:
        json_str = html.split('root.App.main =')[1].split('(this)')[0].split(';\n}')[0].strip()
    except Exception as e:
        print(e)
        print(f'Could not parse_json (i.e. could not identify splits):\n{html[:300]}')
    try:
        data = json.loads(json_str)['context']['dispatcher']['stores']['QuoteSummaryStore']
    except:
        print('Empty')
        return '{}'
    else:
        new_data = json.dumps(data).replace('{}', 'null')
        new_data = re.sub(r'\{[\'|\"]raw[\'|\"]:(.*?),(.*?)\}', r'\1', new_data) # {"raw":(.*?),(.*?)} # r'\1' references the contents of the first parentheses
        json_info = json.loads(new_data)
        return json_info

errors = []

my_timeout = aiohttp.ClientTimeout(total=None,      # End session after this number of total seconds; `None` for unlimited; default is 300 seconds
                                   sock_connect=30, # How long to wait before an open socket allowed to connect
                                   sock_read=30)    # How long to wait with no data being read before timing out
my_connector = aiohttp.TCPConnector(limit=None)     # Total number of simultaneous connections

client_args = dict(trust_env=True, timeout=my_timeout, connector=my_connector)

async def industry_data(ticker_list):
    # This function outputs a dictionary: key = stock ticker, value = dictionary containing:
        # income statement
        # balance sheet
        # other relevant stock information
    sem = asyncio.Semaphore(25)
    # Scrape data
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0)) as session:
        html_list_qs = await scrape_tickers(sem, session, quote_summary_url, ticker_list)

    output = {}

    # Other stock information
    print('start looping')
    for html in html_list_qs:
        try:
            json_info = parse_json(html)
            output[json_info['symbol']] = {}
            # Alter this list to identify stock information of interest
            json_info_keys = [('price','shortName'),('financialData','totalDebt'),('financialData','totalRevenue'),('financialData','revenueGrowth'),
                              ('financialData','operatingMargins'),('defaultKeyStatistics','beta'),('price','regularMarketPrice'),('price','currency'),
                              ('summaryProfile','industry'),('summaryProfile','sector'),('summaryProfile','country'),
                              ('summaryProfile','longBusinessSummary')]
            quote_summary_dict = {}
            for category, stat in json_info_keys:
                try:
                    quote_summary_dict[stat] = json_info[category][stat]
                except:
                    quote_summary_dict[stat] = np.nan
            output[json_info['symbol']]['quote_summary'] = quote_summary_dict

        except Exception as e:
            print(f'Overall exception: {e}')
            try:
                print(json_info['symbol'])
                errors.append(json_info['symbol'])
            except:
                errors.append('error')

    output_df = pd.concat({k: pd.DataFrame().from_dict(v, orient='index') for k, v in output.items()}, axis=0)
    output_df.rename(columns={'Unnamed: 1':'webpage'}, inplace=True)
    output_df.to_csv('industry_data.csv')

    print(output)
    print(output_df)
    return output_df

asyncio.run(industry_data(tickers))

print(f'Number of unique industries: {len(industry_dict)}')
print(pd.DataFrame.from_dict(industry_dict, orient='index'))
print(f'Number of unique sectors: {len(sector_dict)}')
print(pd.DataFrame.from_dict(sector_dict, orient='index'))

t1 = time.time()
tp1 = time.process_time()
print(f'Normal time: {np.round(t1-t0, 3)}s, Process time: {np.round(tp1-tp0, 3)}s')
print(errors)
