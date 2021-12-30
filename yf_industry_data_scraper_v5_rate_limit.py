import asyncio
import aiohttp  # asynchronous version of 'requests' module
import numpy as np
import pandas as pd
import json
import re
import time
import datetime
from russell3000_tickers_vF import Russell3000_Tickers # Class with methods .download() and .to_csv() or .to_list()

r3000 = Russell3000_Tickers()
r3000.download()
r3000.to_csv()
tickers = r3000.to_list()

# If Russell 3000 download time is too long, 'tickers' can be set to S&P 500 instead
# import yahoo_fin.stock_info as si
# tickers = si.tickers_sp500()
# No data for these 23 tickers when running the full r3000; however they have data when running their own shorter list
#tickers = ['ADGI','AVDX','CIVI','CRBU','CRGY','ERAS','ETD','GXO','HRT','IAS','ICVX','INFA','LAW','LYLT','MCW','MLKN','OLPX','PYCR','RIVN','ROCC','VMEO','VRE','XMTR']
# Exceptions for these 11 tickers when running the full r3000; no data/url available on Yahoo Finance
#tickers = ['BFA','BFB','BRKB','CWENA','GEFB','HEIA','JWA','LENB','LGFA','LGFB','MOGA']

# Set display options to show more rows and columns than default
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 200)

income_statement_url = 'https://finance.yahoo.com/quote/{}/financials'
balance_sheet_url    = 'https://finance.yahoo.com/quote/{}/balance-sheet'
quote_summary_url    = 'https://finance.yahoo.com/quote/{}'

print(f'Number of tickers: {len(tickers)}')

t0 = time.time()
tp0 = time.process_time()
status_codes = {}
errors = []

async def get_html(session, url):
    # Web scrape and return html text
    async with session.get(url, ssl=False) as resp:
        if resp.status != 200:
            status_codes[url] = resp.status

            # Attempt to web scrape a second time after waiting 5 seconds
            await asyncio.sleep(5)
            async with session.get(url, ssl=False) as resp2:
                html = await resp2.text()
                return (url, html)

        html = await resp.text()
        return (url, html)

async def scrape_tickers(session, url, ticker_list):
    # Asyncronously send web requests and return data
    delay = 1/12 # Delay (seconds) between sending each web request
    tasks = []
    print(f'begun scraping - {delay:.3f}s per web request')
    for ticker in ticker_list:
        tasks.append(asyncio.create_task(get_html(session, url.format(ticker))))
        await asyncio.sleep(delay)
    html_list = await asyncio.gather(*tasks)
    print('received')
    return html_list

def parse_json(html):
    # Convert html into json
    try:
        json_str = html.split('root.App.main =')[1].split('(this)')[0].split(';\n}')[0].strip()
    except Exception as e:
        print(e)
        errors.append([e])
    try:
        data = json.loads(json_str)['context']['dispatcher']['stores']['QuoteSummaryStore']
    except:
        print('Empty')
        errors.append(['Empty'])
        return '{}'
    else:
        new_data = json.dumps(data).replace('{}', 'null')
        new_data = re.sub(r'\{[\'|\"]raw[\'|\"]:(.*?),(.*?)\}', r'\1', new_data) # {"raw":(.*?),(.*?)} # r'\1' references the contents of the first parentheses
        json_info = json.loads(new_data)
        return json_info

async def industry_data(ticker_list):
    # This function outputs a dictionary: key = stock ticker, value = dictionary containing:
        # relevant stock information

    # Scrape data
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0)) as session:
        html_list_qs = await scrape_tickers(session, quote_summary_url, ticker_list)

    output = {}

    # Alter this list to reflect stock information of interest
    print('start looping')
    json_info_keys = [('price','shortName'),('financialData','totalDebt'),('financialData','totalRevenue'),('financialData','revenueGrowth'),
                      ('financialData','operatingMargins'),('defaultKeyStatistics','beta'),('price','regularMarketPrice'),('price','currency'),
                      ('summaryProfile','industry'),('summaryProfile','sector'),('summaryProfile','country'),
                      ('summaryProfile','longBusinessSummary')]
    for url, html in html_list_qs:
        try:
            json_info = parse_json(html)
            output[json_info['symbol']] = {}
            quote_summary_dict = {}
            for category, stat in json_info_keys:
                try:
                    quote_summary_dict[stat] = json_info[category][stat]
                except:
                    quote_summary_dict[stat] = np.nan
            output[json_info['symbol']]['quote_summary'] = quote_summary_dict

        except Exception as e:
            print(f'Overall exception: {e}\n{url}')
            errors.append([f'Overall exception: {e}: {url}'])

    output_df = pd.concat({k: pd.DataFrame().from_dict(v, orient='index') for k, v in output.items()}, axis=0)
    # Set index header as today's date
    output_df.rename_axis([f'{datetime.date.today()}','webpage'], inplace=True)
    output_df.to_csv('industry_data.csv')
    print(f'Wrote data to \'industry_data.csv\' for {output_df.index.size} tickers (some may have blank data)')
    return output_df

asyncio.run(industry_data(tickers))

t1 = time.time()
tp1 = time.process_time()
print(f'Normal time: {np.round(t1-t0, 3)}s, Process time: {np.round(tp1-tp0, 3)}s')

# Output status codes of web requests
pd.DataFrame().from_dict(status_codes, orient='index').to_csv('status_codes.csv')

# Output list of errors
import csv
with open('errors.csv', 'w', encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerows(errors)
