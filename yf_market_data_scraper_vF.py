# market_data(ticker_list) function scrapes all relevant data for Russell 3000 stocks from Yahoo Finances
    # 'market_data.csv' : stock data
    # 'status_cods.csv' : status code request errors (not 200)
    # 'errors.csv'      : python exceptions

# This script is meant to be run as __main__, not imported
# Typical runtime is around 2.5 hours

import asyncio
import aiohttp  # concurrent version of 'requests' module
import numpy as np
import pandas as pd
import json
import re
import time
import datetime
import random
import csv
from russell3000_tickers_vF import Russell3000_Tickers # Class with methods .download(), .to_list() and .to_csv()

t0 = time.time()
tp0 = time.process_time()

# Get list of Russell 3000 stock tickers
r3000 = Russell3000_Tickers()
r3000.download()
#r3000.to_csv() # Export to csv (for reference only)
tickers = r3000.to_list()

# If scraping through Russell 3000 will take too long, 'tickers' can be set to S&P 500 instead
# import yahoo_fin.stock_info as si
# tickers = si.tickers_sp500()
print(f'Number of tickers: {len(tickers)}')

# Set display options to show more rows and columns
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 200)

quote_summary_url = 'https://finance.yahoo.com/quote/{}'
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
    # Asynchronously loop through web requests and return data once completed
    html_list = []
    count = 0

    print(f'begun scraping - asynchronous')
    for ticker in ticker_list:
        url_html = await get_html(session, url.format(ticker))
        html_list.append(url_html)
        count += 1
        if count % 50 == 0:
            print(f'{count} requests processed')
        await asyncio.sleep(random.randint(1,3)) # sleep to avoid surpassing rate limit

    print('all data received')
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
        print('Empty html')
        errors.append(['Empty html'])
        return '{}'
    else:
        new_data = json.dumps(data).replace('{}', 'null')
        new_data = re.sub(r'\{[\'|\"]raw[\'|\"]:(.*?),(.*?)\}', r'\1', new_data) # {"raw":(.*?),(.*?)} # r'\1' references the contents of the first parentheses
        json_info = json.loads(new_data)
        return json_info

async def market_data(ticker_list):
    # This function outputs a dictionary: key = stock ticker, value = dictionary containing relevant stock information
    if __name__ == '__main__':
        # Scrape data
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0)) as session:
            html_list_qs = await scrape_tickers(session, quote_summary_url, ticker_list)

        output = {}

        # Alter this list to reflect stock information of interest
        json_info_keys = [('price','shortName'),('financialData','totalDebt'),('financialData','totalRevenue'),('financialData','totalCash'),
                          ('financialData','debtToEquity'),('financialData','revenueGrowth'),('financialData','operatingMargins'),('defaultKeyStatistics','beta'),
                          ('price','currency'),('price','regularMarketPrice'),('defaultKeyStatistics','sharesOutstanding'),('price','marketCap'),
                          ('summaryProfile','industry'),('summaryProfile','sector'),('summaryProfile','country'),('summaryProfile','longBusinessSummary')]
        for url, html in html_list_qs:
            try:
                json_info = parse_json(html)
                quote_summary_dict = {}
                for category, stat in json_info_keys:
                    try:
                        quote_summary_dict[stat] = json_info[category][stat]
                    except:
                        quote_summary_dict[stat] = np.nan
                output[json_info['symbol']] = quote_summary_dict

            except Exception as e:
                print(f'Overall exception: {e}\n{url}')
                errors.append([f'Overall exception: {e}: {url}'])

        # Convert nested dictionary to DataFrame
        output_df = pd.DataFrame().from_dict(output, orient='index')
        output_df.rename_axis([f'{datetime.date.today()}'], inplace=True) # Set index header as today's date
        output_df.to_csv('market_data.csv')
        print(f'Wrote data to \'market_data.csv\' for {output_df.index.size} tickers')
        
        # Export status codes of web request errors
        pd.DataFrame().from_dict(status_codes, orient='index').to_csv('status_codes.csv')

        # Export list of exception errors
        with open('errors.csv', 'w', encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerows(errors)

        return output_df
    
    else:
        return None

asyncio.run(market_data(tickers))

t1 = time.time()
tp1 = time.process_time()
print(f'Normal time: {np.round(t1-t0, 3)}s, Process time: {np.round(tp1-tp0, 3)}s')
