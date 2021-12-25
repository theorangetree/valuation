# Scrape a list of Russell 3000 tickers using the iShares IWV ETF webpage
# Writes a csv file with the name 'russell3000.csv'
# https://www.ishares.com/us/products/239714/ishares-russell-3000-etf

import requests
import re
import pandas as pd
from bs4 import BeautifulSoup
from io import StringIO

site_url = 'https://www.ishares.com/us/products/239714/ishares-russell-3000-etf'
file_url = 'https://www.ishares.com/us/products/239714/ishares-russell-3000-etf/1467271812596.ajax?fileType=csv&fileName=IWV_holdings&dataType=fund'

def check_file_url():
    # Attempt to find the file_url within the original webpage

    # Pull html
    site = requests.get(site_url)
    print(f'(Status {site.status_code}) Request sent to: {site_url}')
    soup = BeautifulSoup(site.text, 'html.parser')

    # Find the file_url
    path = soup.find('a', attrs={'class':'icon-xls-export','data-link-event':'holdings:holdings'})['href']
    test_url = f'https://www.ishares.com{path}'
    if test_url == file_url:
        print('file_url FOUND within site_url')
    else:
        print('file_url NOT FOUND within site_url')

class Russell3000_Tickers:
    def __init__(self):
        self.df = pd.DataFrame()

    def download(self):
        # Pull Russell 3000 holdings from an online csv file
        site = requests.get('https://www.ishares.com/us/products/239714/ishares-russell-3000-etf/1467271812596.ajax?fileType=csv&fileName=IWV_holdings&dataType=fund')
        print(f'(Status {site.status_code}) Request sent to: {file_url}')
        csv_text = re.sub(r'^.*Ticker','Ticker', site.text, flags=re.DOTALL) # Remove extra text from start of string
        df = pd.read_csv(StringIO(csv_text)) # Convert csv string to DataFrame
        self.df = df.loc[(df['Exchange'].isin(['NASDAQ','New York Stock Exchange Inc.','Nyse Mkt Llc'])) &
                         (df['Location']=='United States') &
                         (df['Market Value'].str.len()>8)
                         ,'Ticker'].reset_index(drop=True) # Filter for tickers of ordinary U.S. equities

    def import_local(self, csv_file_path):
        self.df = pd.read_csv(csv_file_path)

    def to_csv(self):
        print(f'{len(self.df)} tickers written to \'russell3000.csv\'')
        self.df.to_csv('russell3000.csv', header=False, index=False)

    def to_list(self):
        return self.df.tolist() # Convert Series to list
