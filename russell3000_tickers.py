""" Scrape a list of Russell 3000 tickers using the iShares IWV ETF webpage

Class:
Russell3000_Tickers() -- download and export list of Russell 3000 tickers
    init()            -- set URL variables (may need updating)
    check_file_url()  -- check if file_url is within site_url
    download()        -- read online CSV file to Series; filter for equity stock tickers
    to_csv()          -- export Series of ticker to CSV file
    to_list()         -- return list of tickers
"""
import re
from io import StringIO
import requests
import pandas as pd
from bs4 import BeautifulSoup

class Russell3000Tickers():
    """Download and export list of Russell 3000 tickers"""
    
    def __init__(self):
        """Set site urls; update the urls as needed"""
        self.site_url = 'https://www.ishares.com/us/products/239714/ishares-russell-3000-etf'
        self.file_url = self.site_url + '/1467271812596.ajax?fileType=csv&fileName=IWV_holdings&dataType=fund'
        self.df       = pd.DataFrame()

    def check_file_url(self):
        """Attempt to find the file_url within the original webpage"""
        # Pull html
        site = requests.get(self.site_url)
        print(f'(Status {site.status_code}) Request sent to: {self.site_url}')
        soup = BeautifulSoup(site.text, 'html.parser')

        # Find the file_url
        path = soup.find('a', attrs={'class':'icon-xls-export','data-link-event':'holdings:holdings'})['href']
        test_url = f'https://www.ishares.com{path}'
        if test_url == self.file_url:
            print('file_url FOUND within site_url')
        else:
            print('file_url NOT FOUND within site_url')

    def download(self):
        """Pull Russell 3000 ETF holdings tickers from online CSV file"""
        site = requests.get(self.file_url)
        print(f'(Status {site.status_code}) Request sent to: {self.file_url}')
        csv_text = re.sub(r'^.*Ticker','Ticker', site.text, flags=re.DOTALL) # Remove start text
        df = pd.read_csv(StringIO(csv_text)) # Convert csv string to DataFrame
        self.df = df.loc[(df['Exchange'].isin(['NASDAQ','New York Stock Exchange Inc.','Nyse Mkt Llc'])) &
                         (df['Location']=='United States') &
                         (df['Market Value'].str.len()>8)
                         ,'Ticker'].reset_index(drop=True) # Filter for tickers of U.S. equities

    def to_csv(self):
        """Export tickers to CSV file"""
        print(f'{len(self.df)} tickers written to \'russell3000.csv\'')
        self.df.to_csv('russell3000.csv', header=False, index=False)

    def to_list(self):
        """Return list of tickers"""
        return self.df.tolist() # Convert Series to list
