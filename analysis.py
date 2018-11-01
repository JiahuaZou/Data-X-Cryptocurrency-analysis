import requests
import bs4 as bs
import re
import pandas as pd
from flask_table import Table, Col
def general_assessment():
    website = requests.get('https://coinmarketcap.com/')
    soup = bs.BeautifulSoup(website.content, features='html.parser')
    ptable = pd.DataFrame()
    names = soup.find_all(class_='currency-name-container link-secondary')
    ptable['Name'] = [i.text for i in names]
    marketcap = soup.find_all(class_='no-wrap market-cap text-right')
    ptable['marketcap'] = [re.findall('\$(.*)', i.text)[0] for i in marketcap]
    price = soup.find_all(class_='price')
    ptable['price'] = [re.findall('\$(.*)', i.text)[0] for i in price]
    volume = soup.find_all(class_='volume')
    ptable['volume'] = [re.findall('\$(.*)', i.text)[0] for i in volume]
    return ptable
