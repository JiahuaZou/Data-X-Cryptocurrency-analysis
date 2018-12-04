import requests
import bs4 as bs
import re
import pandas as pd
from flask_table import Table, Col
import wget
import tweepy
from tweepy import TweepError
import json
from datetime import timedelta, datetime, timezone, date
import numpy as np
import sklearn
import nltk
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import string
import flask
from flask import Response, request, send_file, Flask, render_template, session, redirect

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

def all_names():
    website = requests.get('https://coinmarketcap.com/')
    soup = bs.BeautifulSoup(website.content, features='html.parser')
    names = soup.find_all(class_='currency-name-container link-secondary')
    names = [i.text for i in names]
    return names

def reset():
    indicators = ['ptable', 'rforest', 'time']
    for i in indicators:
        session[i] = False

def jerry_learn():
    key_file = 'keys.json'
    with open(key_file) as f:
        keys = json.load(f)
    auth = tweepy.OAuthHandler(keys["consumer_key"], keys["consumer_secret"])
    auth.set_access_token(keys["access_token"], keys["access_token_secret"])
    api = tweepy.API(auth, wait_on_rate_limit=True)
    today = date.today()
    today = datetime(today.year, today.month, today.day)
    week_ago = today - timedelta(days=1)
    start = week_ago.strftime('%Y-%m-%d %H:%M:%S')[0:10]
    timestamp = []
    user = []
    text = []
    retweet_count = []
    i = 0
    for tweet in tweepy.Cursor(api.search, q = '#bitcoin', lang="en", since = start).items():
        i += 1
        timestamp.append(tweet.created_at)
        retweet_count.append(tweet.retweet_count)
        text.append(tweet.text)
        user.append(tweet.user.screen_name)
        if i > 1500:
            break
    start2 = int(round(timestamp[-1].replace(tzinfo=timezone.utc).timestamp()))
    rawlink = "http://api.bitcoincharts.com/v1/trades.csv?symbol=bitstampUSD"
    link = rawlink + "&start=" + str(int(round(start2)))
    filename = wget.download(link)
    btcprice = pd.read_csv(filename, header = None)
    btcprice.columns = ['unixtime', 'price', 'amount']
    converted_time = btcprice['unixtime'].apply(lambda x: datetime.utcfromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))
    d = {'timestamp': timestamp, 'user': user, 'text' : text, 'retweet' : retweet_count}
    df = pd.DataFrame(data = d)
    df.to_csv("most_recent_tweet.csv")
    btcprice['timestamp'] = converted_time
    btcprice2 = btcprice.iloc[::50, :].reset_index()
    del btcprice2['index']
    df2 = df.iloc[::-1].reset_index()
    del df2['index']
    btcprice2['timestamp'] = btcprice2['timestamp'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    def cal_direction(array):
        direction = np.ones(len(array))
        for i in range(len(array) - 1):
            if array[i + 1] - array[i] < 0:
                direction[i + 1] = 0
        return(direction)
    btcprice2 = btcprice2.assign(direction = cal_direction(btcprice2['price'].values))
    direction_tweet = np.zeros(len(df2))
    for x in range(len(df2)):
        for y in range(len(btcprice2)):
            if (btcprice2.loc[y, 'timestamp'] > df2.loc[x, 'timestamp']):
                direction_tweet[x] = btcprice2.loc[y, 'direction']
                break
    stopwords = nltk.corpus.stopwords.words('english')
    ps = nltk.PorterStemmer()
    def clean_text(text):
        text = "".join([word.lower() for word in text if word not in string.punctuation])
        tokens = re.split('\W+', text)
        text = [ps.stem(word) for word in tokens if word not in stopwords]
        return text
    tfidf_vec = TfidfVectorizer(analyzer=clean_text)
    x_tfidf = tfidf_vec.fit_transform(df2['text'])
    x_tfidf.columns = tfidf_vec.get_feature_names()
    x_counts_tfidf = pd.DataFrame(x_tfidf.toarray())
    x_feature = pd.concat([df2[['retweet']], x_counts_tfidf], axis = 1)
    x_feature2 = x_feature.loc[:int(round(0.8*len(x_feature)))-1, :]
    direction_tweet2 = direction_tweet[:int(round(0.8*len(direction_tweet)))]
    x_est = x_feature.loc[int(round(0.8*len(x_feature))):, :]
    train_size = int(round(0.8*len(x_feature2)))
    x_train = x_feature2.loc[:train_size-1, :]
    x_test = x_feature2.loc[train_size:, :]
    y_train = direction_tweet2[:train_size]
    y_test = direction_tweet2[train_size:]
    rf = RandomForestClassifier(n_estimators=50, max_depth=20, n_jobs=-1)
    rf_model = rf.fit(x_train, y_train)
    y_pred = rf_model.predict(x_test)
    label = None
    if sum(y_pred == 0) >= sum(y_pred == 1):
        label = 0
    else:
        lebel = 1
    precision, recall, fscore, support = score(y_test, y_pred, pos_label= label, average='binary')
    val1 = 'Precision: {} / Recall: {} / Accuracy: {}'.format(round(precision, 3),
                                                            round(recall, 3),
                                                            round((y_pred==y_test).sum() / len(y_pred),3))
    y_est = rf_model.predict(x_est)
    p1 = sum(y_est == 1)
    p0 = sum(y_est == 0)
    val2 = None
    if p1 > p0:
        val2 = "The random forest model detects an upward trend based on conversations on tweet with a probability of " + str(p1/len(y_est))
    else:
        val2 = "The random forest model detects an downward trend based on conversations on tweet with a probability of " + str(p0/len(y_est))
    return val1, val2

def download_crypto(name):
    templink = 'https://coinmarketcap.com/currencies/' + name + '/historical-data/'
    website = requests.get(templink)
    new_soup = bs.BeautifulSoup(website.content, features='html.parser')
    name = re.findall('currencies/(.*?)/', templink)[0]
    dates=[i.text for i in new_soup.find_all('td', class_='text-left')]
    opens = new_soup.find_all("td")[1::7]
    high = new_soup.find_all("td")[2::7]
    low = new_soup.find_all("td")[3::7]
    close = new_soup.find_all("td")[4::7]
    volume = new_soup.find_all("td")[5::7]
    market_cap = new_soup.find_all("td")[6::7]
    tbl = pd.DataFrame()
    tbl['name'] = [name for i in range(len(opens))]
    tbl['date'] = dates
    tbl['opens'] = [float(re.sub(',', '', i.text)) for i in opens]
    tbl['high'] = [float(re.sub(',', '', i.text)) for i in high]
    tbl['low'] = [float(re.sub(',', '', i.text)) for i in low]
    tbl['close'] = [float(re.sub(',', '', i.text)) for i in close]
    tbl['volume'] = [float(re.sub(',', '', i.text)) for i in volume]
    try:
        tbl['market_cap'] = [float(re.sub(',', '', i.text)) for i in market_cap]
    except:
        tbl['market_cap'] = [np.nan for i in market_cap]
    return tbl

def time_series():
    import matplotlib.pyplot as plt
    import seaborn as sns
    import datetime
    import sklearn
    plt.style.use('fivethirtyeight')
    plt.rcParams["figure.figsize"] = (15,7)
    from statsmodels.tsa.arima_model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.seasonal import seasonal_decompose
    from scipy import stats
    import statsmodels.api as sm
    from itertools import product
    import warnings
    warnings.filterwarnings('ignore')
    df = pd.read_csv('bitcoin_ytd.csv')
    df = df.iloc[: ,2:9]
    btc_close = df['Close']
    plot1 = btc_close.plot(lw=2.5, figsize=(12, 5))
    rroll_d3 = btc_close.rolling(window=3).mean()
    rroll_d7 = btc_close.rolling(window=7).mean()
    rroll_d14 = btc_close.rolling(window=14).mean()
    plt.figure(figsize=(14,7))
    plt.plot(btc_close, alpha=0.8,label='Original observations')
    plt.plot(rroll_d3, lw=3, alpha=0.8,label='Rolling mean (window 3)')
    plt.plot(rroll_d7, lw=3, alpha=0.8,label='Rolling mean (window 7)')
    plt.plot(rroll_d14, lw=3, alpha=0.8,label='Rolling mean (window 14)')
    plt.title('BTC-USD Close Price 30days')
    plt.tick_params(labelsize=12)
    plt.legend(loc='upper left', fontsize=12)
    plt.savefig("static/plot2.png")
    short_window = 3
    mid_window = 10
    signals = pd.DataFrame(index=btc_close.index)
    signals['signal'] = 0.0
    roll_d3 = btc_close.rolling(window=short_window).mean()
    roll_d10 = btc_close.rolling(window=mid_window).mean()
    signals['short_mavg'] = roll_d3
    signals['mid_mavg'] = roll_d10
    signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['mid_mavg'][short_window:], 1.0, 0.0)
    signals['positions'] = signals['signal'].diff()
    plt.figure(figsize=(14, 7))
    plt.plot( btc_close, lw=3, alpha=0.8,label='Original observations')
    plt.plot(roll_d3, lw=3, alpha=0.8,label='Rolling mean (window 3)')
    plt.plot( roll_d10, lw=3, alpha=0.8,label='Rolling mean (window 10)')
    plt.plot(signals.loc[signals.positions == 1.0].index,
             signals.short_mavg[signals.positions == 1.0],
             '^', markersize=10, color='r', label='buy')
    plt.plot(signals.loc[signals.positions == -1.0].index,
             signals.short_mavg[signals.positions == -1.0],
             'v', markersize=10, color='k', label='sell')
    plt.title('BTC-USD Adj Close Price (The Technical Approach)')
    plt.tick_params(labelsize=12)
    plt.legend(loc='upper left', fontsize=12)
    plt.savefig("static/plot3.png")
    fig = plot1.get_figure()
    fig.savefig("static/plot1.png")
