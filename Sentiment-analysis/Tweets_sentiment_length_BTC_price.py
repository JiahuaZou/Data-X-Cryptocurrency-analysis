#!/usr/bin/env python
# coding: utf-8

# # Data-X Project

# In[247]:


import csv
import re
import numpy as np
import pandas as pd 
import seaborn as sns
import spacy
import nltk
import statsmodels.api as sm
nltk.download('wordnet')
lemmatizer = nltk.stem.WordNetLemmatizer()
ps = nltk.PorterStemmer()
import matplotlib.pyplot as plt
from matplotlib import pyplot
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# In[3]:


get_ipython().system(' pip install vaderSentiment')


# ## Sentiment Analysis for Bitcoin Hashtag on Twitter

# ### 1.Read the files into pandas dataframe 
# 
# 

# The original data:
# 
# Tweets 1 consists of all the cryptocurrency related tweets from Oct 20 to Oct 27, 2018. 
# 
# Tweets 2 consists of all the cryptocurrency related tweets from Nov 1 to Nov 9, 2018

# In[154]:


tweets1 = pd.read_csv("bitcoin.csv")[['Author', 'Text', 'Retweet_Count', 'Timestamp']]
tweets1.head()


# In[155]:


tweets2 = pd.read_csv('bitcointweets.csv')[['Author', 'Text', 'Retweet_Count', 'Timestamp']]
tweets2.head()


# ### 2. Data cleaning function: 
# Remove urls, mentions, punctuations and stopwords

# In[8]:


class CleanText():
    
    def __init__(self, input_text):
        self.input_text = input_text
    
    def remove(self):
        remove_mention = re.sub(r'@\w+', '', self.input_text)
        remove_url = re.sub(r'http.?://[^\s]+[\s]?', '', remove_mention)
        # By compressing the underscore, the emoji is kept as one word
        remove_emoji = remove_url.replace('_','')
        remove_punctuation = re.sub('[^A-Za-z0-9_\s]', '', remove_emoji)
        lowercase = remove_punctuation.lower()
        remove_n = re.sub('[\n\r]', '', lowercase)
        remove_num = re.sub('[[:digit:]]', '', remove_n)
        
        return remove_num.replace('rt', '')


# In[114]:


clean_tweet1 = []
for tweet in tweets1['Text']:
    clean_tweet1.append(CleanText(tweet).remove())
    
clean_tweets1 = tweets1.drop(['Text'], axis = 1)
clean_tweets1['Clean_Text'] = clean_tweet1
clean_tweets1.head()

clean_tweet2 = []
for tweet in tweets2['Text']:
    clean_tweet2.append(CleanText(tweet).remove())
    
clean_tweets2 = tweets2.drop(['Text'], axis = 1)
clean_tweets2['Clean_Text'] = clean_tweet2
clean_tweets2 = clean_tweets2[49:] #only selecting a one-week period: Nov 1 - 8.
clean_tweets2.head()


# ### 3. Sentiment analysis
# 
# We will look at the compound score for each sentence.

# In[116]:


analyzer = SentimentIntensityAnalyzer()

score1 = []
for sentence in clean_tweets1['Clean_Text']:
    score1.append(analyzer.polarity_scores(sentence))

sentiment1 = []
for each in score1:
    sentiment1.append(each['compound'])
clean_tweets1['compound'] = sentiment1

score2 = []
for sentence in clean_tweets2['Clean_Text']:
    score2.append(analyzer.polarity_scores(sentence))

sentiment2 = []
for each in score2:
    sentiment2.append(each['compound'])
clean_tweets2['compound'] = sentiment2


# The threshold for positive comment is >= 0.05, for neutral comment is > - 0.05 and for negative comment is <= -0.05.
# 
# We will classify the tweets using this threshold. 

# In[117]:


comment_class = []
for score in clean_tweets1['compound']:
    if score <= -0.05:
        comment_class.append('Negative')
    elif score >= 0.05:
        comment_class.append('Positive')
    else:
        comment_class.append('Neutral')
clean_tweets1['type'] = comment_class

comment_class = []
for score in clean_tweets2['compound']:
    if score <= -0.05:
        comment_class.append('Negative')
    elif score >= 0.05:
        comment_class.append('Positive')
    else:
        comment_class.append('Neutral')
clean_tweets2['type'] = comment_class


# ### 4. Compute length of each tweets and write both of them to csv file

# In[118]:


clean_tweets1['text_len'] = [len(text) for text in clean_tweets1['Clean_Text']]
x1 = clean_tweets1[['compound', 'text_len']]
x1.to_csv('x1.csv')
clean_tweets2['text_len'] = [len(text) for text in clean_tweets2['Clean_Text']]
x2 = clean_tweets2[['compound', 'text_len']]
x2.to_csv('x2.csv')


# ### *Since R studio has better packages for statistical analysis, I will perform the hypothesis testing part of my analysis in R and please look at the R file for reference. 

# ### 5. Statistical  Facts

# In[122]:


comp_mean1 = np.mean(clean_tweets1['compound'])
print('Average Compound Score, Oct 20-27: ', comp_mean1)
comp_mean2 = np.mean(clean_tweets2['compound'])
print('Average Compound Score, Nov 1-9: ', comp_mean2)
print('On average period x2 slightly more positive than the previous time period')


# In[123]:


sd1 = np.std(clean_tweets1['compound'])
print('Standard deviation for compound score, Oct 20-27: ', sd1)
sd2 = np.std(clean_tweets2['compound'])
print('Standard deviation for compound score, Nov 1-9: ', sd2)
print('Period 2 had more spread out compound score than that in period 1')


# In[124]:


len_mean1 = np.mean(clean_tweets1['text_len'])
print('Average Tweet length, Oct 20-27: ', len_mean1)
len_mean2 = np.mean(clean_tweets2['text_len'])
print('Average Tweet length, Nov 1-9: ', len_mean2)
print('On average period 2 slightly had more lengthy tweets than the period 1 had')


# In[125]:


sd_mean1 = np.std(clean_tweets1['text_len'])
print('Standard deviation for Tweet length, Oct 20-27: ', sd_mean1)
sd_mean2 = np.std(clean_tweets2['text_len'])
print('Standard deviation for Tweet length, Nov 1-9: ', sd_mean2)
print('Period 2 had more spread out Tweet length than that in period 1')


# Just by looking at these stats, we observe that generally period 2 has more positive compound sentiment tweets and more length tweets, and the distribution of its compound sentiment score and length tweets has more variability. 

# In[126]:


X1 = clean_tweets1[['Timestamp', 'compound', 'text_len']]
clean_time = []
for time in clean_tweets1['Timestamp']:
    clean_time += re.findall('[0-9]{4}\-[0-9]{2}\-[0-9]{2}', time)

X1['clean_time'] = clean_time
X1

final = []
for time in X1['clean_time']:
    final += re.findall('[0-9]{2}$', time)
X1['time'] = pd.to_numeric(final)

X2 = clean_tweets2[['Timestamp', 'compound', 'text_len']]
clean_time = []
for time in clean_tweets2['Timestamp']:
    clean_time += re.findall('[0-9]{4}\-[0-9]{2}\-[0-9]{2}', time)

X2['clean_time'] = clean_time
X2

final = []
for time in X2['clean_time']:
    final += re.findall('[0-9]{2}$', time)
X2['time'] = pd.to_numeric(final)


# In[127]:


aggregate_X1 = X1.groupby('time').mean().reset_index()
aggregate_X1


# In[169]:


aggregate_X2 = X2.groupby('time').mean().reset_index()
aggregate_X2.to_csv('aggregate_x2.txt')


# ### Analysis of average compound sentiment score and tweet length by day

# In[158]:


aggregate_X1.plot.line(x = 'time', y = 'compound') #average sentiment compound
aggregate_X1.plot.line(x = 'time', y = 'text_len') #average tweet length


# Referring to the linear regression analysis on compound sentiment score and tweet length, we have the slope coefficient as 0.032, and the corresponding p value is 0.14, which is too large to be used to reject the null. In this case, we can't say anything about the relationship between compound sentiment score and tweet length from Oct 20 to 27. 

# In[159]:


aggregate_X2.plot.line(x = 'time', y = 'compound') #average sentiment compound
aggregate_X2.plot.line(x = 'time', y = 'text_len') #average tweet length


# According to the linear regression outcomes in R, we've found that a linear relationship can be used to describe the association between tweet length and compound sentiment score. The slope coefficient is 0.0205, the p-value is 0.011, the p-value for the F-statistic is 0.011. All of these statistics are small enough to prove that there is linearity between the length of tweets and compound sentiment score during Nov 1 and 8, that is, the more people talked about bitcoin in a single tweet is associated with the more positive the sentiment towards bitcoin.

# So this leads us to ponder: what is the reason behind such linearity between tweet length and compound sentiment score? 
# 
# Was there any external events that impacted people's tweeting behavior and attitude towards bitcoin?
# 
# ##### It turned out that Nov 2nd was the 10-year-old anniversary of the creation of bitcoin. 
# ##### So we will definitely need external data to analyze this!

# Moving further, I collected the "BTC price index" data in these two periods to delve more deeply into the causation behind the linear pattern

# ### BTC Price Index vs. Sentiment score in Period  2

# In[243]:


price_data = pd.read_csv('bitcoin_full.csv')
price_X2 = price_data[['date', 'high', 'low','opens','close','volume']][18:26]
price_X2['day'] = [8,7,6,5,4,3,2,1]
price_X2 = price_X2.sort_values('day', ascending = True)
price_X2


# Next we will plot a side by side line graph which consist of three subgraphs: average compound sentiment score of tweets in period 2, average tweet length, and the last one is one of the attributes in the 'price_X2' data frame - high, low, opens, close and volume

# ### High 

# In[241]:


fig, ax =plt.subplots(1,3)
fig.set_size_inches(15.5, 4.5)
aggregate_X2.plot.line(x = 'time', y = 'compound',ax=ax[0]) 
aggregate_X2.plot.line(x = 'time', y = 'text_len',ax=ax[1])
price_X2.plot.line(x = 'day', y = 'high',ax=ax[2])

plt.figure();


# In[265]:


X = list(aggregate_X2['compound'])
y = list(price_X2['high'])

model = sm.OLS(y, X).fit()
model.summary()


# In[262]:


X = list(aggregate_X2['text_len'])
y = list(price_X2['high'])

model = sm.OLS(y, X).fit()

model.summary()


# ### Low

# In[242]:


fig, ax =plt.subplots(1,3)
fig.set_size_inches(15.5, 4.5)
aggregate_X2.plot.line(x = 'time', y = 'compound',ax=ax[0]) 
aggregate_X2.plot.line(x = 'time', y = 'text_len',ax=ax[1])
price_X2.plot.line(x = 'day', y = 'low',ax=ax[2])

plt.figure();


# In[264]:


X = list(aggregate_X2['compound'])
y = list(price_X2['low'])

model = sm.OLS(y, X).fit()
model.summary()


# In[266]:


X = list(aggregate_X2['text_len'])
y = list(price_X2['low'])

model = sm.OLS(y, X).fit()
model.summary()


# ### Opens

# In[244]:


fig, ax =plt.subplots(1,3)
fig.set_size_inches(15.5, 4.5)
aggregate_X2.plot.line(x = 'time', y = 'compound',ax=ax[0]) 
aggregate_X2.plot.line(x = 'time', y = 'text_len',ax=ax[1])
price_X2.plot.line(x = 'day', y = 'opens',ax=ax[2])

plt.figure();


# In[267]:


X = list(aggregate_X2['compound'])
y = list(price_X2['opens'])

model = sm.OLS(y, X).fit()
model.summary()


# In[268]:


X = list(aggregate_X2['text_len'])
y = list(price_X2['opens'])

model = sm.OLS(y, X).fit()
model.summary()


# ### Close

# In[245]:


fig, ax =plt.subplots(1,3)
fig.set_size_inches(15.5, 4.5)
aggregate_X2.plot.line(x = 'time', y = 'compound',ax=ax[0]) 
aggregate_X2.plot.line(x = 'time', y = 'text_len',ax=ax[1])
price_X2.plot.line(x = 'day', y = 'close',ax=ax[2])

plt.figure();


# In[269]:


X = list(aggregate_X2['compound'])
y = list(price_X2['close'])

model = sm.OLS(y, X).fit()
model.summary()


# In[270]:


X = list(aggregate_X2['text_len'])
y = list(price_X2['close'])

model = sm.OLS(y, X).fit()
model.summary()


# ### Volume

# In[246]:


fig, ax =plt.subplots(1,3)
fig.set_size_inches(15.5, 4.5)
aggregate_X2.plot.line(x = 'time', y = 'compound',ax=ax[0]) 
aggregate_X2.plot.line(x = 'time', y = 'text_len',ax=ax[1])
price_X2.plot.line(x = 'day', y = 'volume',ax=ax[2])

plt.figure();


# In[271]:


X = list(aggregate_X2['compound'])
y = list(price_X2['volume'])

model = sm.OLS(y, X).fit()
model.summary()


# In[272]:


X = list(aggregate_X2['text_len'])
y = list(price_X2['volume'])

model = sm.OLS(y, X).fit()
model.summary()


# In[274]:


print(' P>|t| are all equal to 0.05 in the above linear regression model. If we choose an alpha of 0.05, we can reject the null hypothesis and say that the coefficient is significantly different from 0')


# In[ ]:




