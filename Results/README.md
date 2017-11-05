
## Unit 7 | Assignment - Distinguishing Sentiments

## Background

__Twitter__ has become a wildly sprawling jungle of information&mdash;140 characters at a time. Somewhere between 350 million and 500 million tweets are estimated to be sent out _per day_. With such an explosion of data, on Twitter and elsewhere, it becomes more important than ever to tame it in some way, to concisely capture the essence of the data.

Choose __one__ of the following two assignments, in which you will do just that. Good luck!

    √  ## News Mood

    √  In this assignment, you'll create a Python script to perform a sentiment analysis of the Twitter activity of various news oulets, and to present your findings visually.

    √Your final output should provide a visualized summary of the sentiments expressed in Tweets sent out by the following news organizations: __BBC, CBS, CNN, Fox, and New York times__.


The first plot will be and/or feature the following:

    √* Be a scatter plot of sentiments of the last __100__ tweets sent out by each news
    organization, ranging from -1.0 to 1.0, where a score of 0 expresses a neutral
    sentiment, -1 the most negative sentiment possible, and +1 the most positive sentiment
    possible.
    √* Each plot point will reflect the _compound_ sentiment of a tweet.
    √* Sort each plot point by its relative timestamp.

    √The second plot will be a bar plot visualizing the _overall_ sentiments of the last 100
    tweets from each organization. For this plot, you will again aggregate the compound
    sentiments analyzed by VADER.

The tools of the trade you will need for your task as a data analyst include the following: tweepy, pandas, matplotlib, seaborn, textblob, and VADER.

Your final Jupyter notebook must:

    √* Pull last 100 tweets from each outlet.
    √* Perform a sentiment analysis with the compound, positive, neutral, and negative
    scoring for each tweet. 
    √* Pull into a DataFrame the tweet's source acount, its text, its date, and its
    compound, positive, neutral, and negative sentiment scores.
    √* Export the data in the DataFrame into a CSV file.
    √* Save PNG images for each plot.

As final considerations:

    √* Use the Matplotlib and Seaborn libraries.
    √* Include a written description of three observable trends based on the data. 
    √* Include proper labeling of your plots, including plot titles (with date of analysis)
    and axes labels.
* Include an exported markdown version of your Notebook called  `README.md` in your GitHub repository.  

Written Description of three observable trends:

1 - Tweeting appears to be the new way of sending "Flash Headlines", presumably in an
    effort to entice people to either tune-in or purchase paper.  Reflects today's society 
    with respect to fast-pace, instant blurb, without taking the time to read and discover.
    This observation is based upon "News Agencies" that "Tweet" subject headlines every so
    many seconds.
2 - Totally surprised at only one news agency that appeared to have a more routinely
    neutral bias in their tweets (i.e., CBS News).
3 - Additional surprise with respect to the generally positive bias.  However, this would
    need continued analysis.  Hypothesis being that on Weekends and Sundays in particular,
    there would be less negativity in tweets.


```python
# Dependencies
import numpy as np
import pandas as pd
import json
import tweepy
import matplotlib.pyplot as plt
import seaborn as sns
import time
```


```python
# Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
```


```python
#Keys
consumer_key = "Ed4RNulN1lp7AbOooHa9STCoU"
consumer_secret = "P7cUJlmJZq0VaCY0Jg7COliwQqzK0qYEyUF9Y0idx4ujb3ZlW5"
access_token = "839621358724198402-dzdOsx2WWHrSuBwyNUiqSEnTivHozAZ"
access_token_secret = "dCZ80uNRbFDjxdU2EckmNiSckdoATach6Q8zb7YYYE5ER"
```


```python
# Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())
```


```python
# Target list
target_list = ["@BBCnews", "@CBSnews", "@CNN", "@FoxNews", "@nytimes"]
```


```python
# Lists to hold sentiments
compound_list = []
positive_list = []
negative_list = []
neutral_list = []
sentiment=[]
```


```python
for agency in target_list:
    testn=api.search(agency,count=1, result_type="recent")
    #print(testn["statuses"][0]["text"])
#len(testn)
```


```python
for i in range (5):
    count=1
    for j in range (100):
        #print (count)
        count = count +1
```


```python

for agency in target_list:
    test1=api.search(agency,count=100, result_type="recent") 
    counter=1
    for twt in test1["statuses"]:
        #print(twt)
        temp_text=(twt["text"])
        date=(twt["created_at"])
        compound = analyzer.polarity_scores(twt["text"])["compound"]
        pos = analyzer.polarity_scores(twt["text"])["pos"]
        neu = analyzer.polarity_scores(twt["text"])["neu"]
        neg = analyzer.polarity_scores(twt["text"])["neg"]
        compound_list.append(compound)
        positive_list.append(pos)
        negative_list.append(neg)
        neutral_list.append(neu)
        sentiment.append({"Agency": agency,"Compound": compound, "Positive": 
                          pos,"Negative": neg,"Neutral": neu, "Tweets Ago":counter,
                          "Date":date, "Text":temp_text})
        counter= counter + 1
```


```python
# Convert sentiment to a DataFrame
sentiments_pd = pd.DataFrame.from_dict(sentiment)
sentiments_pd.head()

```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Agency</th>
      <th>Compound</th>
      <th>Date</th>
      <th>Negative</th>
      <th>Neutral</th>
      <th>Positive</th>
      <th>Text</th>
      <th>Tweets Ago</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>@BBCnews</td>
      <td>0.5994</td>
      <td>Sun Nov 05 18:51:18 +0000 2017</td>
      <td>0.090</td>
      <td>0.621</td>
      <td>0.290</td>
      <td>RT @BBCNews: Paradise Papers: Tax haven secret...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>@BBCnews</td>
      <td>0.8176</td>
      <td>Sun Nov 05 18:51:04 +0000 2017</td>
      <td>0.000</td>
      <td>0.605</td>
      <td>0.395</td>
      <td>UK's Lord Ashcroft stayed a non-dom https://t....</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>@BBCnews</td>
      <td>-0.7003</td>
      <td>Sun Nov 05 18:51:03 +0000 2017</td>
      <td>0.298</td>
      <td>0.702</td>
      <td>0.000</td>
      <td>@Ross__Mackenzie @hhurleylol @BBCNews "No evid...</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>@BBCnews</td>
      <td>0.0000</td>
      <td>Sun Nov 05 18:50:58 +0000 2017</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>RT @BBCNews: Live #ParadisePapers updates: htt...</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>@BBCnews</td>
      <td>0.0000</td>
      <td>Sun Nov 05 18:50:54 +0000 2017</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>RT @evertonfc2: Lord Ashcroft -The story of th...</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
sentiments_pd.to_csv("Results/tweet_info.csv",encoding="utf-8",index=False,header=True)
```


```python

for agency in target_list:
    temp_name= str(agency)+".png"
    temp_tweetr=sentiments_pd.loc[(sentiments_pd["Agency"]==agency),:]
    plt.plot(np.arange(len(temp_tweetr["Compound"])),
             temp_tweetr["Compound"], marker="o", linewidth=0.5,
             alpha=0.75)

    # Incorporate graph properties
    plt.title("Sentiment Analysis of Tweets (%s) for %s" % (time.strftime("%x"), agency))
    plt.ylabel("Tweet Polarity")
    plt.xlabel("Tweets Ago")
    plt.savefig(temp_name)
    plt.show()
```


![png](output_13_0.png)



![png](output_13_1.png)



![png](output_13_2.png)



![png](output_13_3.png)



![png](output_13_4.png)



```python
news_agency_bar=sentiments_pd[["Agency","Compound"]]
news_agency_bar=news_agency_bar.rename(columns={"Agency":"News Agency"})
news_agency_bar.head()
                               
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>News Agency</th>
      <th>Compound</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>@BBCnews</td>
      <td>0.5994</td>
    </tr>
    <tr>
      <th>1</th>
      <td>@BBCnews</td>
      <td>0.8176</td>
    </tr>
    <tr>
      <th>2</th>
      <td>@BBCnews</td>
      <td>-0.7003</td>
    </tr>
    <tr>
      <th>3</th>
      <td>@BBCnews</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>@BBCnews</td>
      <td>0.0000</td>
    </tr>
  </tbody>
</table>
</div>




```python
nab=news_agency_bar.groupby(["News Agency"])
agency_bar_sentiment=nab.mean()
agency_bar_sentiment.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Compound</th>
    </tr>
    <tr>
      <th>News Agency</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>@BBCnews</th>
      <td>-0.021956</td>
    </tr>
    <tr>
      <th>@CBSnews</th>
      <td>0.028365</td>
    </tr>
    <tr>
      <th>@CNN</th>
      <td>-0.103121</td>
    </tr>
    <tr>
      <th>@FoxNews</th>
      <td>0.015441</td>
    </tr>
    <tr>
      <th>@nytimes</th>
      <td>0.156727</td>
    </tr>
  </tbody>
</table>
</div>




```python
abs=agency_bar_sentiment.reset_index()
#abs.head()
```


```python
my_color=["skyblue","g","r","b","gold"]
agency_bar_sentiment.plot(kind="bar", color=my_color,width=1)
plt.title("Overall Media Sentiment Based On Twitter (%s)" % (time.strftime("%x")))
plt.ylabel("Tweet Polarity")
plt.savefig("Overall_Sentiment_Comparison.png")
plt.show()
```


![png](output_17_0.png)



```python

```
