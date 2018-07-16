import re
import tweepy
from tweepy import OAuthHandler
from textblob import TextBlob
from os import path
from wordcloud import WordCloud

class TwitterClient(object):

    def __init__(self):

        consumer_key = '4ambeecTTMwHJu3JrTiWMzuId'
        consumer_secret = 'YTtC9UL7emmdlAXPfCncgEnNTrY6ghIueYSHMJp7B0oewkB28T'

        access_token = '795625002691391488-JgNFl4tYSDMFhDPLlNAgRI9pz2OqUZs'
        access_token_secret = 'JaL6fePHPQt0fqTIx7OP7Envr92ZWFGNLuBt1yaQY7awH'
 
        try:
            self.auth = OAuthHandler(consumer_key, consumer_secret)
            self.auth.set_access_token(access_token, access_token_secret)
            # object to fetch tweets
            self.api = tweepy.API(self.auth)
        except:
            print("Authentication Failed")
 
    def clean_tweet(self, tweet):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

    def get_tweet_sentiment(self, tweet):
        analysis = TextBlob(self.clean_tweet(tweet))

        if analysis.sentiment.polarity > 0:
            return 'positive'
        elif analysis.sentiment.polarity == 0:
            return 'neutral'
        else:
            return 'negative'
 
    def get_tweets(self, query, count = 10):
        tweets = []
 
        try:
            fetched_tweets = self.api.search(q = query, count = count)
 
            for tweet in fetched_tweets:
                parsed_tweet = {}
 
                parsed_tweet['text'] = tweet.text
                parsed_tweet['sentiment'] = self.get_tweet_sentiment(tweet.text)
 
                if tweet.retweet_count > 0:
                    if parsed_tweet not in tweets:
                        tweets.append(parsed_tweet)
                else:
                    tweets.append(parsed_tweet)
 
            return tweets
 
        except tweepy.TweepError as e:
            print("Error : " + str(e))
 
def main(): 
    api = TwitterClient()

    product_name = input()
    tweets = api.get_tweets(query = product_name, count = 10000)
    
    ptweets = [tweet for tweet in tweets if tweet['sentiment'] == 'positive']
    print("Positive tweets percentage: {} %".format(100*len(ptweets)/len(tweets)))
    ntweets = [tweet for tweet in tweets if tweet['sentiment'] == 'negative']
    print("Negative tweets percentage: {} %".format(100*len(ntweets)/len(tweets)))
    print("Neutral tweets percentage: {} % ".format(100*(len(tweets) - len(ntweets) - len(ptweets))/len(tweets)))
    #print(type(tweets))
    #print(tweets[:3])





    # sentiment pie chart
    import matplotlib.pyplot as plt
    import numpy as np 
    from matplotlib.backends.backend_pdf import PdfPages

    #pp = PdfPages('Sentiment_analysis.pdf')

    x = [len(ptweets), len(ntweets), len(tweets) - len(ptweets) - len(ntweets)]
    labels = ['Positive', 'Negative', 'Neutral']
    explode= [0,0.1,0]
    colors = ['green', 'red', 'blue']
    plt.figure()
    plt.pie(x, labels = labels, explode=explode, colors=colors, autopct='%1.1f%%')

    plt.title('Twitter Sentiment Analysis of '+product_name)
    plt.savefig(product_name+"_pie_chart.png")
    plt.show()
    

    # whole text as a string - to make wordcloud
    all_text = ''
    for x in tweets:
        all_text += x['text']


    #removing STOPWORDS

    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from string import punctuation
    stop_words = set(stopwords.words('english') + [product_name, 'https', "n't", '...','’',"'s", 'I','A','-']+list(punctuation))
     
    word_tokens = word_tokenize(all_text)
     
    #filtered_sentence = [w for w in all_text if not w in stop_words]
     
    filtered_sentence = []
     
    for w in word_tokens:
        if w.lower() not in stop_words:
            filtered_sentence.append(w)

    #print(all_text)
    '''wordcloud = WordCloud().generate(filtered_sentence)

    
    plt.imshow(wordcloud, interpolation = 'bilinear')
    plt.axis('off')

    wordcloud = WordCloud(max_font_size = 40).generate(filtered_sentence)
    plt.figure()
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
    plt.savefig(product_name+"_word_cloud.png")'''

    from collections import Counter

    diff_words = Counter(filtered_sentence)
    diff_words = [(k, diff_words[k]) for k in sorted(diff_words, key=diff_words.get, reverse=True)]
    x_words = [x[0] for x in diff_words]
    y_freq = [x[1] for x in diff_words]
    print(x_words[:15])
    plt.figure()
    plt.xlabel('Top Words')
    plt.ylabel('frequency')

    #plt.xticks(list(range(1,21)), x_words[:20])  
    plt.bar(x_words[:10], y_freq[:10],edgecolor='black', linewidth=1.2)
    plt.title('Twitter Sentiment Analysis of '+product_name)
    plt.savefig(product_name+"_histogram.png")
    plt.show()
    
    plt.close()
    
    '''import seaborn as sns 
    sns.set(color_codes = True)
    plt.xticks(list(range(1,11)), x_words[:10])
    #sns.distplot(y_freq[:10], bins=10)
    sns.distplot(y_freq[:10], hist_kws=dict(edgecolor="k", linewidth=2))
    plt.savefig(product_name+"_seaborn_plot.png")
    plt.show()'''


 
if __name__ == "__main__":
    # calling main function
    main()
