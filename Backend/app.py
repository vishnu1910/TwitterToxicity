from flask import Flask, flash, request, redirect, url_for
from flask_cors import CORS
import re
import pickle
import pandas as pd
import tensorflow as tf
from bs4 import BeautifulSoup
from ps import PorterStemmer
ps = PorterStemmer()
# from selenium.webdriver import Chrome
# driver = Chrome("D:/vishn/chromedriver_win32/chromedriver.exe")

import toxic_comments.toxic_predict as tc

model = tf.keras.models.load_model('model_weights')
vectorizer = pickle.load(open("vectorizer.pkl", "rb" ))
stopwords = []
with open('stopwords.txt', 'r') as f:
    for line in f:
        stopwords.append(line.strip())
        
def tweet_splitter(tweet_part):
    # print("TWEET PART ======================", tweet_part)
    # print("hello 1")
    _, tweet_section = tweet_part.findChildren("div", recursive = False)
    # print("hello 2")
    autos = tweet_section.find_all("div", {"dir":"auto"})
    # print("hello 3")
    spans = [div.find("span", {"class":"r-poiln3"}) for div in autos]
    # print("hello 4")
    spans = [str(span.text) for span in spans if span and '·' != str(span.text)]
    # print("hello 5", spans)
    content = ""
    # time = username = userid = replied_to = content = None
    # time = tweet_part.find('time')
    # if time:
    #     time = time['datetime']
    # else:
    #     time = None
    # if len(spans) == 2:
    #     username, content = spans
    # elif len(spans)==3:
    #     username, replied_to, content = spans
    # userid = tweet_section.find('div', {"dir":"ltr"}).find('span', {"class":"r-poiln3"}).text
    for i in range(len(spans)):
        content= content+ spans[i] +" "
    print("CONTENT: ", content)
    return content
    
def util_printer(time, username, userid, replied_to, content):
    print("time : ", time)
    print("username : ", username)
    print("userid : ", userid)
    print("replied_to : ", replied_to)
    print("content : ", content)
        
def predict(content):
    print(content)
    if not content:
        return 0
    def stemmer(input_):
        review = re.sub('[^a-zA-Z]',' ',input_)
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if not word in stopwords]
        review = ' '.join(review)
        return review

    text = content
    stemmed_text = pd.Series(stemmer(text))
    vectorized_text = vectorizer.transform(stemmed_text)
    result = model.predict(vectorized_text)
    return round(float(result))

app = Flask(__name__)
CORS(app)

@app.route('/_api_call', methods=['POST'])
def toxicity_detector():
    data = request.get_data()
    data = data.decode("utf-8")
    # print(data)
    
    # driver.get("https://twitter.com/search?q=news&src=typed_query")
    soup = BeautifulSoup(data, "html.parser")
    with open('file.html', 'w', encoding = "utf-8") as f:
        f.write(soup.prettify())
    tweet_part = soup.find("div",{"class": "css-1dbjc4n r-1iusvr4 r-16y2uox r-1777fci r-kzbkwu"})
    #tweet_part = driver.find_element_by_xpath('//[data-testid="tweet"]')
    #print(tweet_part)
    if not tweet_part:
        print("No tweet part")
        return "false"
    content = tweet_splitter(tweet_part)
    
    ans = ""
    # if predict(content) == 1:
    #     ans += "fake"
    # else:
    #     ans += "not-fake"
    toxicity = tc.predict(content)
    if toxicity:
        ans = ans + "|" + toxicity
    return ans