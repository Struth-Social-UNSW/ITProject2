from flask import Flask, render_template, request

import testcase
import tweepy_wrapper

app = Flask(__name__)




# HOME PAGE
@app.route("/home")
def home():
    return render_template('home.html')


@app.route('/home', methods=['POST'])
def webapp():
    searchInput = request.form['searchInput']
    prediction = testcase.test(searchInput)
    return render_template('home.html', searchInput=searchInput, result=prediction)


# ABOUT PAGE
@app.route("/about")
def about():
    return render_template('about.html')


# TWITTER PAGE
@app.route("/twitter")
def twitter():
    return render_template('twitter.html')


@app.route("/twitter", methods=['POST'])
def twitterCall():
    searchTopic = request.form['searchTopic']
    tweets = tweepy_wrapper.RecentTweetsWrapper(searchTopic)
    return render_template('twitter.html', tweets=tweets, searchTopic=searchTopic)




if __name__ == '__main__':
    app.run(debug=True)
