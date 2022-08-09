# imports
from flask import Flask, render_template, request
import testcase
import tweepy_wrapper

app = Flask(__name__)


# HOME PAGE routing
@app.route("/home")
def home():
    return render_template('home.html')


# for passing variables from form to script
@app.route('/home', methods=['POST'])
def webapp():
    searchInput = request.form['searchInput']
    prediction = testcase.test(searchInput)
    return render_template('home.html', searchInput=searchInput, result=prediction)


# ABOUT PAGE routing
@app.route("/about")
def about():
    return render_template('about.html')


# TWITTER PAGE routing
@app.route("/twitter")
def twitter():
    return render_template('twitter.html')


# for passing variables from form to script
@app.route("/twitter", methods=['POST'])
def twitterCall():
    searchTopic = request.form['searchTopic']
    tweets = tweepy_wrapper.RecentTweetsWrapper(searchTopic)
    return render_template('twitter.html', tweets=tweets, searchTopic=searchTopic)


#for analysis ***CURRENTLY NOT WORKING***
@app.route("/twitter", methods=['GET'])
def analyseCall():
    if request.method == 'POST':
        if request.form['analyse'] == "Analyse":
            for checkbox in request.form.getlist('check'):
                print(checkbox)
    # return render_template('twitter.html')


# Running app, debug mode can be changed here
if __name__ == '__main__':
    app.run(debug=True)
