# imports
from re import search
from flask import Flask, render_template, request, redirect, url_for, session
import tweepy_wrapper
import machLearn

app = Flask(__name__)
app.secret_key = 'struthSocialFakeNewsDetection'


# HOME PAGE routing
@app.route("/")
def home():
    return render_template('home.html')


# for passing variables from form to script
@app.route('/', methods=['POST'])
def webapp():
    searchInput = []
    searchInput.append(request.form['searchInput'])
    session['selectedTweets'] = searchInput
    return redirect("analysis", code=302)


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
def handle():
    divShown = False
    if request.form['Submit'] == 'search':
        searchTopic = request.form['searchTopic']
        tweets = tweepy_wrapper.RecentTweetsWrapper(searchTopic)
        divShown = True
        return render_template('twitter.html', tweets=tweets, searchTopic=searchTopic, divShown=divShown)
    elif request.form['Submit'] == 'analyse':
        session['selectedTweets'] = request.form.getlist('tweet')
        for checkbox in request.form.getlist('tweet'):
            print(checkbox)   
        return redirect("analysis", code=302)

# Analysis PAGE routing
@app.route("/analysis")
def analysis():
    array=session.get('selectedTweets', None)
    passedTweets = []
    temparr = machLearn.Main(array)
    passedTweets = temparr
    print(temparr)
    return render_template('analysis.html', passedTweets = passedTweets)

# Running app, debug mode can be changed here
if __name__ == '__main__':
    app.run(debug=True)
