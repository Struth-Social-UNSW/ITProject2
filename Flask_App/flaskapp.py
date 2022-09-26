# imports
from re import search
from flask import Flask, render_template, request, redirect, url_for, session
#from Flask_App.machLearn import fakeCloud, realCloud
import tweepy_wrapper
import machLearn_Run

app = Flask(__name__)
app.secret_key = 'struthSocialFakeNewsDetection'
#Clear image cache after 0 seconds, stops images getting stuck/not updating
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


# HOME PAGE routing
@app.route("/")
def home():
    return render_template('home.html')

# RAW TEXT routing
@app.route("/rawText")
def rawText():
    return render_template('rawText.html')

# for passing variables from form to script
@app.route('/rawText', methods=['POST'])
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
    searchButtonClicked = False
    if request.form['Submit'] == 'search':
        searchButtonClicked = True
        searchTopic = request.form['searchTopic']
        tweets = tweepy_wrapper.RecentTweetsWrapper(searchTopic)
        if tweets:
            divShown = True
        return render_template('twitter.html', tweets=tweets, searchTopic=searchTopic, divShown=divShown, searchButtonClicked=searchButtonClicked)
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
    temparr = machLearn_Run.Main(array)
    passedTweets = temparr
    print(temparr)
    return render_template('analysis.html', passedTweets = passedTweets)

# Running app, debug mode can be changed here
if __name__ == '__main__':
    app.run(debug=True)
