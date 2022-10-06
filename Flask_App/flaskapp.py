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

# ABOUT PAGE routing
@app.route("/about")
def about():
    return render_template('about.html')

###### ML PAGES

# RAW TEXT routing
@app.route("/rawTextML")
def rawTextML():
    return render_template('rawTextML.html')

# for passing variables from form to script
@app.route('/rawTextML', methods=['POST'])
def webappML():
    searchInput = []
    searchInput.append(request.form['searchInput'])
    session['selectedTweets'] = searchInput
    return redirect("analysisML", code=302)

# TWITTER PAGE routing
@app.route("/twitterML")
def twitterML():
    return render_template('twitterML.html')

# for passing variables from form to script
@app.route("/twitterML", methods=['POST'])
def handleML():
    divShown = False
    searchButtonClicked = False
    if request.form['Submit'] == 'search':
        searchButtonClicked = True
        searchTopic = request.form['searchTopic']
        tweets = tweepy_wrapper.RecentTweetsWrapper(searchTopic)
        if tweets:
            divShown = True
        return render_template('twitterML.html', tweets=tweets, searchTopic=searchTopic, divShown=divShown, searchButtonClicked=searchButtonClicked)
    elif request.form['Submit'] == 'analyse':
        session['selectedTweets'] = request.form.getlist('tweet')
        for checkbox in request.form.getlist('tweet'):
            print(checkbox)   
        return redirect("analysisML", code=302)

# Analysis PAGE routing
@app.route("/analysisML")
def analysisML():
    array=session.get('selectedTweets', None)
    passedTweets = []
    temparr = machLearn_Run.Main(array, 'ML')
    passedTweets = temparr
    print(temparr)
    return render_template('analysisML.html', passedTweets = passedTweets)

####### MLDL Pages

# RAW TEXT routing
@app.route("/rawTextMLDL")
def rawTextMLDL():
    return render_template('rawTextMLDL.html')

# for passing variables from form to script
@app.route('/rawTextMLDL', methods=['POST'])
def webappMLDL():
    print("debug")
    searchInput = []
    searchInput.append(request.form['searchInput'])
    session['selectedTweets'] = searchInput
    return redirect("analysisMLDL", code=302)

# TWITTER PAGE routing
@app.route("/twitterMLDL")
def twitterMLDL():
    return render_template('twitterMLDL.html')

# for passing variables from form to script
@app.route("/twitterMLDL", methods=['POST'])
def handleMLDL():
    divShown = False
    searchButtonClicked = False
    if request.form['Submit'] == 'search':
        searchButtonClicked = True
        searchTopic = request.form['searchTopic']
        tweets = tweepy_wrapper.RecentTweetsWrapper(searchTopic)
        if tweets:
            divShown = True
        return render_template('twitterMLDL.html', tweets=tweets, searchTopic=searchTopic, divShown=divShown, searchButtonClicked=searchButtonClicked)
    elif request.form['Submit'] == 'analyse':
        session['selectedTweets'] = request.form.getlist('tweet')
        for checkbox in request.form.getlist('tweet'):
            print(checkbox)   
        return redirect("analysisMLDL", code=302)

# Analysis PAGE routing
@app.route("/analysisMLDL")
def analysisMLDL():
    array=session.get('selectedTweets', None)
    passedTweets = []
    temparr = machLearn_Run.Main(array, 'MLDL')
    passedTweets = temparr
    print(temparr)
    return render_template('analysisMLDL.html', passedTweets = passedTweets)

# Running app, debug mode can be changed here
if __name__ == '__main__':
    app.run(debug=True)
