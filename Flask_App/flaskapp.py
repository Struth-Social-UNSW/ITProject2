# imports
from re import search
from flask import Flask, render_template, request, redirect, url_for, session
import testcase
import tweepy_wrapper

app = Flask(__name__)


# HOME PAGE routing
@app.route("/")
def home():
    return render_template('home.html')


# for passing variables from form to script
@app.route('/', methods=['POST'])
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
def handle():
    if request.form['Submit'] == 'search':
        searchTopic = request.form['searchTopic']
        print('helloworld')
        tweets = tweepy_wrapper.RecentTweetsWrapper(searchTopic)
        return render_template('twitter.html', tweets=tweets, searchTopic=searchTopic)
    elif request.form['Submit'] == 'analyse':
        # people = request.form.getlist('people')
        print('helloworld2')
        for checkbox in request.form.getlist('tweet'):
            print(checkbox)
        return redirect("analysis", code=302)
        # return redirect(url_for('analysis', checkbox=checkbox, **request.args))
        # session['checkbox'] = checkbox
        # return redirect(url_for('analysis', checkbox=checkbox))


# Analysis PAGE routing
@app.route("/analysis")
def analysis():
    return render_template('analysis.html')

# Running app, debug mode can be changed here
if __name__ == '__main__':
    app.run(debug=True)
