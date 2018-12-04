import flask
from flask import Response, request, send_file, Flask, render_template, session, redirect
import pandas as pd
pd.set_option('max_colwidth', 280)
import os
from bokeh.embed import components
from analysis import *

app = Flask(__name__, static_url_path='/static')


@app.route('/', methods=['GET', 'POST'])
def index():
    reset()
    names = all_names()
    if request.method == 'POST':
        button = request.form['submit']
        if button == 'General Assessment of Cryptocurrencies':
            ptable = general_assessment()
            session['ptable'] = True
            return render_template('index.html', ptable=ptable.to_html(escape=False), names=names)
        if button == 'Search':
            name = request.form['purpose']
            ptable = download_crypto(name)
            session['ptable'] = True
            return render_template('index.html', names=names, ptable=ptable.to_html(escape=False))
        if button == "Random Forest ML Analysis of Bitcoin":
            print1, print2 = jerry_learn()
            session['rforest'] = True
            return render_template('index.html', names=names, print1=print1, print2=print2)
        if button == "Bitcoin Time Series Analysis":
            time_series()
            session['time'] = True
            return render_template('index.html', names=names, url1 ='/static/plot1.png', url2 ='/static/plot2.png', url3 ='/static/plot3.png')
        return redirect('/')
    return render_template('index.html', names=names)

if __name__ == '__main__':
    app.debug=True
    app.secret_key = 'A0Zr98j/3yX R~XHH!jmN]LWX/,?RT'
    app.run()
