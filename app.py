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
    if request.method == 'POST':
        button = request.form['submit']
        if button == 'General Assessment of Cryptocurrencies':
            ptable = general_assessment()
            session['ptable'] = True
            return render_template('index.html', ptable=ptable.to_html(escape=False))
        session['ptable'] = False
        return redirect('/')
    return render_template('index.html')

if __name__ == '__main__':
    app.debug=True
    app.secret_key = 'A0Zr98j/3yX R~XHH!jmN]LWX/,?RT'
    app.run()
