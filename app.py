
#import libraries
import numpy as np
import flask
import pickle
import pandas as pd
from pathlib import Path
from pandas import DataFrame
from flask import Flask, request, jsonify, render_template, url_for

#Initialize the flask App
app = Flask(__name__)
with (Path(__file__).parent / 'model' / 'model.pkl').open('rb') as model_file:
    model = pickle.load(model_file)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/', methods=['GET', 'POST'])
def main():

    int_features = [float(request.form[name]) for name in ('rooms', 'Type of housing', 'distance', 'bathroom', 'car', 'Region')]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    return render_template('index.html', prediction_text ='The estimate price is :{}'.format(prediction))
if __name__ == '__main__':  
   app.run(debug = True)
