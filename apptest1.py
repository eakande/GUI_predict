# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 21:22:48 2022

@author: eakan
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
modeltest1 = pickle.load(open('modeltest1.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = modeltest1.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='The predicted inflation value is {}%'.format(output),
                           kulikuli='This trained machine learning algorithm is about 95% accurate')

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = modeltest1.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=False)
