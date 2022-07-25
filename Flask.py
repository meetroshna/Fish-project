#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pickle
import pandas as pd
from flask import Flask, request, jsonify, render_template

app=Flask(__name__)
pickle_in = open("fish.pkl","rb")
logistic_model = pickle.load(pickle_in)

@app.route('/')
def home():
    return render_template('Fish.html')


@app.route('/predict',methods=["POST"])
def predict():
    """
    For rendering results on HTML GUI
    """
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = logistic_model.predict(final_features)
    #prediction = logistic_model.predict([[242.0,23.2,25.4,30.0,11.5200,4.0200]])
    return render_template('Fish.html', prediction_text = 'THE SPECIES OF THE FISH IS  {}'.format(str(prediction)))

if __name__=='__main__':
    app.run()


# In[ ]:




