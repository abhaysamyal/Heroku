from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

loaded_model = joblib.load('dib_78.pkl')

@app.route('/') #decorator
def homepage():
    return render_template('homepage.html')

@app.route('/predict', methods = ['POST']) #decorator
def predict():
    preg = request.form.get('preg')
    plas = request.form.get('plas')
    pres = request.form.get('pres')
    skin = request.form.get('skin')
    test = request.form.get('test')
    mass = request.form.get('mass')
    pedi = request.form.get('pedi')
    age = request.form.get('age')
    
    prediction = loaded_model.predict([[preg,plas,pres,skin,test,mass,pedi,age]])

    if prediction[0]==1:
        val = 'diabetic'

    else:
        val = 'not diabetic'

    return render_template('result.html', value = val)
    
if __name__=='main':
    app.run(debug=True)