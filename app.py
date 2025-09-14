from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Load your trained model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods = ['GET'])
def prediction():
    return render_template('prediction.html')

@app.route('/about', methods = ['GET'])
def about():
    return render_template('about.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get data from form
        data = {
            'credit.policy': int(request.form['credit_policy']),
            'purpose': int(request.form['purpose']),
            'int.rate': float(request.form['int_rate']),
            'installment': float(request.form['installment']),
            'log.annual.inc': float(request.form['log_annual_inc']),
            'dti': float(request.form['dti']),
            'fico': int(request.form['fico']),
            'days.with.cr.line': float(request.form['days_with_cr_line']),
            'revol.bal': float(request.form['revol_bal']),
            'revol.util': float(request.form['revol_util']),
            'inq.last.6mths': int(request.form['inq_last_6mths']),
            'delinq.2yrs': int(request.form['delinq_2yrs']),
            'pub.rec': int(request.form['pub_rec']),
            'Gender': int(request.form['gender']),
            'Married': int(request.form['married']),
        }

        df = pd.DataFrame([data])
        scaled_data = scaler.transform(df)
        prediction = model.predict(scaled_data)[0]
        probability = model.predict_proba(scaled_data)[0][1]

        return render_template('result.html',
                               prediction=prediction,
                               probability=round(probability * 100, 2))

if __name__ == '__main__':
    app.run(debug=True)
