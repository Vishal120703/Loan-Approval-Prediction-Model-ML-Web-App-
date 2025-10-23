# from flask import Flask, render_template, request
# import numpy as np
# import pandas as pd
# import pickle

# app = Flask(__name__)

# # Load your trained model and scaler
# model = pickle.load(open('model.pkl', 'rb'))
# scaler = pickle.load(open('scaler.pkl', 'rb'))

# @app.route('/')
# def home():
#     return render_template('home.html')

# @app.route('/predict', methods = ['GET'])
# def prediction():
#     return render_template('prediction.html')

# @app.route('/about', methods = ['GET'])
# def about():
#     return render_template('about.html')


# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         # Get data from form
#         x = float(request.form['int_rate'])
#         y = request.form['purpose']
#         if(y == "all_other"):
#             z = 1
#         elif (y=="credit_card"):
#             z = 2
#         elif (y=="debt_consolidation"):
#             z = 3
#         elif(y == "educational"):
#             z = 4
#         elif(y == "home_improvement"):
#             z = 5
#         else:
#             z = 6

#         data = {
#             'credit.policy': int(request.form['credit_policy']),
#             # 'purpose': int(request.form['purpose']),
#             'purpose': z,
#             'int.rate': x/100,
#             'installment': float(request.form['installment']),
#             'log.annual.inc': float(request.form['log_annual_inc']),
#             'dti': float(request.form['dti']),
#             'fico': int(request.form['fico']),
#             'days.with.cr.line': float(request.form['days_with_cr_line']),
#             'revol.bal': float(request.form['revol_bal']),
#             'revol.util': float(request.form['revol_util']),
#             'inq.last.6mths': int(request.form['inq_last_6mths']),
#             'delinq.2yrs': int(request.form['delinq_2yrs']),
#             'pub.rec': int(request.form['pub_rec']),
#             'Gender': int(request.form['gender']),
#             'Married': int(request.form['married']),
#         }
#         print(data)

#         df = pd.DataFrame([data])
#         # print(df)
#         scaled_data = scaler.transform(df)
#         prediction = model.predict(scaled_data)[0]
#         probability = model.predict_proba(scaled_data)[0][1]
#         print("prediction",prediction)

#         return render_template('result.html',
#                                prediction=prediction,
#                                probability=round(probability * 100, 2))

# if __name__ == '__main__':
#     app.run(debug=True)




from flask import Flask, render_template, request, send_file
import numpy as np
import pandas as pd
import pickle
import io
from reportlab.pdfgen import canvas

app = Flask(__name__)

# Load your trained model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET'])
def prediction():
    return render_template('prediction.html')

@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        raw_data = request.form.to_dict()

        # Map purpose to integer
        purpose_map = {
            "all_other": 1,
            "credit_card": 2,
            "debt_consolidation": 3,
            "educational": 4,
            "home_improvement": 5
        }
        purpose_val = purpose_map.get(raw_data.get('purpose'), 6)

        # Prepare input data for model
        data = {
            'credit.policy': int(raw_data['credit_policy']),
            'purpose': purpose_val,
            'int.rate': float(raw_data['int_rate']) / 100,
            'installment': float(raw_data['installment']),
            'log.annual.inc': float(raw_data['log_annual_inc']),
            'dti': float(raw_data['dti']),
            'fico': int(raw_data['fico']),
            'days.with.cr.line': float(raw_data['days_with_cr_line']),
            'revol.bal': float(raw_data['revol_bal']),
            'revol.util': float(raw_data['revol_util']),
            'inq.last.6mths': int(raw_data['inq_last_6mths']),
            'delinq.2yrs': int(raw_data['delinq_2yrs']),
            'pub.rec': int(raw_data['pub_rec']),
            'Gender': int(raw_data['gender']),
            'Married': int(raw_data['married']),
        }

        df = pd.DataFrame([data])
        scaled_data = scaler.transform(df)

        # Prediction
        prediction = model.predict(scaled_data)[0]
        probability = model.predict_proba(scaled_data)[0][1]

        # Pass prediction, probability, and input data to template
        return render_template(
            'result.html',
            prediction=prediction,
            probability=round(probability * 100, 2),
            inputs=data  # send cleaned data dictionary
        )

# Route to download PDF report
@app.route('/download-report')
def download_report():
    # Get all input data and prediction from query parameters
    prediction = request.args.get('prediction', 'N/A')
    probability = request.args.get('probability', 'N/A')

    # Collect all other input fields dynamically
    input_fields = {key: request.args.get(key, 'N/A') for key in [
        'credit.policy', 'purpose', 'int.rate', 'installment', 'log.annual.inc', 'dti',
        'fico', 'days.with.cr.line', 'revol.bal', 'revol.util', 'inq.last.6mths',
        'delinq.2yrs', 'pub.rec', 'Gender', 'Married'
    ]}

    # Create PDF in memory
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer)
    p.setFont("Helvetica-Bold", 18)
    p.drawString(180, 800, "Loan Prediction Report")
    p.setFont("Helvetica", 12)

    y = 760
    for key, value in {**input_fields,
                       "Prediction Result": prediction,
                       "Probability (%)": probability
                       }.items():
        p.drawString(100, y, f"{key}: {value}")
        y -= 20

    p.showPage()
    p.save()
    buffer.seek(0)

    return send_file(buffer, as_attachment=True,
                     download_name="loan_prediction_report.pdf",
                     mimetype='application/pdf')


if __name__ == '__main__':
    app.run(debug=True)
