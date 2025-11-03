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

        # Map purpose to integer values for the model
        purpose_map = {
            "all_other": 1,
            "credit_card": 2,
            "debt_consolidation": 3,
            "educational": 4,
            "home_improvement": 5
        }
        purpose_val = purpose_map.get(raw_data.get('purpose'), 6)

        # Prepare data for the model
        data = {
            'credit.policy': int(raw_data['credit_policy']),
            'purpose': purpose_val,
            'int.rate': float(raw_data['int_rate']) / 100,
            'installment': float(raw_data['installment']),
            'log.annual.inc': float(np.log(float(raw_data['log_annual_inc']))),
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

        # Make prediction
        prediction = model.predict(scaled_data)[0]
        probability = model.predict_proba(scaled_data)[0][1]

        # Create a readable version of input data for result display
        display_inputs = {
            "Credit Policy": "Yes" if raw_data.get("credit_policy") == "1" else "No",
            "Purpose": raw_data.get("purpose", "N/A").replace("_", " ").title(),
            "Interest Rate (%)": raw_data.get("int_rate"),
            "Installment": raw_data.get("installment"),
            "Log Annual Income": raw_data.get("log_annual_inc"),
            "Debt to Income (DTI)": raw_data.get("dti"),
            "FICO Score": raw_data.get("fico"),
            "Days with Credit Line": raw_data.get("days_with_cr_line"),
            "Revolving Balance": raw_data.get("revol_bal"),
            "Revolving Utilization": raw_data.get("revol_util"),
            "Inquiries (Last 6 Months)": raw_data.get("inq_last_6mths"),
            "Delinquencies (2 Yrs)": raw_data.get("delinq_2yrs"),
            "Public Records": raw_data.get("pub_rec"),
            "Gender": "Male" if raw_data.get("gender") == "1" else "Female",
            "Marital Status": "Married" if raw_data.get("married") == "1" else "Single"
        }

        # Render the result page with prediction and readable data
        return render_template(
            'result.html',
            prediction=prediction,
            probability=round(probability * 100, 2),
            inputs=display_inputs
        )


# Route to download PDF report
@app.route('/download-report')
def download_report():
    # Get prediction and probability
    prediction = request.args.get('prediction', 'N/A')
    probability = request.args.get('probability', 'N/A')

    # Collect all input fields dynamically from query parameters
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
                       "Probability (%)": probability}.items():
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





