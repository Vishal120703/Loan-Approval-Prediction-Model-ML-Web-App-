# Loan Default Prediction Web App (ML + Flask + TailwindCSS)

## Overview  
This project is a **Loan Default Prediction System** that uses **machine learning** to classify loans as either:  
- ✅ **Fully Paid**  
- ❌ **Not Fully Paid (Default Risk)**  

It is built as a **Flask web application** with a clean **TailwindCSS frontend** and **Chart.js** visualizations.  
The model helps financial institutions and users assess the repayment risk of a loan based on borrower details.  

---

## 🚀 Features  
- 📊 Predict loan repayment outcome  
- 🎯 Probability score for **default risk**  
- 🎨 Responsive UI with **TailwindCSS (Dark Mode)**  
- 📈 Dynamic **Chart.js** visualizations  
- 📚 Explanations for each input field (About Page)  

---

## Input Features  
The model takes the following borrower & loan details:  

- **Credit Policy**  
- **Purpose (0–N encoded)**  
- **Interest Rate**  
- **Installment**  
- **Log Annual Income**  
- **DTI (Debt-to-Income Ratio)**  
- **FICO Score**  
- **Days with Credit Line**  
- **Revolving Balance**  
- **Revolving Utilization**  
- **Inquiries (Last 6 Months)**  
- **Delinquencies (Last 2 Years)**  
- **Public Records**  
- **Gender (1 = Male, 0 = Female)**  
- **Married (1 = Yes, 0 = No)**  

---

## 🛠️ Tech Stack  
- **Backend:** Flask (Python)  
- **Frontend:** TailwindCSS, Chart.js  
- **ML Model:** scikit-learn (Logistic Regression / Random Forest, etc.)  
- **Deployment Ready:** GitHub + (Heroku / Render / Railway)  

---

## ⚙️ How It Works  
1. User enters borrower details in the prediction form.  
2. The ML model processes input and predicts:  
   - ✅ **Fully Paid**  
   - ❌ **Not Fully Paid (Default Risk)**  
3. Displays the result along with a **probability score and visualization**.  

---


## 🔧 Setup Instructions  
```bash
# Clone repo
git clone https://github.com/your-username/loan-default-prediction.git
cd loan-default-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows

# Run Flask app
python app.py
