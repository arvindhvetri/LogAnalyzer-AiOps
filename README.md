# 🔥 AI-Powered Log Analysis & Forecasting Web App

A Flask-based web application that performs intelligent **Log File Analysis**, **Anomaly Detection**, and **Future Severity Forecasting** using Machine Learning and Time Series Modeling.

![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-2.0-lightgrey?logo=flask)
![Machine Learning](https://img.shields.io/badge/ML-IsolationForest-yellowgreen)
![Forecasting](https://img.shields.io/badge/ARIMA-TimeSeries-orange)

---

## 🧠 Features

- 📝 Upload or paste raw log text into the web interface
- 🔍 Clean and preprocess logs into structured tabular data
- 📊 Visualize log patterns with **Beautiful Charts** (severity, time gaps, keyword hits)
- 🚨 Detect **Anomalies** using **Isolation Forest**
- ⏳ Predict **Future Log Severity** using **ARIMA**
- 📥 Automatically saves result images, csv, txt files in `static/results` and displays them
- 📱 **Responsive Design** — Works gracefully on desktop and mobile.

---

## 🖼️ Preview

![Preview](https://github.com/user-attachments/assets/9bdbb24c-0ff4-4c8c-b42e-787b231ea31c)


<sub>*A glimpse into the clean interface of Log Analyzer*</sub>

---

## 📦 Tech Stack

| Layer          | Technology          |
|----------------|---------------------|
| Backend        | Python, Flask       |
| ML/Analytics   | Scikit-learn, Pandas, Matplotlib, Seaborn, Statsmodels |
| Frontend       | HTML, CSS, Bootstrap |
| Forecasting    | ARIMA (AutoRegressive Integrated Moving Average) |
| Anomaly Detection | Isolation Forest |

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/LogAnalyzer-AiOps.git
cd LogAnalyzer-AiOps
```

### 2. Set Up Virtual Environment
```bash
# On Windows:
py -3 -m venv .venv
.venv\Scripts\activate  
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt 
```

### 4. Run the App
```bash
python app.py
```
Visit http://127.0.0.1:5000 in your browser.

### 5. Log Input
- Users input raw log text via the web interface (or file upload).

### 6. Data Processing
Logs are parsed into structured format:
- **Timestamp**
- **Log level** (INFO, WARNING, ERROR, etc.)
- **Message content**

Then engineered with:
- Message length
- Keyword presence (error-related)
- Time difference between logs

### 7.Anomaly Detection – **Isolation Forest**
> **Isolation Forest** is an unsupervised machine learning algorithm ideal for anomaly detection.  
It isolates anomalies by randomly selecting a feature and splitting the data. Since anomalies are few and different, they get isolated quickly, giving them a higher anomaly score.

In this project, features used:
- Log level score
- Message length
- Keyword hit
- Time gap between logs

Logs classified as:
- ✅ Normal
- 🚨 Anomaly

### 8. Severity Forecast – **ARIMA**
> **ARIMA (AutoRegressive Integrated Moving Average)** is a statistical time series forecasting model.  
It predicts future values based on past observations, trend, and noise patterns.

Here, it forecasts **log severity levels** for the next few minutes:
- INFO (1) → CRITICAL (4)
- Trend is visualized with confidence intervals

---

## 🚀 Output
- Upon running app.py, all generated output files will be saved in the `static/results` directory and the uploaded file will be saved in `uploads` directory.

[LogAnalyzer-AiOps.pdf](https://github.com/user-attachments/files/20358285/LogAnalyzer-AiOps.pdf)


