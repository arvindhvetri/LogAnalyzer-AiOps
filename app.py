from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.arima.model import ARIMA

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'txt', 'log'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')
    
    file = request.files['file']
    
    if file.filename == '':
        return render_template('index.html', error='No selected file')
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the log file
        df = parse_logs(filepath)
        df = feature_engineering(df)
        
        # Generate visualizations
        analysis_plot = plot_data_analysis(df)
        arima_plot = plot_arima_forecast(df)
        anomalies = find_anomalies(df)
        
        return render_template('result.html', 
                             analysis_plot=analysis_plot,
                             arima_plot=arima_plot,
                             anomalies=anomalies.to_html(classes='table table-striped'))
    
    return render_template('index.html', error='Invalid file type')

def parse_logs(file_path):
    with open(file_path, "r") as file:
        logs = file.readlines()

    data = []
    for log in logs:
        parts = log.strip().split(" ", 3)
        if len(parts) < 4:
            continue
        timestamp = parts[0] + " " + parts[1]
        level = parts[2]
        message = parts[3]
        data.append([timestamp, level, message])

    df = pd.DataFrame(data, columns=["timestamp", "level", "message"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
    df.dropna(subset=["timestamp"], inplace=True)
    df.sort_values("timestamp", inplace=True)
    return df

def feature_engineering(df):
    level_mapping = {"INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4}
    df["level_score"] = df["level"].map(level_mapping).fillna(0)
    df["message_length"] = df["message"].apply(len)
    df["contains_error_keyword"] = df["message"].apply(
        lambda x: int(any(k in x.lower() for k in ["fail", "error", "exception", "unreachable", "blocked", "brute force"])))
    df["time_diff_sec"] = df["timestamp"].diff().dt.total_seconds().fillna(0)
    
    # Anomaly Detection
    isolation_forest = IsolationForest(contamination=0.05, random_state=42)
    features = df[["level_score", "message_length", "contains_error_keyword", "time_diff_sec"]]
    df["anomaly"] = isolation_forest.fit_predict(features)
    df["is_anomaly"] = df["anomaly"].apply(lambda x: "Anomaly" if x == -1 else "Normal")
    
    return df

def plot_data_analysis(df):
    plt.figure(figsize=(12, 10))
    sns.set(style="darkgrid")

    # 1. Log Level Count
    plt.subplot(3, 2, 1)
    sns.countplot(data=df, x="level", palette="Set2")
    plt.title("Log Level Distribution")

    # 2. Message Length Distribution
    plt.subplot(3, 2, 2)
    sns.histplot(df["message_length"], bins=30, kde=True, color="skyblue")
    plt.title("Message Length Distribution")

    # 3. Time Gap Between Logs
    plt.subplot(3, 2, 3)
    sns.histplot(df["time_diff_sec"], bins=30, kde=True, color="orange")
    plt.title("Time Gap Between Logs (sec)")

    # 4. Error Keyword Presence
    plt.subplot(3, 2, 4)
    sns.countplot(x="contains_error_keyword", data=df, palette="coolwarm")
    plt.title("Error Keywords in Logs")
    plt.xticks([0, 1], ["No", "Yes"])

    # 5. Anomalies Over Time
    plt.subplot(3, 2, 5)
    sns.lineplot(x="timestamp", y="level_score", hue="is_anomaly", data=df)
    plt.title("Anomalies Over Time")
    plt.xticks(rotation=45)

    # 6. Anomaly Count
    plt.subplot(3, 2, 6)
    sns.countplot(x="is_anomaly", data=df, palette="husl")
    plt.title("Anomaly vs Normal")

    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(app.config['RESULT_FOLDER'], "log_analysis.png")
    plt.savefig(plot_path)
    plt.close()
    
    return plot_path

def plot_arima_forecast(df):
    try:
        level_mapping = {"INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4}
        df_severity = df[["timestamp", "level"]].copy()
        df_severity["level"] = df_severity["level"].map(level_mapping).fillna(0)
        df_severity.dropna(inplace=True)
        df_severity.set_index("timestamp", inplace=True)
        
        # Resample to 10-second intervals for better trend visualization
        df_resampled = df_severity.resample('10S').mean().fillna(method='ffill')
        
        # ARIMA model training with more optimized parameters
        model = ARIMA(df_resampled['level'], order=(2, 1, 2))
        model_fit = model.fit()
        
        # Forecast 30 steps ahead (5 minutes at 10-second intervals)
        forecast_steps = 30
        forecast = model_fit.forecast(steps=forecast_steps)
        forecast_index = pd.date_range(
            df_resampled.index[-1] + pd.Timedelta(seconds=10), 
            periods=forecast_steps, 
            freq='10S'
        )
        
        # Calculate confidence intervals
        conf_int = model_fit.get_forecast(steps=forecast_steps).conf_int()
        
        # Create the plot
        plt.figure(figsize=(14, 6))
        
        # Plot historical data
        plt.plot(
            df_resampled.index, 
            df_resampled['level'], 
            label='Observed', 
            color='#4361ee',
            linewidth=2
        )
        
        # Plot forecast
        plt.plot(
            forecast_index, 
            forecast, 
            label='Forecast', 
            color='#f72585',
            linewidth=2,
            linestyle='--'
        )
        
        # Plot confidence interval
        plt.fill_between(
            forecast_index,
            conf_int.iloc[:, 0],
            conf_int.iloc[:, 1],
            color='#f72585',
            alpha=0.1,
            label='95% Confidence Interval'
        )
        
        # Formatting
        plt.title("Extended Log Severity Level Forecast (ARIMA)", fontsize=14, pad=20)
        plt.xlabel("Time", fontsize=12)
        plt.ylabel("Severity Level", fontsize=12)
        plt.legend(loc='upper left', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(app.config['RESULT_FOLDER'], "arima_forecast.png")
        plt.savefig(plot_path, dpi=120, bbox_inches='tight')
        plt.close()
        
        return plot_path
    except Exception as e:
        print(f"ARIMA Error: {e}")
        return None


def find_anomalies(df):
    anomalies = df[df["anomaly"] == -1]
    result_folder = os.path.join("static", "results")
    os.makedirs(result_folder, exist_ok=True)  # Ensure the folder exists

    # Save as CSV
    csv_path = os.path.join(result_folder, "anomalies.csv")
    anomalies[["timestamp", "level", "message", "is_anomaly"]].to_csv(csv_path, index=False)

    # Save as TXT
    txt_path = os.path.join(result_folder, "anomalies.txt")
    anomalies[["timestamp", "level", "message", "is_anomaly"]].to_string(open(txt_path, "w"), index=False)

    return anomalies[["timestamp", "level", "message", "is_anomaly"]]

if __name__ == '__main__':
    app.run(debug=True)