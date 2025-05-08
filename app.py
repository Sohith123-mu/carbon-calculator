from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import seaborn as sns
import matplotlib.pyplot as plt
import os
from datetime import datetime

app = Flask(__name__)


model_cnn = load_model("models/cnn_model.h5", compile=False)


# Load all models and encoders using joblib
scaler = joblib.load("models/scaler.joblib")
feature_columns = joblib.load("models/feature_columns.joblib")
mlb_cooking = joblib.load("models/mlb_cooking.joblib")
mlb_recycling = joblib.load("models/mlb_recycling.joblib")
model_dt = joblib.load("models/dt_model.joblib")
model_rf = joblib.load("models/rf_model.joblib")
#model_cnn = load_model("models/cnn_model.h5")  # CNN still needs keras

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    input_data = request.form.to_dict()

    # Extract multi-label fields
    recycling = [x.strip() for x in input_data['Recycling'].split(',') if x.strip()]
    cooking = [x.strip() for x in input_data['Cooking_With'].split(',') if x.strip()]

    # Numeric fields
    numeric_data = {
        'Monthly Grocery Bill': float(input_data['Monthly Grocery Bill']),
        'Vehicle Monthly Distance Km': float(input_data['Vehicle Monthly Distance Km']),
        'Waste Bag Weekly Count': float(input_data['Waste Bag Weekly Count']),
        'How Long TV PC Daily Hour': float(input_data['How Long TV PC Daily Hour']),
        'How Many New Clothes Monthly': float(input_data['How Many New Clothes Monthly']),
        'How Long Internet Daily Hour': float(input_data['How Long Internet Daily Hour'])
    }

    df_temp = pd.DataFrame([numeric_data])
    df_temp['Body Type_' + input_data['Body Type']] = 1
    df_temp['Diet_' + input_data['Diet']] = 1

    # One-hot/multi-label encoding
    cooking_df = pd.DataFrame(mlb_cooking.transform([cooking]), columns=mlb_cooking.classes_)
    recycling_df = pd.DataFrame(mlb_recycling.transform([recycling]), columns=mlb_recycling.classes_)

    df_combined = pd.concat([df_temp, recycling_df, cooking_df], axis=1).reindex(columns=feature_columns, fill_value=0)

    input_scaled = scaler.transform(df_combined)

    # Predict
    dt_pred = model_dt.predict(input_scaled)[0]
    rf_pred = model_rf.predict(input_scaled)[0]
    cnn_pred = model_cnn.predict(input_scaled)[0][0]

    return render_template("index.html", result={
        "dt": round(dt_pred, 2),
        "rf": round(rf_pred, 2),
        "cnn": round(cnn_pred, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
