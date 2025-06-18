from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and encoder
model = joblib.load("models/crop_recommendation_model.pkl")
le = joblib.load("models/label_encoder.pkl")

# Yield info (optional)
import pandas as pd
df1 = pd.read_csv("indian_crop_weather.csv")
df1.rename(columns={"Crop": "label", "Yield_kg_per_ha": "Production"}, inplace=True)

# Get all districts for the dropdown
districts_df = df1[['Dist Code', 'Dist Name']].drop_duplicates().sort_values('Dist Code')
districts_list = districts_df.to_dict('records')

def get_yield_info(crop_name):
    rows = df1[df1["label"].str.lower() == crop_name.lower()]
    valid = rows["Production"].dropna()
    if valid.empty:
        return "No production data available."
    return f"Average Yield: {valid.mean():.2f} kg/ha"

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    if request.method == "POST":
        try:
            temp = float(request.form["temperature"])
            hum = float(request.form["humidity"])
            ph = float(request.form["ph"])
            rain = float(request.form["rainfall"])
            district = int(request.form["district"])            # Create feature array in the order: temperature, humidity, ph, rainfall, District
            data = np.array([[temp, hum, ph, rain, district]])
            pred = model.predict(data)
            crop = le.inverse_transform(pred)[0]
            yield_info = get_yield_info(crop)
            prediction = f"🌱 Recommended Crop: {crop}<br>📊 {yield_info}"

        except Exception as e:
            prediction = f"Error: {str(e)}"
    return render_template("index.html", prediction=prediction, districts=districts_list)

if __name__ == "__main__":
    app.run(debug=True)
