# Air-Quality-Interpolation-AI
# 🌍 EcoSense: Virtual Air Quality Sensor

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-KNN%20Regression-orange)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-green)

### 🌫️ Estimating Pollution in Unmonitored Zones using Spatial Interpolation

**Domain:** Environmental Science / Smart Cities  
**Technique:** Inverse Distance Weighted (IDW) Regression

---

## 📖 Project Overview
Government air quality sensors are expensive and sparse. A city might have only 2-3 sensors, leaving vast residential areas with no data on the air they breathe.

This project builds a **"Virtual Sensor"**. It uses the **K-Nearest Neighbors (KNN)** algorithm to estimate the PM2.5 (Particulate Matter) level of *any* specific coordinate (Latitude/Longitude) by analyzing data from the nearest actual sensors.

## 🧠 The Science: Spatial Interpolation
Unlike standard classification, this project uses **KNN Regression** with a twist.

1.  **Geolocation Mapping:** Since the raw dataset only contained city names, I built a custom mapper to assign precise Latitude/Longitude coordinates to every data point.
2.  **Inverse Distance Weighting (IDW):**
    * If you are at Location X, the model finds the $K=3$ nearest cities.
    * It doesn't just average them. It applies **distance-based weights**.
    * *Example:* If a sensor 5km away says "Polluted" and a sensor 50km away says "Clean," the model prioritizes the closer one.

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Scikit-Learn (KNeighborsRegressor), Pandas, Numpy
* **Dataset:** [Air Quality Data in India (2015-2020)](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)

## 📂 Project Structure
```text
Spatial-AQI-Estimator/
│
├── data/
│   └── city_day.csv      # Daily Air Quality Averages
│
├── src/
│   └── app.py            # The Logic: Data Mapping + KNN Regression
│
├── requirements.txt      # Project Dependencies
└── README.md             # Documentation

How to Run Locally
Clone the Repository

Bash

git clone [https://github.com/YOUR-USERNAME/Spatial-AQI-Estimator-KNN.git](https://github.com/YOUR-USERNAME/Spatial-AQI-Estimator-KNN.git)
cd Spatial-AQI-Estimator-KNN
Install Dependencies

Bash

pip install -r requirements.txt
Setup Data

Download city_day.csv from Kaggle.

Place it inside the data/ folder.

Run the Virtual Sensor

Bash

python src/app.py
🧪 Usage Example
When you run the tool, you can select "Demo Mode".

Scenario: A location between Delhi and Gurugram (Lat: 28.55, Long: 77.15) has no sensor.

AI Prediction: The model looks at Delhi (North), Gurugram (South), and nearby hubs to triangulate the pollution level.

Output: Estimated PM2.5: 145.2 (Unhealthy)

🔮 Future Scope
Heatmap Visualization: plotting the estimated AQI on a Google Map interface.

Wind Direction Factor: Improving accuracy by accounting for wind blowing pollution from one city to another.

Real-time API: Connecting to OpenWeatherMap for live data fetching.

Created by Ayush S from Maharaja Institute of Technology
