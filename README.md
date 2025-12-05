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
