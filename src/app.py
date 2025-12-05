import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# --- 1. DATA PREPARATION ---
def load_and_prep_data():
    print("🌍 Loading Air Quality Data...")
    try:
        df = pd.read_csv('data/city_day.csv')
    except FileNotFoundError:
        print("Error: 'data/city_day.csv' not found.")
        exit()

    # We need the most recent data to make current predictions
    # Let's filter for data from the year 2020 only to keep it relevant
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[df['Date'].dt.year == 2020]

    # Group by City to get the average PM2.5 for each city
    city_aqi = df.groupby('City')['PM2.5'].mean().reset_index()
    
    # Drop cities where PM2.5 is NaN (missing)
    city_aqi = city_aqi.dropna()

    # --- THE TRICK: Mapping Cities to Lat/Long ---
    # Since the CSV doesn't have coordinates, we map them manually 
    # to create our "Spatial" dataset.
    city_coords = {
        'Ahmedabad': [23.0225, 72.5714], 'Aizawl': [23.7271, 92.7176],
        'Amaravati': [16.5151, 80.5182], 'Amritsar': [31.6340, 74.8723],
        'Bengaluru': [12.9716, 77.5946], 'Bhopal': [23.2599, 77.4126],
        'Brajrajnagar': [21.8286, 83.9238], 'Chandigarh': [30.7333, 76.7794],
        'Chennai': [13.0827, 80.2707], 'Coimbatore': [11.0168, 76.9558],
        'Delhi': [28.7041, 77.1025], 'Ernakulam': [9.9816, 76.2999],
        'Gurugram': [28.4595, 77.0266], 'Guwahati': [26.1445, 91.7362],
        'Hyderabad': [17.3850, 78.4867], 'Jaipur': [26.9124, 75.7873],
        'Jorapokhar': [23.7046, 86.4116], 'Kochi': [9.9312, 76.2673],
        'Kolkata': [22.5726, 88.3639], 'Lucknow': [26.8467, 80.9462],
        'Mumbai': [19.0760, 72.8777], 'Patna': [25.5941, 85.1376],
        'Shillong': [25.5788, 91.8933], 'Talcher': [20.9500, 85.2167],
        'Thiruvananthapuram': [8.5241, 76.9366], 'Visakhapatnam': [17.6868, 83.2185]
    }

    # Add Lat/Long to the dataframe
    lat_list = []
    long_list = []
    
    # Only keep cities we have coordinates for
    final_cities = []
    final_pm = []

    for index, row in city_aqi.iterrows():
        city = row['City']
        if city in city_coords:
            lat_list.append(city_coords[city][0])
            long_list.append(city_coords[city][1])
            final_cities.append(city)
            final_pm.append(row['PM2.5'])

    # Create the final clean dataset for the AI
    X = pd.DataFrame({'Latitude': lat_list, 'Longitude': long_list})
    y = np.array(final_pm)
    
    print(f"✅ Data Prepared: {len(X)} Sensor Stations active.")
    return X, y, city_coords

# --- 2. THE AI ENGINE ---
def train_and_run():
    X, y, city_coords = load_and_prep_data()

    # Train KNN Regressor
    # weights='distance' is CRITICAL here. 
    # It means closer sensors influence the prediction more than far ones.
    knn = KNeighborsRegressor(n_neighbors=3, weights='distance')
    knn.fit(X, y)

    print("\n--- 🌫️ VIRTUAL AQI SENSOR 🌫️ ---")
    print("Predict Air Quality for any location in India.")
    
    while True:
        print("\nOptions:")
        print("1. Enter Coordinates manually")
        print("2. Use a Preset Location (Demo)")
        choice = input("Select (1/2): ")

        input_lat, input_long = 0, 0

        if choice == '1':
            try:
                input_lat = float(input("Enter Latitude: "))
                input_long = float(input("Enter Longitude: "))
            except ValueError:
                print("Invalid number.")
                continue
        elif choice == '2':
            # A location between Delhi and Gurugram (No sensor here)
            print("Using location: 'New Delhi Suburbs' (Lat: 28.55, Long: 77.15)")
            input_lat = 28.55
            input_long = 77.15
        else:
            break

        # Predict
        prediction = knn.predict([[input_lat, input_long]])
        pm_val = prediction[0]

        # Interpret Result
        status = ""
        if pm_val <= 50: status = "🟢 Good"
        elif pm_val <= 100: status = "🟡 Moderate"
        elif pm_val <= 200: status = "🟠 Poor"
        else: status = "🔴 Unhealthy / Hazardous"

        print(f"\n💨 Estimated PM2.5 Level: {pm_val:.2f}")
        print(f"health Status: {status}")
        print("(Calculated using Inverse Distance Weighting from nearest cities)")

if __name__ == "__main__":
    train_and_run()