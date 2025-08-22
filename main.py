# main.py - Startup script

from utils import load_env
from data_fetch import fetch_space_track_data, fetch_weather_data, fetch_space_weather
from analysis import project_orbit, detect_anomalies, define_launch_window

def main():
    load_env()
    
    print("Fetching data...")
    tle_data = fetch_space_track_data()
    weather_data = fetch_weather_data()
    space_weather = fetch_space_weather()
    
    if tle_data and weather_data and space_weather:
        print("Data fetched successfully.")
        
        orbit_projection = project_orbit(tle_data)
        anomalies = detect_anomalies(space_weather)  # Example on space weather data
        
        window_status = define_launch_window(weather_data, space_weather, orbit_projection)
        print(window_status)
        
        if anomalies.size > 0:
            print("Anomalies detected:", anomalies)
    else:
        print("Error fetching data. Check API keys and connections.")

if __name__ == "__main__":
    main()