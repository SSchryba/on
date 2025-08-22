# data_fetch.py - Data ingestion from APIs

import requests
import json
from utils import get_api_key
from config import API_URLS, LAUNCH_SITES

def fetch_space_track_data():
    # Example: Auth and fetch TLE data (simplified; implement full login)
    username = get_api_key('SPACE_TRACK_USERNAME')
    password = get_api_key('SPACE_TRACK_PASSWORD')
    login_url = f"{API_URLS['space_track']}/ajaxauth/login"
    query_url = f"{API_URLS['space_track']}/basicspacedata/query/class/tle_latest/ORDINAL/1/NORAD_CAT_ID/25544/format/json"  # Example: ISS TLE

    session = requests.Session()
    session.post(login_url, data={'identity': username, 'password': password})
    response = session.get(query_url)
    return response.json() if response.status_code == 200 else None

def fetch_weather_data(site='cape_canaveral'):
    api_key = get_api_key('OPENWEATHER_API_KEY')
    lat, lon = LAUNCH_SITES[site]['lat'], LAUNCH_SITES[site]['lon']
    url = f"{API_URLS['openweather']}?lat={lat}&lon={lon}&appid={api_key}&units=imperial"
    response = requests.get(url)
    return response.json() if response.status_code == 200 else None

def fetch_space_weather():
    url = API_URLS['noaa_space_weather']
    response = requests.get(url)
    return json.loads(response.text) if response.status_code == 200 else None