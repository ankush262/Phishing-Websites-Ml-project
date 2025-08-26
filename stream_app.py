import streamlit as st
import pickle
import pandas as pd
import numpy as np
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
import whois
import datetime
import re
from warnings import filterwarnings

# --- 1. Load the Saved Model and Feature Names ---
# IMPORTANT: This app now expects a model trained ONLY on the 9 features listed below.
# Please retrain your model accordingly.
model_filename = 'phishing_svm_model.pkl'
try:
    with open(model_filename, 'rb') as file:
        loaded_model = pickle.load(file)
except FileNotFoundError:
    st.error(f"Error: Model file not found at '{model_filename}'.")
    st.stop()

# Updated list to use only the 9 specified features
FEATURE_NAMES = [
    'SFH', 'popUpWindow', 'SSLfinal_State', 'Request_URL', 'URL_of_Anchor',
    'web_traffic', 'URL_Length', 'age_of_domain', 'having_IP_Address'
]

# --- 2. Feature Extraction Functions ---
# Note: We only need the functions for the 9 selected features.

def get_soup_from_url(url):
    """Fetches and parses a URL, returning a BeautifulSoup object or None."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes
        return BeautifulSoup(response.text, 'html.parser')
    except requests.exceptions.RequestException:
        return None

# 1. SFH (Server Form Handler)
def sfh(url, soup):
    if soup is None:
        return 0 # Neutral if we can't get the page
    
    domain = urlparse(url).netloc
    for form in soup.find_all('form', action=True):
        action = form.get('action')
        if not action or action.lower() in ["", "about:blank"]:
            return -1 # Phishy if form action is blank
        
        action_domain = urlparse(action).netloc
        if action_domain and domain not in action_domain:
            return 0 # Suspicious if form submits to a different domain
    
    return 1 # Legitimate if all forms are fine

# 2. URL_of_Anchor
def url_of_anchor(url, soup):
    if soup is None:
        return 0 # Neutral if we can't get the page

    domain = urlparse(url).netloc
    total_anchors = 0
    unsafe_anchors = 0
    for a in soup.find_all('a', href=True):
        total_anchors += 1
        link = a.get('href')
        link_domain = urlparse(link).netloc
        if not link_domain or domain not in link_domain:
            unsafe_anchors += 1
    
    if total_anchors == 0:
        return 1 # No links, no risk from them
    
    percentage = (unsafe_anchors / total_anchors) * 100
    if percentage < 31:
        return 1
    elif 31 <= percentage <= 67:
        return 0
    else:
        return -1

# 1. having_IP_Address
def having_ip_address(url):
    try:
        # Use regex to check if the netloc is an IP address
        if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", urlparse(url).netloc):
            return -1
        return 1
    except:
        return -1

# 2. URL_Length
def url_length(url):
    if len(url) < 54:
        return 1
    elif 54 <= len(url) <= 75:
        return 0
    else:
        return -1

# 3. SSLfinal_State (A simplified check for a valid SSL certificate)
def ssl_final_state(url):
    try:
        # verify=True checks the SSL certificate
        response = requests.get(url, verify=True, timeout=5)
        # A valid response (even a 404) means SSL handshake was successful
        return 1 
    except requests.exceptions.SSLError:
        # SSL error or connection error likely means no valid SSL
        return -1
    except requests.exceptions.RequestException:
        # Other exceptions might mean the site is down or other issues
        return 0

# 4. age_of_domain (Derived from Domain_registeration_length)
def age_of_domain(url):
    try:
        domain = whois.whois(urlparse(url).netloc)
        if domain and domain.creation_date:
            # Handle cases where creation_date is a list
            creation_date = domain.creation_date[0] if isinstance(domain.creation_date, list) else domain.creation_date
            age = (datetime.datetime.now() - creation_date).days
            # Phishing sites are often new, so a domain older than a year is a good sign.
            return 1 if age >= 365 else -1
        return -1
    except Exception: # whois can throw various errors
        return 0 


def get_placeholder_feature():
    # Returning 0 as a neutral/suspicious value
    return 0
    

def extract_features_and_predict(url):
    # --- Hardcoded Rule: Check for IP Address in URL ---
    # Legitimate websites almost never use a raw IP address for user-facing pages.
    # This is a very strong indicator of phishing, so we can override the model.
    if having_ip_address(url) == -1:
        # Create a DataFrame to explain the override to the user
        override_data = {name: 'N/A (Rule Triggered)' for name in FEATURE_NAMES}
        override_data['having_IP_Address'] = -1
        override_data['URL_Length'] = url_length(url) # Show the conflicting feature
        override_data['SSLfinal_State'] = ssl_final_state(url) # Show the other strong signal
        features_df = pd.DataFrame([override_data])
        # Return phishing prediction (0) and an explanation message
        return 0, features_df, "Hardcoded Rule Triggered: URLs with direct IP addresses are flagged as phishing."
    # Fetch and parse the HTML content once
    soup = get_soup_from_url(url)

    features = [
        sfh(url, soup),            # SFH
        get_placeholder_feature(), # popUpWidnow
        ssl_final_state(url),      # SSLfinal_State
        get_placeholder_feature(), # Request_URL
        url_of_anchor(url, soup),  # URL_of_Anchor
        get_placeholder_feature(), # web_traffic
        url_length(url),           # URL_Length
        age_of_domain(url),        # age_of_domain
        having_ip_address(url)     # having_IP_Address
    ]
  
    features_df = pd.DataFrame([features], columns=FEATURE_NAMES)
    prediction = loaded_model.predict(features_df)
    return prediction[0], features_df, None # None indicates no override message

# Streamlit UI 
st.set_page_config(page_title="Phishing Website Detector", page_icon="🎣")
st.title("🎣 Phishing Website Detector")
st.write(
    "Enter a URL to check if it's a potential phishing site. "
    "This app uses a pre-trained Support Vector Machine (SVM) model to make a prediction."
)
st.info("This model now uses a specific set of 9 features for its prediction.")
st.warning("Disclaimer: This is a demonstration and may not be 100% accurate. Some features are simplified.")

url_input = st.text_input("Enter the URL to check:", "http://www.google.com/")

if st.button("Analyze URL"):
    if url_input:
        with st.spinner("Analyzing... This may take a moment."):
            try:
                prediction, features_df, override_message = extract_features_and_predict(url_input)
                
                st.subheader("Analysis Result")
                if override_message:
                    st.warning(override_message)

                if prediction == 0:
                    st.error("Prediction: This website is likely PHISHING. (Danger)")
                else:
                    st.success("Prediction: This website is likely LEGITIMATE. (Safe)")

                with st.expander("Show Extracted Features"):
                    if override_message:
                        st.write("The model's prediction was bypassed. The result is based on the triggered rule.")
                    else:
                        st.write("The model made its prediction based on the following 9 features:")
                    st.dataframe(features_df)

            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
    else:
        st.warning("Please enter a URL to analyze.")
