import os
import streamlit as st
import numpy as np
import pandas as pd 
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import load_model
from PIL import Image
import requests
import googlemaps
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from geopy.distance import geodesic
from sklearn.neighbors import NearestNeighbors

# Ensure CUDA is disabled (for running without GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Set page configuration
st.set_page_config(
    page_title="🌍 SmartRecycle 🌍 - AI-Powered Waste Management System",
    page_icon="♻️",
    layout="wide",
)

# API Keys
google_api_key = "AIzaSyANESeNA-wRkwU3XIDekR2gLaQ63cEeVos"
youtube_api_key = "AIzaSyCWGmxXEGYDvNYwOs2lNG8hZKMpmUGhfcY"
gmaps = googlemaps.Client(key=google_api_key)

# Load Model
model = load_model("model/Resnet_Neural_Network_model.h5")
feature_extractor = ResNet50(weights="imagenet", include_top=False, pooling="avg")

# Waste categories
categories = ['cardboard', 'plastic', 'glass', 'medical', 'paper',
              'e-waste', 'organic_waste', 'textiles', 'metal', 'Wood']

# Session state to manage intents
if "intent" not in st.session_state:
    st.session_state.intent = None
if "predicted_category" not in st.session_state:
    st.session_state.predicted_category = None
if "keyword_input" not in st.session_state:
    st.session_state.keyword_input = ""

# Helper Functions
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def fetch_youtube_videos(query, intent):
    # Adjust the query to make it more specific based on the intent
    if intent == "reuse":
        query = f"{query} how to reuse ideas"
    elif intent == "recycle":
        query = f"{query} recycling process tips"
    url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&q={query}&type=video&maxResults=3&key={youtube_api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        videos = response.json().get("items", [])
        return [
            {
                "title": v["snippet"]["title"],
                "link": f"https://www.youtube.com/watch?v={v['id']['videoId']}",
                "thumbnail": v["snippet"]["thumbnails"]["medium"]["url"],
            }
            for v in videos
        ]
    return []

def calculate_cosine_similarity(user_input, video_details):
    texts = [f"{v['title']}" for v in video_details]
    texts.append(user_input.lower())
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    return similarities.flatten()

def get_coordinates_from_zip(zip_code):
    geocode_result = gmaps.geocode(zip_code)
    if geocode_result:
        location = geocode_result[0]["geometry"]["location"]
        return location["lat"], location["lng"]
    return None, None

def find_nearest_drop_off_location(zip_code, category, radius):
    # Placeholder function to simulate nearest drop-off location search
    # You can integrate actual distance calculation here
    return []

# Main App
st.title("🌍 SmartRecycle 🌍 - AI-Powered Waste Management System")

# File uploader for the drop-off locations CSV
uploaded_file = st.file_uploader("Upload the drop-off locations CSV file", type=["csv"])

# Process the uploaded file
if uploaded_file is not None:
    try:
        # Read the CSV file into a DataFrame
        drop_off_locations = pd.read_csv(uploaded_file)
        
        # Clean column names (strip leading/trailing spaces)
        drop_off_locations.columns = drop_off_locations.columns.str.strip()
        
        # Show the first few rows of the uploaded dataset
        st.write("Preview of the uploaded drop-off locations dataset:")
        st.dataframe(drop_off_locations.head())  # Display first few rows of the DataFrame
        
    except Exception as e:
        st.error(f"Error reading the file: {e}")

# Image uploader
uploaded_image = st.file_uploader("Upload an image of the waste", type=["jpg", "jpeg", "png"])

if uploaded_image:
    # Predict waste category
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    processed_image = preprocess_image(image)
    features = feature_extractor.predict(processed_image)
    predictions = model.predict(features)
    category_index = np.argmax(predictions, axis=1)[0]
    st.session_state.predicted_category = categories[category_index]
    st.write(f"Predicted Waste Category: *{st.session_state.predicted_category}*")

    # Intent buttons
    st.subheader("What would you like to do?")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("♻️ Reuse"):
            st.session_state.intent = "reuse"
    with col2:
        if st.button("♻️ Recycle"):
            st.session_state.intent = "recycle"
    with col3:
        if st.button("🗑️ Disposal"):
            st.session_state.intent = "disposal"

    # Clear inputs when intent changes
    if "prev_intent" not in st.session_state or st.session_state.intent != st.session_state.prev_intent:
        st.session_state.prev_intent = st.session_state.intent
        st.session_state.keyword_input = ""  # Reset keyword input for reuse/recycle

    # Handle intent actions
    if st.session_state.intent in ["reuse", "recycle"]:
        st.subheader("Provide Keywords for Suggestions:")
        st.session_state.keyword_input = st.text_input("Enter keywords:", value=st.session_state.keyword_input)
        if st.button("Get Recommendations"):
            user_input = st.session_state.keyword_input
            query = f"{st.session_state.predicted_category} {user_input}"
            recommendations = fetch_youtube_videos(query, st.session_state.intent)
            if recommendations:
                for rec in recommendations:
                    st.markdown(
                        f"""
                        <div style="display: flex; align-items: center; margin-bottom: 10px;">
                            <img src="{rec['thumbnail']}" alt="Video Thumbnail" style="width: 120px; height: 90px; margin-right: 10px;">
                            <a href="{rec['link']}" target="_blank">{rec['title']}</a>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.write("No recommendations found.")

    elif st.session_state.intent == "disposal":
        st.subheader("Provide Location for Nearby Centers:")
        zip_code = st.text_input("Enter your ZIP code:")
        radius = st.slider("Search Radius (in miles)", 1, 50, 10)
        if st.button("Find Centers"):
            nearest_locations = find_nearest_drop_off_location(zip_code, st.session_state.predicted_category, radius)
            if nearest_locations:
                st.subheader("📍 Nearby Drop-off Locations")
                for loc in nearest_locations:
                    st.write(f"Name: {loc['name']}, Address: {loc['address']}, Phone: {loc['phone']}")
            else:
                st.write("No nearby drop-off locations found.")
