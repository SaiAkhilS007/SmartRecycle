import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import load_model
from PIL import Image
import pickle
import googlemaps
import requests
from geopy.distance import geodesic

# Set page configuration at the very beginning
st.set_page_config(
    page_title="AI-Powered Waste Management System",  # Tab name
    page_icon="♻️",  # Tab icon
    layout="wide"  # Use wide layout for better spacing
)

# Paths to the model and resources
CLASSIFIER_MODEL_PATH = "model/Resnet_Neural_Network_model.h5"

# Initialize API keys and clients
google_api_key = "AIzaSyANESeNA-wRkwU3XIDekR2gLaQ63cEeVos"
custom_search_engine_id = "c784b70b531b748dc"
youtube_api_key = "AIzaSyCWGmxXEGYDvNYwOs2lNG8hZKMpmUGhfcY"
gmaps = googlemaps.Client(key=google_api_key)

# Waste categories
categories = ['cardboard', 'plastic', 'glass', 'medical', 'paper',
              'e-waste', 'organic_waste', 'textiles', 'metal', 'Wood']

# Load the trained classification model
model = load_model(CLASSIFIER_MODEL_PATH)

# Load ResNet50 feature extractor
feature_extractor = ResNet50(weights="imagenet", include_top=False, pooling="avg")

# Fetch top 3 articles
def fetch_top_articles(waste_category, intent):
    query = f"{waste_category} {intent} tips"
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&cx={custom_search_engine_id}&key={google_api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        results = response.json().get("items", [])
        return [{"title": item.get("title"), "link": item.get("link")} for item in results[:3]]
    except Exception as e:
        return []

# Fetch top 3 YouTube videos
def fetch_top_youtube_videos(waste_category, intent):
    query = f"{waste_category} {intent} tutorial"
    url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&maxResults=3&q={query}&type=video&key={youtube_api_key}"
    try:
        response = requests.get(url)
        results = response.json().get("items", [])
        return [
            {
                "title": item["snippet"]["title"],
                "link": f"https://www.youtube.com/watch?v={item['id']['videoId']}",
                "thumbnail": item["snippet"]["thumbnails"]["medium"]["url"],
            }
            for item in results
        ]
    except Exception as e:
        return []

# Fetch nearby locations
def fetch_nearby_locations(user_location, waste_category, intent, radius):
    intent_keyword = {"reuse": "donation center", "recycle": "recycling center", "disposal": "disposal center"}.get(intent, "")
    keyword = f"{waste_category} {intent_keyword}"
    try:
        places = gmaps.places_nearby(location=user_location, radius=radius, keyword=keyword)
        return [
            {
                "name": place["name"],
                "address": place["vicinity"],
                "distance": geodesic(user_location, (place["geometry"]["location"]["lat"], place["geometry"]["location"]["lng"])).miles,
            }
            for place in places.get("results", [])[:3]
        ]
    except Exception as e:
        return []

# Set up the app layout
st.title("🌍 AI-Powered Waste Management System")
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image
    def preprocess_image(img):
        img = img.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return preprocess_input(img_array)

    processed_image = preprocess_image(image)
    features = feature_extractor.predict(processed_image)
    predictions = model.predict(features)
    predicted_label = np.argmax(predictions, axis=1)[0]
    category = categories[predicted_label]

    st.subheader(f"Predicted Waste Category: **{category}**")

    # User inputs for intent and location
    intent = st.selectbox("What would you like to do?", ["reuse", "recycle", "disposal"])
    zip_code = st.text_input("Enter your ZIP code:")
    radius = st.slider("Search radius (in miles):", 1, 50, 10)

    if st.button("Find Recommendations"):
        def get_coordinates_from_zip(zip_code):
            geocode_result = gmaps.geocode(zip_code)
            location = geocode_result[0]['geometry']['location']
            return (location['lat'], location['lng'])

        if zip_code:
            user_location = get_coordinates_from_zip(zip_code)
            locations = fetch_nearby_locations(user_location, category, intent, radius * 1609.34)
            articles = fetch_top_articles(category, intent)
            videos = fetch_top_youtube_videos(category, intent)

            st.subheader("📍 Nearby Locations:")
            for loc in locations:
                st.write(f"**{loc['name']}** - {loc['address']} ({loc['distance']:.2f} miles away)")

            st.subheader("📚 Articles:")
            for article in articles:
                st.write(f"[{article['title']}]({article['link']})")

            st.subheader("🎥 Video Tutorials:")
            for video in videos:
                st.markdown(
                    f"""
                    <div style="display: flex; align-items: center; margin-bottom: 10px;">
                        <img src="{video['thumbnail']}" alt="Video Thumbnail" style="width: 120px; height: 90px; margin-right: 10px;">
                        <a href="{video['link']}" target="_blank">{video['title']}</a>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
