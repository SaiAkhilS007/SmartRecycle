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
from sklearn.neighbors import NearestNeighbors


# Force TensorFlow to use CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Set page configuration
st.set_page_config(
    page_title="🌍 SmartRecycle 🌍 - AI-Powered Waste Management System",
    page_icon="♻️",
    layout="wide",
)

# API Keys
google_api_key = "AIzaSyAIP4GUoss0Z8bm6e9j7g4kaoWe0yu-tC8"
youtube_api_key = "AIzaSyAqrbUiRO5WD800M8vnJLbPxKVd2gl6SzE"
gmaps = googlemaps.Client(key=google_api_key)

# Load Model
CLASSIFIER_MODEL_PATH = "model/Resnet_Neural_Network_model.h5"
model = load_model(CLASSIFIER_MODEL_PATH)

# Waste categories
categories = ['cardboard', 'plastic', 'glass', 'medical', 'paper',
              'e-waste', 'organic_waste', 'textiles', 'metal', 'wood']

# Load ResNet50 feature extractor
feature_extractor = ResNet50(weights="imagenet", include_top=False, pooling="avg")

# File path for the CSV containing drop-off locations
DROP_OFF_LOCATIONS_CSV_PATH = os.path.join("DataSet", "final_maryland_drop_off_locations_with_coordinates.csv")

# Load the drop-off locations dataset
try:
    drop_off_locations = pd.read_csv(DROP_OFF_LOCATIONS_CSV_PATH)
    drop_off_locations.columns = drop_off_locations.columns.str.strip()  # Clean column names
except Exception as e:
    drop_off_locations = None
    st.error(f"Error loading the drop-off locations CSV file: {e}")

# Fetch YouTube videos
def fetch_top_youtube_videos(waste_category, intent):
    query = f"{waste_category} {intent} tutorial"
    url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&maxResults=5&q={query}&type=video&key={youtube_api_key}"
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

# Cosine similarity calculation
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


# Image preprocessing
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

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

# Set up session state for intent, PIN code, and radius
if "intent" not in st.session_state:
    st.session_state.intent = None
if "predicted_category" not in st.session_state:
    st.session_state.predicted_category = None
if "keyword_input" not in st.session_state:
    st.session_state.keyword_input = ""

def find_nearest_drop_off_location(zip_code, category, radius):
    # Get coordinates of the user's ZIP code
    user_lat, user_lng = get_coordinates_from_zip(zip_code)
    if user_lat is None or user_lng is None:
        st.write("Unable to find coordinates for the given ZIP code. Please enter a Baltimore ZIP code.")
        return []

    # Check if drop-off locations are available
    if drop_off_locations is not None:
        # Filter the drop-off locations based on the category
        filtered_locations = drop_off_locations[drop_off_locations['category'].str.contains(category, case=False, na=False)]

        # Ensure there are some filtered locations available
        if filtered_locations.empty:
            st.write("No drop-off locations found for this category.")
            return []

        # Extract latitude and longitude for filtered locations
        drop_off_coords = filtered_locations[['latitude', 'longitude']].values

        # Initialize KNN to find the nearest drop-off locations
        knn = NearestNeighbors(n_neighbors=3, metric='euclidean')  # We can adjust the number of neighbors as needed
        knn.fit(drop_off_coords)

        # Query the nearest neighbors based on the user's location
        user_coords = np.array([[user_lat, user_lng]])
        distances, indices = knn.kneighbors(user_coords)

        # Return the nearest locations
        nearest_locations = []
        for idx in indices[0]:
            nearest_locations.append({
                "name": filtered_locations.iloc[idx]["name"],
                "address": filtered_locations.iloc[idx]["address"],
                "distance": distances[0][list(indices[0]).index(idx)]  # Distance to the drop-off location
            })

        return nearest_locations
    else:
        st.write("Drop-off locations data is not available.")
        return []


# Main App
st.title("🌍 SmartRecycle 🌍")

# Image uploader
uploaded_image = st.file_uploader("Upload an image of the waste", type=["jpg", "jpeg", "png"])

if uploaded_image:
    # Predict waste category
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    processed_image = preprocess_image(image)
    features = feature_extractor.predict(processed_image)
    predictions = model.predict(features)
    category_index = np.argmax(predictions, axis=1)[0]
    st.session_state.predicted_category = categories[category_index]
    st.write(f"Predicted Waste Category: {st.session_state.predicted_category}")

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

    # Handle intent actions
    if st.session_state.intent in ["reuse", "recycle"]:
        st.subheader("🎥 Video Tutorials:")
        user_description = st.text_area("Describe what you'd like to do with the waste:")

        if user_description:
            videos = fetch_top_youtube_videos(st.session_state.predicted_category, st.session_state.intent)

            if videos:
                # Calculate cosine similarity for each video
                similarities = calculate_cosine_similarity(user_description, videos)

                # Add similarity scores to video details
                for i, video in enumerate(videos):
                    video["similarity"] = similarities[i]

                # Sort videos by similarity
                videos = sorted(videos, key=lambda x: x["similarity"], reverse=True)

                # Display top 3 videos
                for video in videos[:3]:
                    st.markdown(
                        f"""
                        <div style="display: flex; align-items: center; margin-bottom: 10px;">
                            <img src="{video['thumbnail']}" alt="Video Thumbnail" style="width: 120px; height: 90px; margin-right: 10px;">
                            <a href="{video['link']}" target="_blank">{video['title']}</a><br>
                            <span>Cosine Similarity: {video['similarity']:.2f}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.write("No videos found.")
    elif st.session_state.intent == "disposal":
        st.subheader("Provide Location for Nearby Centers:")
        zip_code = st.text_input("Enter your ZIP code:")
        radius = st.slider("Search Radius (in miles)", 1, 50, 10)
        if st.button("Find Centers"):
            if zip_code:
                nearest_locations = find_nearest_drop_off_location(zip_code, st.session_state.predicted_category, radius)
                if nearest_locations:
                    for location in nearest_locations:
                        st.write(f"*{location['name']}*")
                        st.write(f"Address: {location['address']}")
                        st.write(f"Distance: {location['distance']:.2f} miles")
                else:
                    st.write("We are not in this area,Sorry!")
            else:
                st.write("Please enter a valid ZIP code.")