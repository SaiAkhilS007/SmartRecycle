import os
import streamlit as st
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input  # type: ignore
from tensorflow.keras.models import load_model  # type: ignore
from PIL import Image
import googlemaps
import requests
from geopy.distance import geodesic

# Force TensorFlow to use CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Set page configuration
st.set_page_config(
    page_title="🌍 SmartRecycle 🌍 - AI-Powered Waste Management System",
    page_icon="♻️",
    layout="wide"
)

# Paths to the model and resources
CLASSIFIER_MODEL_PATH = "model/Resnet_Neural_Network_model.h5"

# Initialize API keys and clients
google_api_key = "AIzaSyAIP4GUoss0Z8bm6e9j7g4kaoWe0yu-tC8"
youtube_api_key = "AIzaSyAqrbUiRO5WD800M8vnJLbPxKVd2gl6SzE"
gmaps = googlemaps.Client(key=google_api_key)

# Waste categories
categories = ['Cardboard', 'Plastic', 'Glass', 'Medical', 'Paper',
              'E-Waste', 'Organic Waste', 'Textiles', 'Metal', 'Wood']

# Load the trained classification model
model = load_model(CLASSIFIER_MODEL_PATH)

# Load ResNet50 feature extractor
feature_extractor = ResNet50(weights="imagenet", include_top=False, pooling="avg")

def fetch_top_youtube_videos(waste_category, intent):
    query = f"{waste_category} {intent} tutorial"
    url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&maxResults=3&q={query}&type=video&key={youtube_api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for HTTP codes >= 400
        results = response.json().get("items", [])
        if not results:
            st.warning("No video tutorials found. Please try another category.")
        return [
            {
                "title": item["snippet"]["title"],
                "link": f"https://www.youtube.com/watch?v={item['id']['videoId']}",
                "thumbnail": item["snippet"]["thumbnails"]["medium"]["url"],
            }
            for item in results
        ]
    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP error occurred while fetching YouTube videos: {http_err}")
    except requests.exceptions.RequestException as req_err:
        st.error(f"Error occurred while fetching YouTube videos: {req_err}")
    except Exception as e:
        st.error("An unexpected error occurred. Please try again later.")
    return []  # Return an empty list if an error occurs

def fetch_nearby_locations(user_location, waste_category, intent, radius):
    intent_keyword = {"reuse": "donation center", "recycle": "recycling center", "disposal": "disposal center"}.get(intent, "")
    keyword = f"{waste_category} {intent_keyword}"
    try:
        places = gmaps.places_nearby(location=user_location, radius=radius, keyword=keyword)
        results = places.get("results", [])
        if not results:
            st.warning("No nearby locations found. Try increasing the search radius or using a different category.")
        return [
            {
                "name": place["name"],
                "address": place["vicinity"],
                "distance": geodesic(
                    user_location,
                    (place["geometry"]["location"]["lat"], place["geometry"]["location"]["lng"])
                ).miles,
            }
            for place in results[:3]
        ]
    except googlemaps.exceptions.ApiError as api_err:
        st.error(f"API error occurred while fetching locations: {api_err}")
    except googlemaps.exceptions.TransportError as transport_err:
        st.error(f"Transport error occurred while fetching locations: {transport_err}")
    except requests.exceptions.RequestException as req_err:
        st.error(f"Error occurred while fetching locations: {req_err}")
    except Exception as e:
        st.error("An unexpected error occurred. Please try again later.")
    return []  # Return an empty list if an error occurs


# Custom CSS for button layout and styling
st.markdown(
    """
    <style>
    body {
        font-family: 'Roboto', sans-serif;
        background-color: #121212;
        color: white;
        margin: 0;
        padding: 0;
    }
    h1, h2, h3 {
        text-align: center;
        color: white;
    }
    .container {
        background-color: #1E1E2F;
        padding: 20px;
        border-radius: 15px;
        margin: 20px auto;
        max-width: 800px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.5);
    }
    .button-container {
        display: flex;
        justify-content: space-between; /* Space buttons across the screen */
        padding: 20px;
        margin-top: 20px;
    }
    .custom-button {
        background-color: #8E44AD;
        color: white;
        padding: 30px 60px; /* Increase button size */
        font-size: 24px; /* Bigger font size */
        font-weight: bold;
        border: none;
        border-radius: 25px; /* Rounded button */
        cursor: pointer;
        transition: all 0.3s ease;
        text-align: center;
        text-decoration: none;
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.2);
    }
    .custom-button:hover {
        background-color: #9B59B6;
        transform: scale(1.1); /* Slight enlargement on hover */
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.3);
    }
    .custom-button.reuse {
        background-color: #4CAF50; /* Green */
    }
    .custom-button.reuse:hover {
        background-color: #388E3C; /* Darker green */
    }
    .custom-button.recycle {
        background-color: #2196F3; /* Blue */
    }
    .custom-button.recycle:hover {
        background-color: #1E88E5; /* Darker blue */
    }
    .custom-button.disposal {
        background-color: #f44336; /* Red */
    }
    .custom-button.disposal:hover {
        background-color: #d32f2f; /* Darker red */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Session state for intent, ZIP code, and radius
if "intent" not in st.session_state:
    st.session_state.intent = None
if "zip_code" not in st.session_state:
    st.session_state.zip_code = ""
if "radius" not in st.session_state:
    st.session_state.radius = 10  # Default radius in miles

# App layout
st.markdown("<div class='container'><h1>🌍 SmartRecycle</h1></div>", unsafe_allow_html=True)

# Add instructions
# Add an instructions section
st.markdown(
    """
    <div class='container'>
        <h2>🛠️ How to Use This Application</h2>
        <ol style='text-align: left; font-size: 18px; line-height: 1.8; color: #ffffff;'>
            <li>Upload an image of waste by clicking on the <strong>"Choose an Image File"</strong> button.</li>
            <li>Wait for the system to process and classify the type of waste.</li>
            <li>Once the waste category is displayed, choose one of the following actions:
                <ul>
                    <li><strong>♻️ Reuse:</strong> Get video recommendations for reusing the waste.</li>
                    <li><strong>♻️ Recycle:</strong> Find video tutorials for recycling the waste.</li>
                    <li><strong>🗑️ Disposal:</strong> Enter your ZIP code to find nearby disposal locations.</li>
                </ul>
            </li>
            <li>Follow the recommendations to manage your waste responsibly.</li>
        </ol>
        <p style='text-align: left; font-size: 16px; line-height: 1.5; color: #a8a8a8;'>
            <strong>Note:</strong> Ensure that the image clearly shows the waste item for better classification results.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# CSS for instructions styling
st.markdown(
    """
    <style>
    .container h2 {
        color: #4CAF50; /* Green heading for instructions */
        text-align: center;
        margin-bottom: 20px;
    }
    .container ol {
        padding-left: 20px;
    }
    .container ul {
        padding-left: 40px;
        list-style-type: disc;
    }
    .container ul li {
        margin-top: 10px;
    }
    .container p {
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# File uploader for waste image
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

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

    st.markdown(f"<div class='container'><h2>Predicted Waste Category: <span style='color: #4CAF50;'>{category}</span></h2></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <h2 style='text-align: left; color: #FF0000; font-family: Arial, sans-serif;'>
            What would you like to do?
        </h2>
        """,
        unsafe_allow_html=True,
    )

    # HTML Buttons with form actions
    st.markdown(
        f"""
        <form method="post">
            <button class="custom-button reuse" name="intent" value="reuse">♻️ Reuse</button>
            <button class="custom-button recycle" name="intent" value="recycle">♻️ Recycle</button>
            <button class="custom-button disposal" name="intent" value="disposal">🗑️ Disposal</button>
        </form>
        """,
        unsafe_allow_html=True,
    )

    # Detect button clicks
    form_intent = st.query_params.get("intent", None)
    if form_intent:
        st.session_state.intent = form_intent

    # Reuse and Recycle functionality
    if st.session_state.intent in ["reuse", "recycle"]:
        videos = fetch_top_youtube_videos(category, st.session_state.intent)

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

    # Disposal functionality
    elif st.session_state.intent == "disposal":
        st.session_state.zip_code = st.text_input("Enter your ZIP code:", value=st.session_state.zip_code)
        st.session_state.radius = st.slider("Search radius (in miles):", 1, 50, st.session_state.radius)

        if st.button("Find Disposal Locations"):
            def get_coordinates_from_zip(zip_code):
                geocode_result = gmaps.geocode(zip_code)
                location = geocode_result[0]['geometry']['location']
                return (location['lat'], location['lng'])

            if st.session_state.zip_code:
                user_location = get_coordinates_from_zip(st.session_state.zip_code)
                locations = fetch_nearby_locations(user_location, category, st.session_state.intent, st.session_state.radius * 1609.34)

                st.subheader("📍 Nearby Locations:")
                for loc in locations:
                    st.write(f"**{loc['name']}** - {loc['address']} ({loc['distance']:.2f} miles away)")



                    # Add custom CSS for buttons
