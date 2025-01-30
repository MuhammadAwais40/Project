
# Import libraries
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import requests
from PIL import Image
from io import BytesIO
from gtts import gTTS
import os
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns

# Pexels API Key (replace with your API key)
PEXELS_API_KEY = 'eUiSujGMJgWtYh9kSKj9yRtuvBPWGf24oYLMyXuvCPfq4hJva7W04VHt'

# Function to speak text using gTTS
def speak_text(text):
    """
    Generate and play a speech audio file using gTTS.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
        temp_path = temp_audio_file.name  # Get the path to the file

    try:
        # Generate speech audio
        tts = gTTS(text)
        tts.save(temp_path)  # Save audio to the temporary file
        st.audio(temp_path, format="audio/mp3")  # Play the audio in Streamlit
    except Exception as e:
        st.error(f"Error generating speech: {e}")
    finally:
        # Clean up: remove the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

# Function to fetch car images
def fetch_car_image(car_name, car_model, car_year, num_images=1):
    """
    Fetch car images using Pexels API.
    """
    headers = {"Authorization": PEXELS_API_KEY}
    query = f"{car_year} {car_name} {car_model} car"
    url = f"https://api.pexels.com/v1/search?query={query}&per_page={num_images}"
    
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        if data.get("photos"):
            return [photo["src"]["medium"] for photo in data["photos"]]
    return None  # Return None if no image is found

# Convert predicted rating to star icons
def render_stars(rating):
    """
    Converts a numerical rating (e.g., 4.3) into a star rating visualization.
    """
    full_stars = int(rating)  # Number of full stars
    half_star = 1 if rating - full_stars >= 0.5 else 0  # One half star if applicable
    empty_stars = 5 - full_stars - half_star  # Remaining empty stars

    # Build star display using Unicode or emoji
    stars = "⭐" * full_stars + "🌑" * half_star + "⚪" * empty_stars
    return stars

# Load dataset
car_data = pd.read_excel("E:\\cars_datasets.xlsx")

# Feature and target setup
features = ['Car Names', 'Model', 'Year', 'Mileage', 'Engine', 'Engine Size', 'Power',
            'Transmission', 'Body Type', 'Wheel Size', 'Trim', 'Color']
target_price = 'Price'
target_rating = 'Rating'

X = car_data[features]
y_price = car_data[target_price]
y_rating = car_data[target_rating]

# Define categorical and numerical columns
categorical_cols = ['Car Names', 'Model', 'Engine', 'Transmission', 'Body Type', 'Trim', 'Color']
numerical_cols = ['Year', 'Mileage', 'Engine Size', 'Power', 'Wheel Size']

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

# Random Forest models
price_model = RandomForestRegressor(n_estimators=100, random_state=42)
rating_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Pipelines
pipeline_price = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', price_model)
])

pipeline_rating = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', rating_model)
])

# Split dataset
X_train, X_test, y_price_train, y_price_test = train_test_split(X, y_price, test_size=0.2, random_state=42)
_, _, y_rating_train, y_rating_test = train_test_split(X, y_rating, test_size=0.2, random_state=42)

# Train models
pipeline_price.fit(X_train, y_price_train)
pipeline_rating.fit(X_train, y_rating_train)

# Save models
joblib.dump(pipeline_price, "price_model.pkl")
joblib.dump(pipeline_rating, "rating_model.pkl")

# Load models
price_model = joblib.load("price_model.pkl")
rating_model = joblib.load("rating_model.pkl")

# App title
st.title("🚗 Car Details and Information")
st.markdown("Enter 🚓 car 📋 details to predict 💲 price, ⭐ rating, 🎤 voice and 🖼️ view images 🚕.")

# Sidebar inputs for car features
st.sidebar.title("🚓🚕🚙 Car Details")
st.sidebar.subheader("🔍 Search your cars")

car_name = st.sidebar.text_input("Car Name", placeholder="e.g., Honda Civic")
car_model = st.sidebar.text_input("Car Model", placeholder="e.g., TypeR")
car_year = st.sidebar.number_input("Car Year", min_value=1900, max_value=2026, value=2020)
mileage = st.sidebar.number_input("Mileage (in km)", 0, 300000, 10000)
engine = st.sidebar.selectbox("Engine Type", ['Gasoline', 'Petrol', 'Diesel', 'Hybrid', 'Electric'])
engine_size = st.sidebar.slider("Engine Size (L)", 1.0, 5.0, 2.0)
power = st.sidebar.number_input("Power (HP)", 50, 600, 150)
transmission = st.sidebar.selectbox("Transmission", ['Manual', 'Automatic', 'CVT'])
body_type = st.sidebar.selectbox("Body Type", ['Sedan', 'SUV', 'Hatchback', 'Coupe', 'Convertible'])
wheel_size = st.sidebar.slider("Wheel Size (inches)", 14, 20, 16)
trim = st.sidebar.selectbox("Trim", ['Base', 'Premium', 'Sport', 'Luxury'])
color = st.sidebar.selectbox("Color", ['Red', 'Blue', 'Black', 'White', 'Silver'])

# Advanced options
st.sidebar.subheader("💡 Advanced Options")
num_images = st.sidebar.number_input("Number of Images to Fetch", 1, 5, 1)
show_evaluation = st.sidebar.checkbox("Show Model Evaluation Metrics")

# Predict button
if st.sidebar.button("Predict 🚓"):
    # Prepare input data
    input_data = pd.DataFrame({
        'Car Names': [car_name],
        'Model': [car_model], 
        'Year': [car_year],
        'Mileage': [mileage],
        'Engine': [engine],
        'Engine Size': [engine_size],
        'Power': [power],
        'Transmission': [transmission],
        'Body Type': [body_type],
        'Wheel Size': [wheel_size],
        'Trim': [trim],
        'Color': [color]
    })
    
    # Predictions
    predicted_price = price_model.predict(input_data)[0]
    predicted_rating = rating_model.predict(input_data)[0]

    # Display results
    st.subheader("✨ Prediction Results")
    st.write(f"**💲 Predicted Price:** ${predicted_price:,.2f}")
    st.write(f"**⭐ Predicted Rating:** {predicted_rating:.1f} / 5.0")
    st.markdown(f"**💫 Rating Visualization:** {render_stars(predicted_rating)}")

    # Generate and play the voice output
    voice_text = (
        f"The car you selected is a {car_year} {car_name} {car_model}. "
        f"The predicted price is {predicted_price:,.2f} dollars, and the rating is {predicted_rating:.1f} out of 5."
    )
    speak_text(voice_text)

    # Fetch and display car images
    image_urls = fetch_car_image(car_name, car_model, car_year, num_images)
    if image_urls:
        st.subheader("🖼️ Car Images")
        for url in image_urls:
            st.image(url, caption=f"{car_year} {car_name} {car_model}", use_column_width=True)
    else:
        st.error("❌ No image found for this car. Try different details.")

    # Model evaluation metrics
    if show_evaluation:
        st.subheader("📊 Model Evaluation Metrics")
        y_price_pred = price_model.predict(X_test)
        y_rating_pred = rating_model.predict(X_test)

        st.write("**💲 Price Model Evaluation**")
        st.write(f"🧿 RMSE: {mean_squared_error(y_price_test, y_price_pred, squared=False):.2f}")
        st.write(f"💯 R² Score: {r2_score(y_price_test, y_price_pred):.2f}")

        st.write("**⭐ Rating Model Evaluation**")
        st.write(f"🧿 RMSE: {mean_squared_error(y_rating_test, y_rating_pred, squared=False):.2f}")
        st.write(f"💯 R² Score: {r2_score(y_rating_test, y_rating_pred):.2f}")

        # Visualization
        st.subheader("📈 Data Visualization")
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        sns.histplot(y_price_test, kde=True, ax=ax[0])
        ax[0].set_title('Price Distribution')
        sns.scatterplot(x=y_price_test, y=y_price_pred, ax=ax[1])
        ax[1].set_title('Actual vs Predicted Price')
        st.pyplot(fig)

# Additional information
st.info("This application provides advanced car details and informations. 🚘")
