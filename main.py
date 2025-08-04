import os
import streamlit as st
import tensorflow as tf
import numpy as np
from disease_info import fetch_description, get_openai_client
from PIL import Image

# Load OpenAI client once
client = get_openai_client()

# TensorFlow model prediction function (using PIL for uploaded file)
def model_prediction(test_image):
    model = tf.keras.models.load_model('trained_model.keras')
    # Streamlit UploadedFile supports .read(); open with PIL then convert
    pil_img = Image.open(test_image).convert("RGB")
    pil_img = pil_img.resize((128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(pil_img)
    input_arr = np.expand_dims(input_arr, axis=0)  # batch dimension
    prediction = model.predict(input_arr)
    result_index = int(np.argmax(prediction))
    return result_index

# Disease classes (must match model's output ordering)
CLASS_NAME = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

def humanize_label(raw_label: str) -> str:
    return raw_label.replace("___", " ").replace("_", " ")

# Sidebar and navigation
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "Leaf Image.jpeg"
    st.image(image_path, use_container_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍

    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works 
    1. **Upload Image:** Go to the **Disease Recognition** page in the Sidebar and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes Deep learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!
    """)

elif app_mode == "About":
    st.header("About")
    image_path = "dataset-Image.png"
    st.image(image_path, use_container_width=True)
    st.markdown("""
    #### About Dataset
    This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this GitHub repo. This dataset consists of about 87K RGB images of healthy and diseased crop leaves which are categorized into 38 different classes. The total dataset is divided into an 80/20 ratio of training and validation sets preserving the directory structure. A new directory containing 33 test images is created later for prediction purposes.

    #### Content
    1. Train (70295 images)  
    2. Valid (17572 images)  
    3. Test (33 images)
    """)

elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")

    # Warn if OpenAI key missing
    if client is None:
        st.warning("OPENAI_API_KEY is not set; disease descriptions will not be fetched from OpenAI.")

    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])
    if test_image:
        st.image(test_image, use_container_width=True)

    if st.button("Predict"):
        if not test_image:
            st.error("Please upload an image first.")
        else:
            with st.spinner("Predicting disease and fetching description..."):
                try:
                    result_index = model_prediction(test_image)
                except Exception as e:
                    st.error(f"Model prediction failed: {e}")
                    st.stop()

                if result_index < 0 or result_index >= len(CLASS_NAME):
                    st.error("Model output index out of range.")
                    st.stop()

                predicted_raw = CLASS_NAME[result_index]
                readable = humanize_label(predicted_raw)
                st.success(f"Model is predicting it's: {readable}")

                # Fetch description, with timeout safety
                try:
                    description = fetch_description(predicted_raw, client=client)
                    if not description:
                        st.warning("Received empty description from OpenAI.")
                    st.markdown(f"### About {readable}")
                    st.write(description)
                except Exception as ex:
                    st.error(f"Failed to get description: {ex}")
