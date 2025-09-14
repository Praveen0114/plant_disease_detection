import streamlit as st
import os
import google.generativeai as genai
from datetime import datetime, timedelta
import csv
import pandas as pd
from PIL import Image
import io
import numpy as np
import tensorflow as tf
import requests  

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    st.warning("dotenv module not found. Please make sure to install it using 'pip install python-dotenv'.")

csv_file_path = 'question_history.csv'

def store_question(question):
    try:
        current_time = datetime.now()
        expiration_time = current_time + timedelta(days=7)  

        data = [question, expiration_time.strftime('%Y-%m-%d')]

        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)
    except Exception as e:
        st.error(f"An error occurred while storing the question: {e}")

def read_question_history():
    try:
        with open(csv_file_path, mode='r') as file:
            reader = csv.reader(file)
            history_data = list(reader)
        return history_data
    except FileNotFoundError:
        return []

def ask_bard(query):
    API_KEY = " Enter ur API Key here" 

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "query": query
    }

    response = requests.post(
        "https://bard.googleapis.com/v1/query", 
        json=payload, headers=headers
    )
    
    if response.status_code == 200:
        return response.json().get('response', 'No response from Bard')
    else:
        return "Error: Unable to get response from Bard"

def configure_api_key():
    try:
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    except Exception as e:
        st.warning(f"An error occurred while configuring the API key: {e}")

def gemini_pro(input_text, prompt):
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content([prompt, input_text])
        return response.text
    except Exception as e:
        st.error(f"An error occurred during chat: {e}")

@st.cache_resource
def load_model(path):
    # Xception Model
    xception_model = tf.keras.models.Sequential([
        tf.keras.applications.xception.Xception(include_top=False, weights='imagenet', input_shape=(512, 512, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(4, activation='softmax')
    ])

    # DenseNet Model
    densenet_model = tf.keras.models.Sequential([
        tf.keras.applications.densenet.DenseNet121(include_top=False, weights='imagenet', input_shape=(512, 512, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(4, activation='softmax')
    ])

    inputs = tf.keras.Input(shape=(512, 512, 3))
    xception_output = xception_model(inputs)
    densenet_output = densenet_model(inputs)
    outputs = tf.keras.layers.average([densenet_output, xception_output])

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.load_weights(path)
    
    return model

# Helper function to clean and preprocess the image
def clean_image(image):
    image = image.convert('RGB')
    image = image.resize((512, 512))  # Resize image to match model input
    image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    return np.expand_dims(image, axis=0)  # Add batch dimension

# Helper function to get prediction from the model
def get_prediction(model, image):
    predictions = model.predict(image)
    return predictions, predictions[0]

# Helper function to interpret the prediction results
def make_results(predictions, predictions_arr):
    classes = ['Healthy', 'Diseased Class 1', 'Diseased Class 2', 'Diseased Class 3']  # Example class labels
    prediction = np.argmax(predictions_arr)
    status = 'Healthy' if prediction == 0 else 'Diseased'
    return {'status': status, 'prediction': classes[prediction]}

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

model = load_model('model_weights.h5')

st.title('Plant Disease Detection and Chatbot')
st.write("Upload your plant's leaf image to get a health prediction, or ask our chatbot any questions about plant diseases!")

st.sidebar.title("Plant Disease Information")
st.sidebar.write("""
### Instructions:
1. **Upload Image**: Choose an image of the plant's leaf that you want to analyze.
2. **Get Prediction**: The system will analyze the image and provide a diagnosis.
3. **Chatbot Assistance**: Use the chatbot to get more detailed information about plant diseases or follow-up questions.

### Disease Applications:
- **Agricultural Diagnostics**: Helps farmers quickly identify and respond to crop diseases.
- **Research**: Facilitates research in plant pathology by providing an automated disease prediction system.
- **Educational Tool**: Assists students and educators in learning about plant diseases.
""")

uploaded_file = st.file_uploader("Choose an Image file", type=["png", "jpg"])

if uploaded_file:
    progress = st.text("Processing Image")
    my_bar = st.progress(0)
    
    image = Image.open(io.BytesIO(uploaded_file.read()))
    st.image(np.array(image.resize((700, 400), Image.Resampling.LANCZOS)), width=None)

    my_bar.progress(40)
    
    image = clean_image(image)
    
    predictions, predictions_arr = get_prediction(model, image)
    my_bar.progress(70)
    
    result = make_results(predictions, predictions_arr)
    my_bar.progress(100)
    
    st.write(f"The plant {result['status']} with a {result['prediction']} prediction.")
    
    progress.empty()
    my_bar.empty()

st.subheader('Chat with the Plant Disease Expert')

prompt = """Provide an informative and detailed response to the user's query."""

input_text = st.text_input('Ask a question about plant diseases:')

if st.button('Send Message'):
    if input_text:
        configure_api_key()
        store_question(input_text)
        response = gemini_pro(input_text, prompt)
        st.write("Response:", response)
    else:
        st.warning("Please enter a question.")

history_data = read_question_history()
if history_data:
    st.subheader("Recent Search History")
    df = pd.DataFrame(history_data, columns=["Question", "Expire On"])
    st.write(df.tail(50).iloc[::-1])
else:
    st.info("No question history available.")
