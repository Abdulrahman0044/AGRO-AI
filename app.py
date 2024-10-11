import streamlit as st
import torch
import CNN
from torchvision import transforms
from PIL import Image
import openai
import torchvision.transforms.functional as TF
import numpy as np

# OpenAI setup
openai.api_key = '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'

# Define the model class 
class PlantDiseaseNet(torch.nn.Module):
   

# Load the plant disease prediction model
model=CNN.CNN(39)
model_path = "plant_disease_model_1_latest.pt"
model.eval()

# Define a mapping from class indices to disease names
class_to_disease = {
                  0: 'Apple___Apple_scab',
                  1: 'Apple___Black_rot',
                  2: 'Apple___Cedar_apple_rust',
                  3: 'Apple___healthy',
                  4: 'Background_without_leaves',
                  5: 'Blueberry___healthy',
                  6: 'Cherry___Powdery_mildew',
                  7: 'Cherry___healthy',
                  8: 'Corn___Cercospora_leaf_spot Gray_leaf_spot',
                  9: 'Corn___Common_rust',
                  10: 'Corn___Northern_Leaf_Blight',
                  11: 'Corn___healthy',
                  12: 'Grape___Black_rot',
                  13: 'Grape___Esca_(Black_Measles)',
                  14: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                  15: 'Grape___healthy',
                  16: 'Orange___Haunglongbing_(Citrus_greening)',
                  17: 'Peach___Bacterial_spot',
                  18: 'Peach___healthy',
                  19: 'Pepper,_bell___Bacterial_spot',
                  20: 'Pepper,_bell___healthy',
                  21: 'Potato___Early_blight',
                  22: 'Potato___Late_blight',
                  23: 'Potato___healthy',
                  24: 'Raspberry___healthy',
                  25: 'Soybean___healthy',
                  26: 'Squash___Powdery_mildew',
                  27: 'Strawberry___Leaf_scorch',
                  28: 'Strawberry___healthy',
                  29: 'Tomato___Bacterial_spot',
                  30: 'Tomato___Early_blight',
                  31: 'Tomato___Late_blight',
                  32: 'Tomato___Leaf_Mold',
                  33: 'Tomato___Septoria_leaf_spot',
                  34: 'Tomato___Spider_mites Two-spotted_spider_mite',
                  35: 'Tomato___Target_Spot',
                  36: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                  37: 'Tomato___Tomato_mosaic_virus',
                  38: 'Tomato___healthy'
                  }
                

import time




# Define the rate limit 
rate_limit = 60  # 60 requests per minute

# Keep track of the last request time
last_request_time = None

def ask_openai_with_rate_limit(question):
    global last_request_time

    # Calculate the time elapsed since the last request
    if last_request_time is not None:
        time_elapsed = time.time() - last_request_time
        if time_elapsed < 60 / rate_limit:
            # Sleep to ensure rate limiting
            time.sleep((60 / rate_limit) - time_elapsed)

    # Make the API request
    response = openai.Completion.create(
        engine="davinci",
        prompt=question,
        max_tokens=150
    )

    # Update the last request time
    last_request_time = time.time()

    return response.choices[0].text.strip()




def predict_disease(image_path):
    # Load the image
    image = Image.open(image_path)
    
    # Resize the image
    image = image.resize((224, 224))
    
    # Convert the image to a tensor
    input_data = TF.to_tensor(image)
    
    # Adjust tensor dimensions
    input_data = input_data.view((-1, 3, 224, 224))
    
    # Make prediction
    with torch.no_grad():
        output = model(input_data)
    
    # Convert tensor to numpy array
    output = output.detach().numpy()
    
    # Get index of the max value
    index = np.argmax(output)
    
    return class_to_disease[index]

def main():
    st.title("FarmTrend")

    st.sidebar.header("Choose a Feature")
    choice = st.sidebar.radio("", ("Conversational Agent", "Plant Disease Prediction"))

    if choice == "Conversational Agent":
        st.header("AI Assistant")
        user_input = st.text_area("Ask a question on Agriculture in Africa:")
        if st.button("Ask"):
            response = ask_openai_with_rate_limit(user_input)
            st.write(response)

    elif choice == "Plant Disease Prediction":
        st.header("Plant Disease Classifier")
        uploaded_file = st.file_uploader("Upload an image of a plant", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image.", use_column_width=True)
            st.write("Classifying...")
            prediction = predict_disease(uploaded_file)
            st.write(f"Predicted Disease: {prediction}")

if __name__ == "__main__":
    main()
