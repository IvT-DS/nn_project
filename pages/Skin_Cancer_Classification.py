import streamlit as st
from PIL import Image
import json
import torch
import torch.nn as nn
import time
from torchvision import io
from torchvision import transforms as T
from torchvision.models import resnet34, ResNet34_Weights
from torch.nn import functional as F
import requests
from io import BytesIO

device = torch.device('cpu')
model = resnet34(weights=ResNet34_Weights.DEFAULT)
model.fc = nn.Linear(512, 1)
model.load_state_dict(torch.load('model_binary.pt', map_location=device))
model.to(device)

idx2class2 = {0: 'benign', 1: 'malignant'}

class_translation2 = {
    "benign": "доброкачественная опухоль",
    "malignant": "злокачественная опухоль",
}

# with open("imagenet_class_index.json") as file:
#     labels = json.load(file)

# Функция предсказания 
# def get_prediction(image):
#     transform = T.Compose([
#         T.Resize((224, 224)),
#         T.ToTensor(),
#     ])
#     image = transform(image).unsqueeze(0)
#     model.eval()
#     with torch.no_grad():
#         pred = torch.argmax(F.softmax(model(image)[0], dim=0)).item()
#         class_name = labels[str(pred)][1]
#     return class_name

def get_prediction(image):
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])
    img = transform(image).unsqueeze(0).to(device)
    model.eval()
    start_time = time.time() # Старт отсчета времени работы модели 

    with torch.no_grad():
        output = model(img)
        pred_class = torch.sigmoid(output).round().item()

        class_name = idx2class2[pred_class]
        class_translation = class_translation2[class_name]
        another_class_name = "malignant" if class_name == "benign" else "benign"
        another_class_translation = class_translation2[another_class_name]

        probabilities = torch.sigmoid(output)

        proba1 = torch.sigmoid(output).item() * 100 # вероятность что злокачественная
        proba2 = 100 - proba1 # Для бинарной классификации, вероятность второго класса равна 1 - вероятность первого класса

    end_time = time.time() # Конец отсчета времени работы модели 
    elapsed_time = end_time - start_time # Время работы модели 
    
    if class_name == "benign":
        class_result = f"На изображении с вероятностью {proba2:.3f}% {class_translation}"
        overall_result = f"Вероятность, что на изображении {another_class_translation} равна {proba1:.3f}%"
    else:
        class_result = f"На изображении с вероятностью {proba1:.3f}% {class_translation}"
        overall_result = f"Вероятность, что на изображении {another_class_translation} равна {proba2:.3f}%"
    

    return class_result, elapsed_time, overall_result

st.write("## Skin Cancer Classification 🫶")
st.sidebar.write("Choose application above")
st.markdown(
        """
        Этот streamlit-сервис позволит оценить доброкачественность образований на коже по фото\n
        *Не является медицинской рекомендацией, обязательно проконсультируйтесь с врачом*👩‍⚕️
    """
)
button_style = """
    <style>
    .center-align {
        display: flex;
        justify-content: center;
    }
    </style>
"""
image_source = st.radio("Choose the option of uploading the image:", ("File", "URL"))

try:
    if image_source == "File":
        uploaded_files = st.file_uploader("Upload the image", type=["jpg", "png", "jpeg"], accept_multiple_files=True) 
        if uploaded_files:
            for uploaded_file in uploaded_files:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded image", use_column_width=True)
                if st.button(f"Предсказать класс для {uploaded_file.name}"):
                    predicted_class, elapsed_time, overall_result = get_prediction(image)
                    st.success(predicted_class)
                    st.success(overall_result)
                    st.info(f"Время ответа модели: {elapsed_time:.4f} секунд")
    else:
        url = st.text_input("Enter the URL of image...")
        if url:
            response = requests.get(url)
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))
                st.image(image, caption="Uploaded image", use_column_width=True)
                if st.button(f"Предсказать класс"):
                    predicted_class, elapsed_time, overall_result = get_prediction(image)
                    st.success(predicted_class)
                    st.success(overall_result)
                    st.info(f"Время ответа модели: {elapsed_time:.4f} секунд")
            else:
                st.error("Ошибка при получении изображения. Убедитесь, что введена правильная ссылка.")
except Exception as e:
    st.error(f"Произошла ошибка при обработке изображения {str(e)}")

st.markdown(button_style, unsafe_allow_html=True)  # Применяем стиль к кнопке
        