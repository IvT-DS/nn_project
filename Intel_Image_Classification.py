import streamlit as st
from PIL import Image
import json
import torch
import torch.nn as nn
from torchvision import io
from torchvision import transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from torch.nn import functional as F
import requests
from io import BytesIO
import time
import numpy as np

device = torch.device('cpu')
model = resnet50(weights=ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(2048, 6)
model.load_state_dict(torch.load('model_multiclass_cpu.pt', map_location=device))
model.to(device)

class_translation = {
    "buildings": "Здания",
    "forest": "Лес",
    "glacier": "Ледник",
    "mountain": "Гора",
    "sea": "Море",
    "street": "Улица",
}
idx2class= {0: 'buildings',
                1: 'forest',
                2: 'glacier',
                3: 'mountain',
                4: 'sea',
                5: 'street'}

# Функция предсказания 
def get_prediction(image):
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])
    
    image = transform(image).unsqueeze(0).to(device)
    
    start_time = time.time() # Старт отсчета времени работы модели 
    model.to(device)
    model.eval()

    with torch.no_grad():
        output = model(image)  
        pred_class = torch.argmax(output).round().item() 
        probability = np.round(torch.softmax(output, dim=1)[0, pred_class].item() * 100, 2)  # Преобразуем вероятность в проценты
        probabilities = torch.softmax(output, dim=1)[0].tolist()
        class_probabilities = [
        f"{class_translation[idx2class[i]]}: {prob*100:.2f}%" for i, prob in enumerate(probabilities)]

        class_result = f"На изображении с вероятностью {probability}%: **{class_translation[idx2class[pred_class]]}**" # Класс, определенный моделью
        overall_result = f"Предсказания для всех классов: " + ", ".join(class_probabilities) # Вероятности принадлжености ко всем классам
    
    end_time = time.time() # Конец отсчета времени работы модели 
    elapsed_time = end_time - start_time # Время работы модели 

    return class_result, elapsed_time, overall_result  # f"На изображении : {class_translation[idx2class[pred_class]]}"

st.write("## Intel Image Classification 🏞️")
st.sidebar.write("Choose application above")
st.markdown(
        """
        Этот streamlit-сервис позволит определить, что изображено на картинке
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
image_source = st.radio("Choose the option of uploading the image:", ("File", "URL"))  # Меню выбора того как загрузить картинку: по ссылке или из файла

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
