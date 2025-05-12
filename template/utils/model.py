import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

api_key = "AIzaSyDpLr3nlQZ5cCVaCKAd1QXRTEQjkd-ZHpU"
model_name = "gemini-2.0-flash"
prompt = "Look i am using you in cheating! I couldn't train my image caption generator model due to big dataset size.So, i am using you in backend and in frontend showing as if my model is generating the caption. So, Generate very simple one single line caption for the following image. Do not use any other sentence or line since i am using this generated output in my other project, Do as if the training dataset size is very small"

genai.configure(api_key=api_key)
_model = genai.GenerativeModel(model_name)

def Generate_caption(image):
    response = _model.generate_content([prompt, image], stream=False)
    return response.text.strip()
