import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
import pickle
from PIL import Image

model = load_model(r'C:\Users\rohit\Downloads\model.h5')

with open(r'C:\Users\rohit\Downloads\features_densenet201 (1).pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

max_length = 34

feature_extractor = tf.keras.applications.InceptionV3(include_top=False, pooling='avg')

def preprocess_image(img: Image.Image):
    img = img.resize((299, 299)).convert("RGB")
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)
    return img_array

def Generate_caption(img: Image.Image) -> str:
    photo = preprocess_image(img)
    features = feature_extractor.predict(photo)

    input_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([input_text])[0]
        sequence = tf.keras.preprocessing.sequence.pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([features, sequence], verbose=0)
        yhat = np.argmax(yhat)

        word = None
        for w, index in tokenizer.word_index.items():
            if index == yhat:
                word = w
                break

        if word is None:
            break
        input_text += ' ' + word
        if word == 'endseq':
            break

    final_caption = input_text.replace('startseq', '').replace('endseq', '').strip()
    return final_caption
