import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load pre-trained InceptionV3 model
image_model = InceptionV3(weights='imagenet')

# Remove the last layer (output softmax layer) from the InceptionV3
image_model = Model(image_model.input, image_model.layers[-2].output)

# Load pre-trained caption generation model
caption_model = tf.keras.models.load_model('model/caption_generator.h5')

# Load tokenizer
tokenizer = pd.read_csv('model/tokenizer.csv')
tokenizer = tokenizer['word'].values

# Set maximum length of the caption
max_length = 34

# Load image features
features = np.load('model/image_features.npy', allow_pickle=True).item()

# Function to preprocess image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(299, 299))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# Function to encode image into features
def encode_image(image):
    img = preprocess_image(image)
    feature = image_model.predict(img)
    feature = np.reshape(feature, feature.shape[1])
    return feature

# Function to generate caption
def predict_caption(model, image_path, tokenizer, max_length, features):
    in_text = '<start>'
    for _ in range(max_length):
        sequence = [tokenizer.word_index[w] for w in in_text.split() if w in tokenizer.word_index]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([features[image_path], sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word[yhat]
        in_text += ' ' + word
        if word == '<end>':
            break
    caption = in_text.split()
    caption = caption[1:-1]
    caption = ' '.join(caption)
    return caption

# Example usage
image_path = 'image.jpg'
caption = predict_caption(caption_model, image_path, tokenizer, max_length, features)
print(caption)
