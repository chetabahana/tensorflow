# Defiine params and lib
import requests
from io import BytesIO

from PIL import Image
import numpy as np

# Parameters
input_size = (150,150)

#define input shape
channel = (3,)
input_shape = input_size + channel

#define labels
labels = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# # Define preprocess function
def preprocess(img,input_size):
    nimg = img.convert('RGB').resize(input_size, resample= 0)
    img_arr = (np.array(nimg))/255
    return img_arr

def reshape(imgs_arr):
    return np.stack(imgs_arr, axis=0)


# # Load models
from tensorflow.keras.models import load_model

# ada 2 cara load model, jika cara pertama berhasil maka bisa lasngusng di lanjutkan ke fungsi prediksi
MODEL_PATH = 'model/medium_project/model.h5'
model = load_model(MODEL_PATH,compile=False)

# # Predict the image
# read image
im = Image.open('contoh_prediksi.jpg')
X = preprocess(im,input_size)
X = reshape([X])
y = model.predict(X)

print( labels[np.argmax(y)], np.max(y) )
print( labels[np.argmax(y)], np.max(y) )

# read image
im = Image.open('dataset/train/dandelion/2522454811_f87af57d8b.jpg')
X = preprocess(im,input_size)
X = reshape([X])
y = model.predict(X)

print( labels[np.argmax(y)], np.max(y) )
