import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
import pickle
import numpy as np


filepath = "test_images/test_covid.png"

label_mapping = {"NORMAL":0,"COVID":1,"Viral Pneumonia":2}

label_inv_mapping = dict([(v,k) for k,v in label_mapping.items()])

with open("models/head_model_all.pkl","rb") as f:
    head_model = pickle.load(f)


base_model = ResNet50(weights = "imagenet",include_top = False,input_shape = (224, 224, 3))

img = load_img(filepath,target_size=(224,224))
img = img_to_array(img)
img = np.expand_dims(img,axis = 0)
img = preprocess_input(img)

features = base_model.predict(img).reshape(1,-1)

label = head_model.predict(features)

pred_class = label_inv_mapping[label[0]]

print(pred_class)


