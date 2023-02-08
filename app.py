import copy
from io import BytesIO
import json
import os
from unittest import result
import torchvision.transforms as T
from torchvision import models
import torchvision.transforms as transforms
import torch.nn as nn
import torch
import csv
from PIL import Image
from flask import Flask, redirect, url_for, request, render_template, jsonify
from werkzeug.utils import secure_filename
# from gevent.pywsgi import WSGIServer

app = Flask(__name__)
txt_file = "src/class_list.txt"
with open('src/class_list.csv', newline='') as f:
    reader = csv.reader(f)
    imagenet_class_index = list(reader)[0]
    
model =models.efficientnet_b1(pretrained=False)
model.classifier[1] = nn.Linear(in_features=1280, out_features=196)
model = model.to('cpu')
path = 'outputs/model_196.pth'
model.load_state_dict(torch.load(path,map_location=torch.device('cpu'))["model_state_dict"])
model.eval()

def transform_image(image_bytes):
    my_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )
    ])
    image = Image.open(BytesIO(image_bytes))
    return my_transforms(image) #.unsqueeze(0)

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model(tensor[None, ...])
    _, predicted = torch.max(outputs, 1)

    return imagenet_class_index[predicted]

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        img_bytes = f.read()
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        predicted = get_prediction(image_bytes=img_bytes)
        return predicted
    return None

if __name__=="__main__":
    app.run(host='0.0.0.0', port=5000)