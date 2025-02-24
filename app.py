from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
import secrets
import glob
import numpy as np
import pandas as pd

from loss import focal_tversky, tversky
from model import Model
from model_segmentation import Model_Seg
from prediction import Prediction
from plot_mri import plot_scan
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from tensorflow.keras.models import model_from_json
from flask import Flask
import threading
import time

secret_key = secrets.token_hex(16)

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
# UPLOAD_FOLDER = 'static/uploads/'
UPLOAD_FOLDER = 'static/uploads/'

 
app.secret_key = secret_key
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif', 'tif'])


def background_task():
    # Perform the background task (e.g., model inference)
    time.sleep(5)  # Simulate a task taking time
    print("Background task completed")

# @app.route('/')
def index():
    # Start the background task in a separate thread
    thread = threading.Thread(target=background_task)
    thread.start()
    return "Task started in background!"


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
# @app.route('/predict/',methods=['GET','POST'])

def get_prediction(filename):
        # model = Model.get_model()
        # model_seg = Model_Seg.get_model()
        with open('./saved_models/resnet-50-MRI.json', 'r') as json_file:
            json_Model = json_file.read()
        model = model_from_json(json_Model)
        model.load_weights('./saved_models/weights.hdf5')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
        print("----------------------------------------------------")
        print("Classification model loaded and compiled successfully!")
        print("----------------------------------------------------")

        with open('./saved_models/ResUNet-MRI.json', 'r') as json_file:
            json_savedModel = json_file.read()
        model_seg = model_from_json(json_savedModel)
        model_seg.load_weights('./saved_models/weights_seg.hdf5')
        model_seg.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        adam = tf.keras.optimizers.Adam(learning_rate=0.05, epsilon=0.1)
        model_seg.compile(optimizer=adam, loss=focal_tversky, metrics=[tversky])
        print("----------------------------------------------------")
        print("Segmentation model loaded and compiled successfully!")
        print("----------------------------------------------------\n")

        path = [f"./static/uploads/{filename}"]
        obj = Prediction(path, model, model_seg)
        result = obj.make_prediction()
        if(result[2] == 0):
            return [] , False
        return result, True
    
def clean_dir():
    files = glob.glob("static/uploads/*")
    for f in files:
        os.remove(f)
    files = glob.glob("static/predicted/*")
    for f in files:
        os.remove(f)
 
@app.route('/')
def home():
    clean_dir()
    return render_template('index.html')
 
@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
        
    file = request.files['file']
    if file.filename == '':
        flash('No image selected')
        return redirect(request.url)
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        result, detected = get_prediction(filename)
        if(detected):
            df = pd.DataFrame([result])
            df.columns = ["image_path", "predicted_mask", "has_mask"]
            plot_scan(df)
            return render_template('predict.html', filename=filename)
        else:
            clean_dir()
            flash("Hurray! No Tumor Detected")
            return render_template('index.html')
    else:
        flash('Allowed image types are - tif, png, jpg, jpeg, gif')
        return redirect(request.url)
        
@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response
 
@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)
 
if __name__ == '__main__':
    app.run(debug=True)

