import os
import logging
from flask import Flask, request, render_template, redirect, url_for, flash
import torch
from PIL import Image
import io
import base64
from ultralytics import YOLO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.secret_key = 'supersecretkey'

logging.basicConfig(level=logging.DEBUG)

# Load YOLOv8 model
model_path = os.path.join(os.getcwd(), 'yolov8n.pt')
logging.debug(f"Loading YOLO model from {model_path}")
model = YOLO(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        flash('No image part in the request')
        return redirect(request.url)

    file = request.files['image']

    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)

    if file:
        logging.debug("Image file received for prediction")
        image = Image.open(file.stream).convert("RGB")
        results = model(image)

        if not results:
            flash("No objects detected in the image.")
            return render_template('index.html', prediction_text='No objects detected.')

        predictions = results[0].boxes.xyxy  # Get the bounding boxes
        if len(predictions) == 0:
            flash("No objects detected in the image.")
            return render_template('index.html', prediction_text='No objects detected.')

        predictions_str = '\n'.join([str(pred) for pred in predictions])
        image_with_boxes = results[0].plot()  # Render the boxes on the image
        img_io = io.BytesIO()
        Image.fromarray(image_with_boxes).save(img_io, 'JPEG')
        img_io.seek(0)
        img_base64 = base64.b64encode(img_io.getvalue()).decode('ascii')

        return render_template('index.html', prediction_text=predictions_str, image_data=img_base64)

if __name__ == "__main__":
    app.run(debug=True)
