from flask import Flask, request, render_template, Response, jsonify
import os
import model_logic as ai
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import base64

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    try:
        if "image" not in request.files:
            return "No file uploaded", 400

        file = request.files["image"]
        if file.filename == "":
            return "No selected file", 400

        filename = secure_filename(file.filename)
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(image_path)

        detections, grade, fruit_name, confidence, quality = ai.analyze(image_path)
        
        web_image_path = image_path.replace('\\', '/')
        
        confidence_percent = round(confidence * 100, 2) if confidence <= 1 else round(confidence, 2)

        return render_template("result.html",
                             image_path=web_image_path,
                             detections=detections,
                             grade=grade,
                             fruit_name=fruit_name,
                             confidence=confidence_percent,
                             quality=quality)
    
    except Exception as e:
        return f"An error occurred: {str(e)}", 500

@app.route("/webcam")
def webcam():
    return render_template("webcam.html")

@app.route("/analyze_webcam", methods=["POST"])
def analyze_webcam():
    try:
        data = request.json
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Save temporary image
        temp_path = os.path.join(UPLOAD_FOLDER, "webcam_temp.jpg")
        with open(temp_path, 'wb') as f:
            f.write(image_bytes)
        
        # Analyze the image
        detections, grade, fruit_name, confidence, quality = ai.analyze(temp_path)
        
        confidence_percent = round(confidence * 100, 2) if confidence <= 1 else round(confidence, 2)
        
        return jsonify({
            'success': True,
            'grade': grade,
            'fruit_name': fruit_name,
            'confidence': confidence_percent,
            'quality': quality,
            'detections': detections
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)