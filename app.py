from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import os
from model_logic import FruitAI

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ai = FruitAI()

def gen_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success: break
        processed = ai.process_frame(frame)
        ret, buffer = cv2.imencode('.jpg', processed)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    camera.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    if file:
        path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(path)
        detections, grade, img = ai.predict_image(path)
        res_name = 'res_' + file.filename
        cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], res_name), img)
        return render_template('index.html', result=res_name, grade=grade, detections=detections)
    return redirect(url_for('index'))
app = app
if __name__ == '__main__':
    app.run(debug=True, threaded=True)