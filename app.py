from flask import Flask, render_template, request
import torch 
import cv2
import numpy as np

app = Flask(__name__)

# Load YOLOv7 model
model = torch.hub.load('WongKinYiu/yolov7', 'yolov7')

# Load weights
weights_path = 'D:/Documents/GitHub/yolov7_tracking-cars/tc_weights.pt'
model.load_state_dict(torch.load(weights_path))

@app.route('/')
def man():
    return render_template('catch_cars.html')

@app.route('/result', methods=['POST'])
def home():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    
    if file.filename == '':
        return "No selected file"
    
    if file:
        # Read the image from the file
        image_np = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        # Perform inference
        result = model(img)

        # Display results (modify as needed)
        print(result)

        return render_template('detect.html', data=result)

if __name__ == "__main__":
    app.run(debug=True)
