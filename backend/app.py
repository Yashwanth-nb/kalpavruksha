import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
from ultralytics import YOLO
import requests
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": [
            "https://kalpavruksha-lake.vercel.app",
            "https://kalpavruksha-01-krfn.vercel.app",
            "https://*.vercel.app",  # Allow all Vercel preview URLs
            "http://localhost:5173",
            "http://localhost:3000"
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Accept"],
        "supports_credentials": True
    }
}) # Allow all origins for development

# --- Model Loading ---
# Model 1: Detection Model (finds disease regions)
# Model 2: Classification Model (classifies entire image)
MODEL_PATH_1 = 'bestcoconutdisease.pt'
MODEL_PATH_2 = 'best.pt'

try:
    # Load detection model with safe globals
    torch.serialization.add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])
    model1 = YOLO(MODEL_PATH_1)
    model1_type = 'detection'
    print("Detection model loaded successfully!")
except Exception as e:
    print(f"Error loading detection model: {e}")
    model1 = None
    model1_type = None

try:
    # Load classification model with safe globals
    torch.serialization.add_safe_globals(['ultralytics.nn.tasks.ClassificationModel'])
    model2 = YOLO(MODEL_PATH_2)
    model2_type = 'classify'
    print("Classification model loaded successfully!")
except Exception as e:
    print(f"Error loading classification model: {e}")
    model2 = None
    model2_type = None

# --- Class Names ---
# Model 1: Detection - 5 disease classes
class_names_1 = ['bud root dropping', 'bud rot', 'gray leaf spot', 'leaf rot', 'stembleeding']
# Model 2: Classification - 6 disease classes
class_names_2 = ['caterpillars', 'drying', 'flaccidity', 'healthy', 'leaflet', 'yellowing']


@app.route('/predict', methods=['POST'])
def predict():
    if model1 is None and model2 is None:
        return jsonify({'error': 'Models not loaded'}), 500
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Read image file
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        all_detections = []
        
        # Run Model 1 (Detection Model)
        if model1 is not None and model1_type == 'detection':
            results1 = model1.predict(image, verbose=False)
            if results1 and results1[0].boxes and len(results1[0].boxes) > 0:
                boxes1 = results1[0].boxes
                for i in range(len(boxes1)):
                    conf = boxes1.conf[i].item()
                    cls_index = int(boxes1.cls[i].item())
                    all_detections.append({
                        'class': class_names_1[cls_index],
                        'confidence': conf,
                        'model': 'detection'
                    })
                    print(f"Detection model found: {class_names_1[cls_index]} ({conf:.2f})")
        
        # Run Model 2 (Classification Model)
        if model2 is not None and model2_type == 'classify':
            results2 = model2.predict(image, verbose=False)
            # Classification results are in probs, not boxes
            if results2 and hasattr(results2[0], 'probs'):
                probs = results2[0].probs
                top5_indices = probs.top5
                top5_conf = probs.top5conf
                
                for idx, conf in zip(top5_indices, top5_conf):
                    cls_index = int(idx)
                    confidence = float(conf)
                    disease = class_names_2[cls_index]
                    
                    # Only add if confidence > 0.1 and not healthy (unless it's the top prediction)
                    if confidence > 0.1:
                        all_detections.append({
                            'class': disease,
                            'confidence': confidence,
                            'model': 'classification'
                        })
                        print(f"Classification model found: {disease} ({confidence:.2f})")
        
        # Find the detection with highest confidence across both models
        # Filter out 'healthy' unless it's the only high-confidence result
        non_healthy = [d for d in all_detections if d['class'] != 'healthy']
        
        if non_healthy:
            best_detection = max(non_healthy, key=lambda x: x['confidence'])
            predicted_class = best_detection['class']
            confidence = best_detection['confidence']
            
            print(f"Final Prediction: {predicted_class}, Confidence: {confidence:.2f}, Model: {best_detection['model']}")
            
            return jsonify({
                'prediction': predicted_class,
                'confidence': confidence,
                'all_detections': all_detections,
                'total_diseases': len(non_healthy)
            })
        elif all_detections:
            # Only healthy detections
            best_detection = max(all_detections, key=lambda x: x['confidence'])
            print(f"Final Prediction: healthy, Confidence: {best_detection['confidence']:.2f}")
            return jsonify({
                'prediction': 'healthy',
                'confidence': best_detection['confidence'],
                'all_detections': all_detections,
                'total_diseases': 0
            })
        
        # If no detections are found at all
        print("No predictions from either model.")
        return jsonify({
            'prediction': 'healthy',
            'confidence': 0.5,
            'all_detections': [],
            'total_diseases': 0
        })

    except Exception as e:
        # Log the full error to the terminal for debugging
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Make sure to use a different port than your React app (Vite default is 5173)
    app.run(debug=True, port=5000)
