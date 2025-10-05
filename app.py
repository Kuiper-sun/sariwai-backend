# app.py (FINAL VERSION)

import os
import time
from flask import Flask, request, jsonify
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from PIL import Image
import torch

app = Flask(__name__)
MODEL_PATH = "Kuiper-sun/sariwai-rt-detr-v2" 

try:
    image_processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
    model = AutoModelForObjectDetection.from_pretrained(MODEL_PATH)
    print("✅ Object Detection Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

def apply_freshness_rules(eye_status, gill_status):
    hierarchy = {'fresh': 0, 'not-fresh': 1, 'old': 2}
    eye_found = eye_status != "Not Found"
    gill_found = gill_status != "Not Found"

    if not eye_found and not gill_found: return "Undetermined"

    eye_level = hierarchy.get(eye_status.lower().replace('_', '-'), -1) if eye_found else -1
    gill_level = hierarchy.get(gill_status.lower().replace('_', '-'), -1) if gill_found else -1

    final_level = -1
    if eye_found and gill_found: final_level = max(eye_level, gill_level)
    elif eye_found: final_level = eye_level
    else: final_level = gill_level

    if final_level == 0: return 'Fresh'
    elif final_level == 1: return 'Not Fresh'
    elif final_level == 2: return 'Old'
    else: return 'Undetermined'

@app.route('/healthz')
def healthz():
    return "OK", 200

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        image = Image.open(file.stream).convert("RGB")
        inputs = image_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        results = image_processor.post_process_object_detection(outputs, threshold=0.2, target_sizes=target_sizes)[0]

        print("\n" + "="*50)
        print("            STARTING NEW PREDICTION ANALYSIS")
        print("="*50)
        
        # LOGIC TO HANDLE NO DETECTIONS
        if not results or not results["scores"].nelement():
             print("‼️  CRITICAL: The model detected NOTHING above the threshold.")
             print("="*50 + "\n")
             # Return a specific response for this case
             return jsonify({
                'status': 'No Fish Detected',
                'eye_prediction': 'Not Found',
                'gill_prediction': 'Not Found',
                'eye_score': -1.0,
                'gill_score': -1.0,
            })

        print(f"✅  Model found {len(results['scores'])} potential objects:")
        for i, score in enumerate(results["scores"]):
            label_id = results["labels"][i].item()
            label = model.config.id2label[label_id]
            print(f"  - Detection: Label = '{label}', Confidence = {score.item():.4f}")
        
        best_eye = {'score': -1.0, 'status': 'Not Found'}
        best_gill = {'score': -1.0, 'status': 'Not Found'}

        for score, label_id, box in zip(results["scores"], results["labels"], results["boxes"]):
            label = model.config.id2label[label_id.item()]
            
            if 'eye' in label.lower() and score > best_eye['score']:
                best_eye['score'] = score.item()
                best_eye['status'] = label.rsplit('_', 1)[0]

            if 'gill' in label.lower() and score > best_gill['score']:
                best_gill['score'] = score.item()
                best_gill['status'] = label.rsplit('_', 1)[0]
        
        final_status = apply_freshness_rules(best_eye['status'], best_gill['status'])
        
        end_time = time.time()
        duration = end_time - start_time

        print("\n--- Analysis Summary ---")
        print(f"Best Eye Found: {best_eye['status']} (Score: {best_eye['score']:.4f})")
        print(f"Best Gill Found: {best_gill['status']} (Score: {best_gill['score']:.4f})")
        print(f"Final Determined Status: {final_status}")
        print("-" * 20)
        print(f"⏱️  Total Execution Time: {duration:.4f} seconds")
        print("="*50 + "\n")

        return jsonify({
            'status': final_status,
            'eye_prediction': best_eye['status'],
            'gill_prediction': best_gill['status'],
            'eye_score': best_eye['score'],
            'gill_score': best_gill['score'],
        })

    except Exception as e:
        print(f"❌ An error occurred during prediction: {str(e)}")
        return jsonify({'error': f"An error occurred during prediction: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)