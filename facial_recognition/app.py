from flask import Flask, request, jsonify, render_template
import os
import face_recognition
import numpy as np
import cv2
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Define the folders for uploaded images and known faces
UPLOAD_FOLDER = './static/uploads'
KNOWN_FACES_FOLDER = './static/identified'
DETECTED_FACES_FOLDER = './static/detected'
VIDEO_FOLDER = './videos'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure folders exist
for folder in [UPLOAD_FOLDER, DETECTED_FACES_FOLDER, VIDEO_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Load known faces
known_encodings = []
known_names = []

for filename in os.listdir(KNOWN_FACES_FOLDER):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(KNOWN_FACES_FOLDER, filename)
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)
        if encoding:
            known_encodings.append(encoding[0])
            known_names.append(filename)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    # Perform face recognition
    uploaded_image = face_recognition.load_image_file(file_path)
    uploaded_encodings = face_recognition.face_encodings(uploaded_image)

    results = []
    
    # Detect faces
    face_locations = face_recognition.face_locations(uploaded_image)
    detected_faces = []

    for face_location in face_locations:
        top, right, bottom, left = face_location
        face_image = uploaded_image[top:bottom, left:right]

        # Save the detected face image using PIL
        detected_face_filename = f'detected_{filename}_{top}.png'
        detected_face_path = os.path.join(DETECTED_FACES_FOLDER, detected_face_filename)
        detected_faces.append({'path': f'/static/detected/{detected_face_filename}'})  # Fixed relative path for browser access

        detected_face_image = Image.fromarray(face_image)
        detected_face_image.save(detected_face_path)

    # Now check if there are matches for each uploaded face encoding
    for uploaded_encoding in uploaded_encodings:
        distances = face_recognition.face_distance(known_encodings, uploaded_encoding)
        top_matches = np.argsort(distances)[:5]

        has_match = False
        for match_index in top_matches:
            if distances[match_index] < 0.6:  # Threshold for a match
                results.append({
                    'detected_faces': detected_faces,  # Attach detected faces here once
                    'status': 'match',
                    'matched_image': known_names[match_index]
                })
                has_match = True
                break  # Stop after finding the first match for this face

        if not has_match:
            results.append({
                'detected_faces': detected_faces,  # Attach detected faces here once
                'status': 'no match',
                'matched_image': None
            })

    return jsonify({'results': results})

@app.route('/clean', methods=['POST'])
def clean_files():
    # Clean up uploaded images
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Clean up detected faces
    for filename in os.listdir(DETECTED_FACES_FOLDER):
        file_path = os.path.join(DETECTED_FACES_FOLDER, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    return jsonify({'status': 'success', 'message': 'All uploaded and detected images have been deleted.'})

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video uploaded'}), 400

    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No selected video'}), 400

    video_filename = secure_filename(video_file.filename)
    video_path = os.path.join(VIDEO_FOLDER, video_filename)
    video_file.save(video_path)

    detected_faces = []
    comparison_results = []

    video_capture = cv2.VideoCapture(video_path)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            top, right, bottom, left = face_location
            face_image = rgb_frame[top:bottom, left:right]
            detected_face_filename = f'detected_frame_{len(detected_faces)}.png'
            detected_face_path = os.path.join(DETECTED_FACES_FOLDER, detected_face_filename)
            detected_faces.append({'path': f'/static/detected/{detected_face_filename}'})

            detected_face_image = Image.fromarray(face_image)
            detected_face_image.save(detected_face_path)

            distances = face_recognition.face_distance(known_encodings, face_encoding)
            top_matches = np.argsort(distances)[:5]
            result = []

            for match_index in top_matches:
                if distances[match_index] < 0.6:  # Threshold for a match
                    result.append(known_names[match_index])

            comparison_results.append(result)

    video_capture.release()

    return jsonify({
        'detected_faces': detected_faces,
        'comparison_results': comparison_results
    })

if __name__ == "__main__":
    app.run(debug=True)

