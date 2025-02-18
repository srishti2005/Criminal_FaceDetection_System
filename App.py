

from flask import Flask, render_template, request, redirect, url_for
import cv2
import face_recognition
import os
import csv
import numpy as np

app = Flask(__name__)

# Load criminal dataset
def load_criminal_dataset():
    criminal_encodings = []
    criminal_names = []
    criminal_details = []

    with open('criminals.csv', mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            name, crime,age, image_path = row
            image = face_recognition.load_image_file(os.path.join('criminal_dataset', image_path))
            encoding = face_recognition.face_encodings(image)[0]
            criminal_encodings.append(encoding)
            criminal_names.append(name)
            criminal_details.append({'name': name, 'crime': crime, 'age': age})

    return criminal_encodings, criminal_names, criminal_details

criminal_encodings, criminal_names, criminal_details = load_criminal_dataset()

from flask import Flask, render_template, request, url_for
import os
import datetime

app = Flask(__name__, static_folder='static')

# Example structure for detected results
detected_results = []

@app.route("/", methods=["GET", "POST"])

def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        # Save the uploaded video
        video_path = os.path.join('static', 'uploaded_video.mp4')
        file.save(video_path)

        # Process the video
        results = process_video(video_path)

        return render_template('index.html', results=results)

    return render_template('index.html', results=None)

import cv2
import numpy as np
import face_recognition
import os

import cv2
import numpy as np
import face_recognition
import os

def process_video(video_path):
    results = []
    video_capture = cv2.VideoCapture(video_path)

    if not video_capture.isOpened():
        print("Error: Could not open video file.")
        return results

    frame_count = 0
    skip_frames = 30  # Process every 30th frame to reduce redundancy
    detected_faces = set()  # Store names of detected criminals to avoid duplicates

    # Get video frame rate and calculate time per frame
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    time_per_frame = 1 / fps if fps > 0 else 0

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break  # Exit loop if no more frames

        frame_count += 1
        if frame_count % skip_frames != 0:
            continue  # Skip frames

        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(criminal_encodings, face_encoding)
            face_distances = face_recognition.face_distance(criminal_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = criminal_names[best_match_index]
                details = criminal_details[best_match_index]

                if name in detected_faces:
                    continue  # Skip if already detected in this session

                detected_faces.add(name)

                # Calculate timestamp when the criminal was first spotted
                timestamp = frame_count * time_per_frame
                minutes = int(timestamp // 60)
                seconds = int(timestamp % 60)
                timestamp_str = f"{minutes:02d}:{seconds:02d}"

                # Draw rectangle
                top, right, bottom, left = face_location
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

                # Save the frame and store the relative path
                frame_filename = f'{name}_frame.jpg'
                frame_path = os.path.join('static', 'detected_frames', frame_filename)
                cv2.imwrite(frame_path, frame)

                # Add timestamp to the results
                results.append({
                    'frame_path': f'detected_frames/{frame_filename}',
                    'details': details,
                    'timestamp': timestamp_str
                })

    video_capture.release()
    return results




if __name__ == '__main__':
    app.run(debug=True)
 

