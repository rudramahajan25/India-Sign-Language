from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import threading

app = Flask(__name__)

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Global variables
cap = None
camera_running = False

# Function to capture frames from the camera, perform hand detection, and return processed frames
def detect_hands():
    global cap, camera_running

    cap = cv2.VideoCapture(0)  # Use 0 for the default camera
    camera_running = True

    while camera_running:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the RGB image with Mediapipe Hands
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the hands
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Encode the frame as JPEG
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()

        # Yield the frame to be sent to the client
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    if camera_running:
        return Response(detect_hands(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return ''

@app.route('/start_detection')
def start_detection():
    global camera_running
    if not camera_running:
        threading.Thread(target=detect_hands).start()  # Run in a separate thread
        camera_running = True
    return 'Hand detection started'

@app.route('/stop_detection')
def stop_detection():
    global cap, camera_running
    camera_running = False
    if cap is not None:
        cap.release()
    return 'Hand detection stopped'

if __name__ == '__main__':
    app.run(debug=True)
