import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# Function to start hand detection
def start_detection():
    global cap, camera_running, hands, camera_window
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera
    camera_running = True

    # Create a Toplevel window for the camera feed
    camera_window = tk.Toplevel(root)
    camera_window.title("Camera Feed")
    camera_window.geometry("640x480")


    while camera_running:
        ret, frame = cap.read()

        # Convert the BGR image to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Convert the grayscale image to RGB
        rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

        # Process the RGB image with Mediapipe Hands
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the hands
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get the bounding box of the hand
                h, w, c = frame.shape
                bbox = [
                    int(min(hand_landmarks.landmark[i].x * w for i in range(21))),
                    int(min(hand_landmarks.landmark[i].y * h for i in range(21))),
                    int(max(hand_landmarks.landmark[i].x * w for i in range(21))),
                    int(max(hand_landmarks.landmark[i].y * h for i in range(21)))
                ]

                # Crop the region of interest (ROI) for the hand
                hand_roi = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

                # Check if the hand_roi has a valid size before displaying it
                if hand_roi.shape[0] > 0 and hand_roi.shape[1] > 0:
                    # Display the cropped hand ROI in the camera window
                    cv2.imshow('Hand ROI', hand_roi)
                    cv2.waitKey(1)

        # Display the frame with hand landmarks
        cv2.imshow('Hand Tracking', frame)

    # Release the Toplevel camera window when camera detection stops
    camera_window.destroy()

# Function to stop hand detection
def stop_detection():
    global cap, camera_running
    if camera_running:
        cap.release()
        cv2.destroyAllWindows()
        camera_running = False

# Create the main application window
root = tk.Tk()
root.title("Hand Detection Application")
root.geometry("800x600")  # Set the initial window size

# Create a button to start hand detection
start_button = ttk.Button(root, text="Start Hand Detection", command=start_detection)
start_button.pack(pady=10)

# Create a button to stop hand detection
stop_button = ttk.Button(root, text="Stop Hand Detection", command=stop_detection)
stop_button.pack(pady=10)

# Initialize OpenCV variables
cap = None
camera_running = False

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Initialize camera window
camera_window = None

# Start the Tkinter event loop
root.mainloop()
