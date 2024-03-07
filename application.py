import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk

# Function to start the camera feed
def start_camera():
    global cap, camera_running
    cap = cv2.VideoCapture(0)  # 0 represents the default camera (usually the built-in webcam)
    camera_running = True
    show_camera_feed()

# Function to display the camera feed
def show_camera_feed():
    global camera_running
    if camera_running:
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            label.config(image=photo)
            label.photo = photo
            label.after(10, show_camera_feed)

# Function to stop the camera feed
def stop_camera():
    global cap, camera_running
    if camera_running:
        cap.release()
        camera_running = False
        label.config(image=None)

# Create the main application window
root = tk.Tk()
root.title("Camera Feed Application")
root.geometry("400x400")  # Set the initial window size

# Configure the style for buttons
style = ttk.Style()
style.configure("TButton", foreground="white", background="blue", font=("Helvetica", 12))

# Create a label widget for displaying the camera feed
label = ttk.Label(root)
label.pack(padx=10, pady=10, expand=True, fill="both")

# Create buttons to start and stop the camera feed
start_button = ttk.Button(root, text="Start Camera", command=start_camera)
start_button.pack(pady=10)
stop_button = ttk.Button(root, text="Stop Camera", command=stop_camera)
stop_button.pack()

# Initialize OpenCV variables
cap = None
camera_running = False

# Start the Tkinter event loop
root.mainloop()
