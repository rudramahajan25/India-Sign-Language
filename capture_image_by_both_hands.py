import cv2
import mediapipe as mp
import os

output_folder = "D:\VIT\ISL detction project\dataset\B"
os.makedirs(output_folder, exist_ok=True)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)  
image_counter = 1

while True:
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
        
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

      
            h, w, c = frame.shape
            bbox = [
                int(min(hand_landmarks.landmark[i].x * w for i in range(21))),
                int(min(hand_landmarks.landmark[i].y * h for i in range(21))),
                int(max(hand_landmarks.landmark[i].x * w for i in range(21))),
                int(max(hand_landmarks.landmark[i].y * h for i in range(21)))
            ]

         
            hand_roi = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

      
            if hand_roi.shape[0] > 0 and hand_roi.shape[1] > 0:
        
                if hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x < 0.5:
                    cv2.imshow('Left Hand ROI', hand_roi)
                else:
                    cv2.imshow('Right Hand ROI', hand_roi)

    cv2.imshow('Grayscale Image', gray_frame)
    cv2.imshow('Hand Tracking', frame)


    key = cv2.waitKey(1)
    if key == ord('c'):
        image_filename = os.path.join(output_folder, f'{image_counter}.png')
        cv2.imwrite(image_filename, frame)
        print(f"Image {image_counter} captured and saved as {image_filename}")
        image_counter += 1

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
