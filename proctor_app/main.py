import cv2
import dlib
import numpy as np
import time
from head_pose import estimate_head_pose,get_head_pose_angles,get_face_direction
from detect_eyes import detect_gaze


def draw_frame_center():
    # Determine frame center
    frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)  # (x_center, y_center)
    rect_width, rect_height = 300, 300  # Define the size of the rectangle

    # Calculate the top-left corner of the rectangle
    rect_x1 = frame_center[0] - rect_width // 2
    rect_y1 = frame_center[1] - rect_height // 2

    return rect_width, rect_height,rect_x1, rect_y1


def capture_closer_face(t):
    cv2.rectangle(frame, (x, y), (x+w, y+h), rectangle_color, 2)
    distance = (size_threshold) / w
    cv2.putText(frame, f"Distance: {distance:.2f}mm", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if t is None:
        t = time.time()  # Start timing when face gets close
    elif (time.time() - t) >= timeframe:
        # If face has been close for longer than the threshold, capture the frame
        cv2.imwrite('close_face.jpg', frame)
        t = None  # Reset the timer

def face_direction():
    rotation_vector, translation_vector = estimate_head_pose(shape, frame)
    angles = get_head_pose_angles(rotation_vector, translation_vector)

    # Define thresholds for angles (in degrees)
    threshold_yaw_left = -135  # Head turns left
    threshold_yaw_right = 135  # Head turns right
    threshold_pitch_up = 170   # Head looks up
    threshold_pitch_down = -140 # Head looks down

    
    if angles[1] < threshold_yaw_left:
        direction = 'Left'
    elif angles[1] > threshold_yaw_right:
        direction = 'Right'
    elif angles[0] > threshold_pitch_up:
        direction = 'Up'
    elif angles[0] < threshold_pitch_down:
        direction = 'Down'
    else:
        direction = 'Center'
    
    return angles,direction





# Load the pre-trained Haar Cascade model for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
predictor = dlib.shape_predictor("./proctor_app/files/shape_predictor_68_face_landmarks.dat")  # You need to download this file

# Start capturing video from the first webcam on your computer
cap = cv2.VideoCapture(0)

start_time = time.time()

timeframe = 5  # Timeframe in seconds

rectangle_color = (0, 0, 255)  # Blue for no face detected

size_threshold =  500 * 500 # Threshold for the face size (width * height)

time_face_close = None  # Time when face is detected close

time_eyes_non_centered=None


while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale( gray,
        
        scaleFactor=1.5,
        minNeighbors=7,     
        minSize=(40, 40))

   
    rect_width, rect_height,rect_x1, rect_y1=draw_frame_center()

    if len(faces)==0:
        cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x1 + rect_width, rect_y1 + rect_height), (0, 0, 255), 2)
        if (time.time() - start_time) >= timeframe:
            cv2.imwrite(f'No_detected_face.jpg', frame)
    else:
        for (x, y, w, h) in faces:

            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            landmarks = predictor(gray, rect)
            shape = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]

            
            face_size = w * h
            if face_size > size_threshold:
                capture_closer_face(time_face_close)
            else:
                time_face_close = None  # Reset the timer if face is not close

                
                rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
                landmarks = predictor(gray, rect)
                direction = get_face_direction(landmarks,gray)

                # Check if the face is centered
                is_centered = direction=="Center, Center"

                
                rectangle_color = (0, 0, 255) if not is_centered else (0, 255, 0)  # Red if not centered, green if centered

                # Draw rectangle around the face
                cv2.rectangle(frame, (x, y), (x+w, y+h), rectangle_color, 2)
            

                cv2.putText(frame, f"Pose: {direction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                

                

                if not is_centered:
                    # Capture the frame
                    if (time.time() - start_time) >= timeframe:
                        cv2.imwrite(f'head_pose_{direction}.jpg', frame)

                else:
                   
                    roi_gray = gray[y:y+h, x:x+w]
                    eyes = eye_cascade.detectMultiScale(roi_gray)
                    

                    for (ex, ey, ew, eh) in eyes:
                        eye_region = roi_gray[ey:ey+eh, ex:ex+ew]
                        
                        gaze_direction = detect_gaze(eye_region)
                     
                        if gaze_direction !="Unable to determine gaze":
                            print(gaze_direction,time.time())  # or use this information as needed
                            horizontal, vertical=gaze_direction
                            cv2.putText(frame, f"Horizontal Eye: {horizontal}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            cv2.putText(frame, f"Vertical Eye: {vertical}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                        
                cv2.putText(frame, f"Direction: {direction}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                


    cv2.imshow('Head Pose Estimation', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close all windows
cap.release()
cv2.destroyAllWindows()


