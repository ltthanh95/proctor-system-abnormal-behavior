import numpy as np
import cv2
import time
# def are_eyes_centered(shape, frame):
#     # Get the average position of the eyes
#     left_eye_center = np.mean([shape[i] for i in range(36, 42)], axis=0)
#     right_eye_center = np.mean([shape[i] for i in range(42, 48)], axis=0)
#     eyes_center = (left_eye_center + right_eye_center) / 2

#     # Get the width of the face
#     face_width = shape[16][0] - shape[0][0]  # Distance between the two sides of the face

#     # Check if the eyes are approximately in the middle of the face
#     # Allowing some margin for error
#     margin = face_width * 0.05  # 15% of face width as margin
#     face_center_x = frame.shape[1] / 2

#     return (face_center_x - margin) <= eyes_center[0] <= (face_center_x + margin)

# def get_eye_direction(shape, eye_indices):
#     # Calculate the horizontal and vertical center of the eye
#     eye_center = np.mean([shape[i] for i in eye_indices], axis=0)

#     # Calculate the left, right, top, and bottom points of the eye
#     left_corner = shape[eye_indices[0]]
#     right_corner = shape[eye_indices[-1]]
#     top_corner = min(shape[i][1] for i in eye_indices)
#     bottom_corner = max(shape[i][1] for i in eye_indices)

#     # Determine horizontal direction
#     if eye_center[0] < left_corner[0]:
#         horiz_direction = "Left"
#     elif eye_center[0] > right_corner[0]:
#         horiz_direction = "Right"
#     else:
#         horiz_direction = "Center"

#     # Determine vertical direction
#     if eye_center[1] < top_corner:
#         vert_direction = "Up"
#     elif eye_center[1] > bottom_corner:
#         vert_direction = "Down"
#     else:
#         vert_direction = "Center"

#     return horiz_direction, vert_direction

# def analyze_eyes_direction(shape):
#     # Indices for the left and right eyes
#     left_eye_indices = [36, 37, 38, 39, 40, 41]
#     right_eye_indices = [42, 43, 44, 45, 46, 47]



#     left_eye_direction, left_eye_vert_direction = get_eye_direction(shape, left_eye_indices)
#     right_eye_direction,right_eye_vert_direction = get_eye_direction(shape, right_eye_indices)

#     return left_eye_direction,left_eye_vert_direction, right_eye_direction,right_eye_vert_direction




import cv2
import numpy as np



def detect_gaze(eye_region):
    # Convert to grayscale
    # Check if eye_region is valid
    if eye_region is None or eye_region.size == 0:
        return "Invalid eye region"

    # Check if eye_region is already grayscale
    if len(eye_region.shape) == 2:
        gray_eye = eye_region
    else:
        # Convert to grayscale
        gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)

    blurred_eye = cv2.GaussianBlur(gray_eye, (7, 7), 0)
    # Use adaptive thresholding
    thresholded_eye = cv2.adaptiveThreshold(blurred_eye, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours
    contours, _ = cv2.findContours(thresholded_eye, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Assuming the largest contour is the pupil
        max_contour = max(contours, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(max_contour)

        # Pupil center
        pupil_center = (x + w / 2, y + h / 2)

        # Eye region center
        eye_center = (eye_region.shape[1] / 2, eye_region.shape[0] / 2)

        # Determine gaze direction
        if pupil_center[0] < eye_center[0]:
            horizontal = "Looking Left"
        elif pupil_center[0] > eye_center[0]:
            horizontal = "Looking Right"
        else:
            horizontal = "Looking Center"

        if pupil_center[1] < eye_center[1]:
            vertical = "Looking Up"
        elif pupil_center[1] > eye_center[1]:
            vertical = "Looking Down"
        else:
            vertical = "Looking Center"

        return horizontal, vertical

    return "Unable to determine gaze"
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#     for (x, y, w, h) in faces:
        # roi_gray = gray[y:y+h, x:x+w]
        # eyes = eye_cascade.detectMultiScale(roi_gray)

        # for (ex, ey, ew, eh) in eyes:
        #     eye_region = roi_gray[ey:ey+eh, ex:ex+ew]
            
        #     gaze_direction = detect_gaze(eye_region)
        #     print(gaze_direction,time.time())  # or use this information as needed

#     cv2.imshow('Frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
