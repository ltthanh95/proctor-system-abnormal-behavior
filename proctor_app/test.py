import cv2
import dlib
import numpy as np

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./proctor_app/files/shape_predictor_68_face_landmarks.dat")

def point_to_np(point):
    return np.array([point.x, point.y])

def get_face_direction(landmarks):
    nose = point_to_np(landmarks.part(30))
    left_face = point_to_np(landmarks.part(2))  # Point on the left side of the face
    right_face = point_to_np(landmarks.part(14))  # Point on the right side of the face

    nose_to_left_face = np.linalg.norm(nose - left_face)
    nose_to_right_face = np.linalg.norm(nose - right_face)

    if abs(nose_to_left_face - nose_to_right_face) < 20:  # Threshold might need adjustment
        return "Facing Forward"
    elif nose_to_left_face > nose_to_right_face:
        return "Turned Right"
    else:
        return "Turned Left"

# Capture video from the first camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    print(faces)

    for face in faces:
        landmarks = predictor(gray, face)
        direction = get_face_direction(landmarks)
        cv2.putText(frame, direction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
