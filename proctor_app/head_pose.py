import numpy as np
import cv2
def estimate_head_pose(shape, frame):
    # Define the model points (the 3D points of a generic model face)
    # You may need to adjust these points to be more accurate
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left Mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ])

    # 2D image points from the facial landmarks
    image_points = np.array([
        shape[30],     # Nose tip
        shape[8],      # Chin
        shape[36],     # Left eye left corner
        shape[45],     # Right eye right corner
        shape[48],     # Left Mouth corner
        shape[54]      # Right mouth corner
    ], dtype="double")

    # Camera internals
    size = frame.shape
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype = "double"
    )

    # Assuming no lens distortion
    dist_coeffs = np.zeros((4,1)) 

    # Solve for pose
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    return rotation_vector, translation_vector

def get_head_pose_angles(rotation_vector, translation_vector):
    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    projection_matrix = np.hstack((rotation_matrix, translation_vector.reshape(3, 1)))
    _, _, _, _, _, _, angles = cv2.decomposeProjectionMatrix(projection_matrix)
    return angles.flatten()[:3]  # Returns pitch, yaw, and roll


def point_to_np(point):
    return np.array([point.x, point.y])

def get_face_direction(landmarks):
    # Horizontal direction logic
    nose = point_to_np(landmarks.part(30))
    left_face = point_to_np(landmarks.part(2))   # Point on the left side of the face
    right_face = point_to_np(landmarks.part(14)) # Point on the right side of the face

    nose_to_left_face = np.linalg.norm(nose - left_face)
    nose_to_right_face = np.linalg.norm(nose - right_face)

    if abs(nose_to_left_face - nose_to_right_face) < 25:  # Threshold might need adjustment
        direction_horizontal = "Center"
    elif nose_to_left_face > nose_to_right_face:
        direction_horizontal = "Turned Right"
    else:
        direction_horizontal = "Turned Left"
    
    # Vertical direction logic
    chin = point_to_np(landmarks.part(8))
    left_eye = point_to_np(landmarks.part(36))
    right_eye = point_to_np(landmarks.part(45))
    eye_midpoint = (left_eye + right_eye) / 2

    nose_to_chin = np.linalg.norm(chin - nose)
    nose_to_eye_midpoint = np.linalg.norm(eye_midpoint - nose)

    # Adjust these ratios and thresholds based on your observation and testing
    vertical_ratio = nose_to_eye_midpoint / nose_to_chin

    if vertical_ratio > 0.5:  # Example threshold, adjust based on testing
        direction_vertical = "Looking Down"
    elif vertical_ratio < 0.20:  # Example threshold, adjust based on testing
        direction_vertical = "Looking Up"
    else:
        direction_vertical = "Center"

    return f"{direction_horizontal}, {direction_vertical}"

