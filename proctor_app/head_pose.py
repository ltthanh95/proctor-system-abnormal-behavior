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

# def get_face_direction(landmarks):
#     # Horizontal direction logic
#     nose = point_to_np(landmarks.part(30))
#     left_face = point_to_np(landmarks.part(2))   # Point on the left side of the face
#     right_face = point_to_np(landmarks.part(14)) # Point on the right side of the face

#     nose_to_left_face = np.linalg.norm(nose - left_face)
#     nose_to_right_face = np.linalg.norm(nose - right_face)

#     if abs(nose_to_left_face - nose_to_right_face) < 20:  # Threshold might need adjustment
#         direction_horizontal = "Center"
#     elif nose_to_left_face > nose_to_right_face:
#         direction_horizontal = "Turned Right"
#     else:
#         direction_horizontal = "Turned Left"
    
#     # Vertical direction logic
#     chin = point_to_np(landmarks.part(8))
#     left_eye = point_to_np(landmarks.part(36))
#     right_eye = point_to_np(landmarks.part(45))
#     eye_midpoint = (left_eye + right_eye) / 2

#     nose_to_chin = np.linalg.norm(chin - nose)
#     nose_to_eye_midpoint = np.linalg.norm(eye_midpoint - nose)

#     # Adjust these ratios and thresholds based on your observation and testing
#     vertical_ratio = nose_to_eye_midpoint / nose_to_chin

#     if vertical_ratio > 0.3:  # Example threshold, adjust based on testing
#         direction_vertical = "Looking Down"
#     elif vertical_ratio < 0.15:  # Example threshold, adjust based on testing
#         direction_vertical = "Looking Up"
#     else:
#         direction_vertical = "Center"

#     return f"{direction_horizontal}, {direction_vertical}"

def get_eye_direction(eye_points, gray_frame):
    # Create a mask for the eye and find the average intensity of the regions
    min_x = min(eye_points[:, 0])
    max_x = max(eye_points[:, 0])
    min_y = min(eye_points[:, 1])
    max_y = max(eye_points[:, 1])
    eye_frame = gray_frame[min_y:max_y, min_x:max_x]
    _, eye_frame = cv2.threshold(eye_frame, 42, 255, cv2.THRESH_BINARY)

    # Split the eye image into two halves
    w = eye_frame.shape[1]
    left_side = eye_frame[:, :w // 2]
    right_side = eye_frame[:, w // 2:]

    # Calculate the average intensity of each side
    left_side_white = cv2.countNonZero(left_side)
    right_side_white = cv2.countNonZero(right_side)

    # Determine the gaze direction based on the intensity
    if left_side_white > right_side_white:
        return "Looking Right"
    elif right_side_white > left_side_white:
        return "Looking Left"
    else:
        return "Center"

def get_face_direction(landmarks, gray_frame):
    # Extract eye points from landmarks
    left_eye_points = np.array([point_to_np(landmarks.part(i)) for i in range(36, 42)])
    right_eye_points = np.array([point_to_np(landmarks.part(i)) for i in range(42, 48)])

    # Determine the gaze direction for each eye
    left_eye_direction = get_eye_direction(left_eye_points, gray_frame)
    right_eye_direction = get_eye_direction(right_eye_points, gray_frame)

    # Combine the results
    if left_eye_direction == right_eye_direction:
        return left_eye_direction
    else:
        return "Center, Center"  # or some other logic to handle conflicting directions

# ... [rest of your script]
