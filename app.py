# Import necessary libraries
import secrets  # For generating secure tokens
from flask import Flask, render_template, redirect, request, Response, session  # Flask for web application
import pyrebase  # Pyrebase for Firebase integration
import requests  # HTTP requests
import json  # JSON handling
import cv2  # OpenCV for computer vision
import math  # Mathematical operations
import mediapipe as mp  # MediaPipe for pose estimation
import numpy as np  # NumPy for numerical operations
from playsound import playsound  # Playing sounds
import keyboard  # Simulating keyboard input
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
# Set a default weight (could be user-configurable)
weight = 60

# Initialize Flask
app = Flask(__name__)

# Define a function to calculate an angle given three points (shoulder, elbow, wrist)
def calculate_angle(a, b, c):
    """
    Calculate the angle between three points (shoulder, elbow, wrist).

    Args:
        a (list): Coordinates of the first point.
        b (list): Coordinates of the second point (midpoint).
        c (list): Coordinates of the third point.

    Returns:
        float: The calculated angle in degrees.
    """
    a = np.array(a)  # First point
    b = np.array(b)  # Midpoint
    c = np.array(c)  # End point

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Define a function to calculate the distance between two points
def calculate_distance(a, b):
    """
    Calculate the Euclidean distance between two points.

    Args:
        a (list): Coordinates of the first point.
        b (list): Coordinates of the second point.

    Returns:
        float: The calculated distance.
    """
    a = np.array(a)
    b = np.array(b)
    distance = math.hypot(b[0] - a[0], b[1] - a[1])
    return distance

# Define a function for detecting jumps
def jump_detector():
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)  # Set width
    cap.set(4, 700)   # Set height
    time.sleep(1)

    # Initialize variables for jump detection
    stage = None
    inputGoal = 5
    basepoints = 0
    basePointList = []
    hip_cord_l = 0
    hip_cord_r = 0
    shoulder_angle = 0
    shoulder_angle_r = 0
    counter = 0

    # Initialize MediaPipe Pose Estimation
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            # Recolor frame to RGB
            res = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res.flags.writeable = False
            # Make pose detection
            results = pose.process(res)
            res.flags.writeable = True
            res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                # Get coordinates of left shoulder, wrist, and hip
                shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                wrist_l = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                hip_cord_l = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
                # Calculate shoulder angle
                shoulder_angle = calculate_angle(hip_l, shoulder_l, wrist_l)

                # Get coordinates of right shoulder, wrist, and hip
                shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                wrist_r = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                hip_r = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                hip_cord_r = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
                # Calculate shoulder angle
                shoulder_angle_r = calculate_angle(hip_r, shoulder_r, wrist_r)

            except:
                pass

            # Draw UI elements on the frame
            cv2.rectangle(res, (0, 0), (1280, 60), (0, 0, 0), -1)
            cv2.putText(res, 'Calibrating Depth - JUMP 2 to 3 times and Raise your hands', (30, 40),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.rectangle(res, (730, 960 - 60), (1280, 960), (0, 0, 0), -1)
            cv2.putText(res, str(hip_cord_l), (750, 960 - 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 255, 255), 2,
                        cv2.LINE_AA)
            basePointList.append(hip_cord_l)

            # Render pose landmarks on the frame
            mp_drawing.draw_landmarks(res, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )

            # Encode the frame and yield it
            _, buffer = cv2.imencode(".jpg", res)
            res = buffer.tobytes()
            yield (b' --frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + res + b'\r\n')

            # Check for jump conditions and break
            if shoulder_angle > 90 and shoulder_angle_r > 90:
                basepoints = ((hip_cord_r + hip_cord_l) / 2)
                break
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    # Calculate the jump height and base point
    jumpPoint = min(basePointList)
    print("Jump height : ", jumpPoint)
    print("Base Point : ", basepoints)
    time.sleep(3)

    # Continue with jump detection
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            # Recolor frame to RGB
            res = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res.flags.writeable = False
            # Make pose detection
            results = pose.process(res)
            res.flags.writeable = True
            res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                # Get coordinates of left hip
                hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                hip_cord_l = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
                # Calculate shoulder angle
                shoulder_angle = calculate_angle(hip_l, shoulder_l, wrist_l)

                # Get coordinates of right hip
                hip_r = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                hip_cord_r = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
                # Calculate shoulder angle
                shoulder_angle_r = calculate_angle(hip_r, shoulder_r, wrist_r)

            except:
                pass

            # Detect the jump stage and trigger jump
            if hip_cord_l < jumpPoint:
                stage = "Jump"
            if hip_cord_l > jumpPoint and stage == 'Jump':
                stage = "Stand"
                keyboard.press_and_release('space')

            # Draw UI elements on the frame
            cv2.rectangle(res, (440, 0), (840, 60), (0, 0, 0), -1)
            cv2.putText(res, 'JUMP COUNTER', (460, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 255, 255), 1,
                        cv2.LINE_AA)

            cv2.line(res, (0, int(700 * jumpPoint)), (1280, int(700 * jumpPoint)), (0, 255, 0), 3)
            cv2.line(res, (0, int(700 * basepoints)), (1280, int(700 * basepoints)), (0, 0, 255), 3)

            cv2.rectangle(res, (0, 0), (100, 70), (0, 0, 0), -1)
            cv2.putText(res, str(counter), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.rectangle(res, (730, 960 - 60), (1280, 960), (0, 0, 0), -1)
            cv2.putText(res, str(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y), (750, 960 - 15),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 255, 255), 2, cv2.LINE_AA)

            # Render pose landmarks on the frame
            mp_drawing.draw_landmarks(res, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )

            # Encode the frame and yield it
            _, buffer = cv2.imencode(".jpg", res)
            res = buffer.tobytes()
            yield (b' --frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + res + b'\r\n')

            # Check for inputGoal and break
            if int(inputGoal) <= int(counter):
                break

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    # Close the video capture and destroy any open CV windows
    cap.release()
    cv2.destroyAllWindows()

# Define a route for the index page
@app.route("/")
def index():
    return render_template("home.html")

# Define a route for the jumping video stream
@app.route("/jumping")
def video():
    return Response(jump_detector(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Run the Flask app if this script is executed directly
if __name__ == "__main__":
    app.run(debug=True)