import cv2
import mediapipe as mp
import numpy as np
from collections import namedtuple
#region
PERI_ORAL = [2, 326, 327, 426, 436, 432, 422, 424, 418, 421, 200, 201, 194, 204, 202, 212, 216, 206, 98, 97]
PERI_ORAL_RGB = (0,100,0)
PERI_ORAL = np.asarray(PERI_ORAL)

# namedtuple for landmark and color of landmark
region_rgb = namedtuple('region_rgb', 'landmarks color')

peri_oral_c = region_rgb(PERI_ORAL, PERI_ORAL_RGB)




#provide image, y and x, color
def _draw_rectangle(frame, yx, r, g, b):
    for x in range(-2, 2):
        for y in range(-2, 2):
            if yx[0] + y < frame.shape[0] and yx[1] + x < frame.shape[1]:
                frame[yx[0] + y, yx[1] + x] = [r, g, b]
    return frame

#like the draw rectangle function, but for lists of landmarks
def _draw_multiple_rectangles(frame, landmarks, region, r, g, b):
    new_frame = frame
    for landmark in region:
        point = _return_landmark_coordinate(frame, landmarks, landmark)
        new_frame = _draw_rectangle(frame, point, r, g, b)
    return new_frame

#input all facemesh landmarks, the image dimensions, and the specific landmark index
def _return_landmark_coordinate(frame, landmarks, landmark_index):
    img_dim = frame.shape
    # Get centroid landmark
    position = landmarks.landmark[landmark_index]

    # Get pixel coordinate value of centroid
    cy = int(np.round(img_dim[0] * position.y))
    cx = int(np.round(img_dim[1] * position.x))
    return cy, cx


LANDMARK_CENTER_OF_FACE_INDEX = 1
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened():
        success, image = cap.read()

        # get x,y dimensions of image
        color_image_dim = image.shape

        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            break

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = face_mesh.process(image)

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACE_CONNECTIONS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)

                # landmark_coordinates =_return_landmark_coordinate(face_landmarks,(480, 640, 3), 4)
                image = _draw_multiple_rectangles(image, face_landmarks, peri_oral_c.region, peri_oral_c.color)

                # Draw rectangles at centroid
                # image = _draw_rectangle(image, cx, cy, 0, 0, 255)
                cv2.imshow('MediaPipe FaceMesh', image)
            while cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()