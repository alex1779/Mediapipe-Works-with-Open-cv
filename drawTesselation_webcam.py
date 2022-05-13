# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 11:51:12 2022

@author: Alejandro Maggioni
"""

import cv2
import mediapipe as mp
import imutils
from drawing import drawTesselation

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


def main():
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    cap = cv2.VideoCapture(0)
    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                break

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, _ = image.shape
            results = face_mesh.process(image)

            # Draw the face mesh annotations on the image.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:

                    drawTesselation(
                        image=image,
                        face_landmarks=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        color=(0, 0, 255),
                        thickness=1
                        )

            image = imutils.resize(image, height=720)
            cv2.imshow('MediaPipe Face Mesh Tesselation', image)
            if cv2.waitKey(5) & 0xFF == 27:
                cv2.destroyAllWindows()
                cap.release()
                break
    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    main()
