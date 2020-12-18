import cv2
import mediapipe as mp
import numpy as np


mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

cam = cv2.VideoCapture(0)
file = open('landmarks.txt', 'w')
while True:
    ret, frame = cam.read()

    image_rows, image_cols = frame.shape[:2]
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        continue
    annotated_image = frame.copy()
    face_landmarks = results.multi_face_landmarks[0]
    idx_to_coordinates = {}
    for idx, landmark in enumerate(face_landmarks.landmark):
        if landmark.visibility < 0 or landmark.presence < 0:
            continue
        landmark_px = mp_drawing._normalized_to_pixel_coordinates(landmark.x, landmark.y, image_cols, image_rows)
        if landmark_px:
            idx_to_coordinates[idx] = landmark_px
            cv2.circle(annotated_image, landmark_px, int((0.5 + landmark.z) * 5), (255, 0, 0), -1)
            file.write(f'{landmark_px[0]}\t{landmark_px[1]}\t{(0.5 + landmark.z) * 550}')
            file.write('\n')
    file.close()
    exit()

    cv2.imshow('Output', annotated_image)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
