import cv2
import mediapipe as mp
import numpy as np
camera = cv2.VideoCapture(0)
ai = mp.solutions.selfie_segmentation.SelfieSegmentation()
copies = []
while True:
    success, frame = camera.read()
    if not success:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mask = ai.process(rgb).segmentation_mask > 0.5
    photo = np.zeros_like(frame)
    for copy, old_mask in copies:
        photo[old_mask] = copy[old_mask]
    photo[mask] = frame[mask]
    cv2.imshow("Clone Camera", photo)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("f"):
        copies.append((frame.copy(), mask.copy()))
    if key == ord("s"):
        cv2.imwrite("clones.png", photo)
    if key == ord("c"):
        copies.clear()
    if key == ord("q"):
        break
camera.release()
cv2.destroyAllWindows()