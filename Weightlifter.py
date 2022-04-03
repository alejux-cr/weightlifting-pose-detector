import cv2
import numpy as np
import time

from Pose import PoseDetector

cap = cv2.VideoCapture("vids/clean_20220226_120439.mp4")
width = 768
height = 1024

# cap = cv2.VideoCapture(0)
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

detector = PoseDetector()
while True:
  success, img = cap.read()
  if img is not None:
    img = cv2.resize(img, (width, height))
    img = detector.detect_pose(img)
    cv2.imshow("Weightlifting", img)
    key = cv2.waitKey(1)
    if  key == ord('q'):
      cap.release()
      break

  else:
    break

cap.release()
cv2.destroyAllWindows()
