import cv2
import numpy as np
import time

from Pose import PoseDetector

# cap = cv2.VideoCapture("vids/clean_20220226_120439.mp4")
# width = 768
# height = 1024

cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

detector = PoseDetector()
squat_count = 0
direction = 0

while True:
  success, img = cap.read()
  if img is not None:
    img = cv2.resize(img, (width, height))
    img = detector.detect_pose(img, False)
    landmarks_list = detector.find_position(img, False)

    if len(landmarks_list) > 0:
      hip_angle = detector.find_angle(img, 12, 24, 26)
      hip_percentage = np.interp(hip_angle, (60, 190), (0, 100))

      knee_angle = detector.find_angle(img, 24, 26, 28, True, True)
      knee_percentage = np.interp(knee_angle, (60, 180), (0, 100))

      ankle_angle = detector.find_angle(img, 26, 28, 32)
      ankle_percentage = np.interp(hip_angle, (210, 310), (0, 100))
     
      print("HIP % " + str(hip_percentage))
      print("KNEE % " + str(knee_percentage))

      color = (255, 0, 255)
      # Check for squats
      if  80 < hip_percentage < 100 and 80 < knee_percentage < 100:
        color = (0, 255, 0)
        if direction == 0:
          squat_count += 0.5
          direction = 1

      if 0 < hip_percentage < 20  and 0 < knee_percentage < 20:
        color = (0, 255, 0)
        if direction == 1:
          squat_count += 0.5
          direction = 0

      cv2.rectangle(img, (0, 375), (100, 470), (0, 255, 0), cv2.FILLED)
      cv2.putText(img, str(int(squat_count)), (25, 455), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 10)

    cv2.imshow("Weightlifting", img)
    key = cv2.waitKey(1)
    if  key == ord('q'):
      cap.release()
      break

  else:
    break

cap.release()
cv2.destroyAllWindows()
