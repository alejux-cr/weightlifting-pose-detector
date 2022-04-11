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
position = None

while cap.isOpened():
  ret, frame = cap.read()
  if frame is not None:
    frame = cv2.resize(frame, (width, height))
    frame = detector.detect_pose(frame, False)
    landmarks_list = detector.find_position(frame, False)

    if len(landmarks_list) > 0:
      hip_angle = detector.find_angle(frame, 12, 24, 26)
      hip_percentage = np.interp(hip_angle, (60, 190), (0, 100))

      knee_angle = detector.find_angle(frame, 24, 26, 28, True, True)
      knee_percentage = np.interp(knee_angle, (60, 180), (0, 100))

      ankle_angle = detector.find_angle(frame, 26, 28, 32)
      ankle_percentage = np.interp(hip_angle, (210, 310), (0, 100))
     
      print("HIP % " + str(hip_percentage))
      print("KNEE % " + str(knee_percentage))

      color = (255, 0, 255)
      # Check for squats
      if  80 < hip_percentage < 100 and 80 < knee_percentage < 100:
        color = (0, 255, 0)
        if position == 'down':
          squat_count += 0.5
          position = 'up'

      if 0 < hip_percentage < 20  and 0 < knee_percentage < 20:
        color = (0, 255, 0)
        if position == 'up':
          squat_count += 0.5
          position = 'down'

      cv2.rectangle(frame, (0, 375), (100, 470), (0, 255, 0), cv2.FILLED)
      cv2.putText(frame, str(int(squat_count)), (25, 455), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 10)

    cv2.imshow("Weightlifting", frame)
    key = cv2.waitKey(1)
    if  key == ord('q'):
      cap.release()
      break

  else:
    break

cap.release()
cv2.destroyAllWindows()
