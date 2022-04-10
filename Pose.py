import math
import cv2
import mediapipe as mp
import numpy as np

class PoseDetector():

  def __init__(self, static_img_mode=False, complexity=2, smooth_landmarks=True, enable_segmentation=True,
                smooth_segmentation=True, min_detection_conf=0.6, min_tracking_conf=0.9):
    
    self.BG_COLOR = (192, 192, 192) # gray
    self.mp_draw = mp.solutions.drawing_utils
    self.mp_pose = mp.solutions.pose
    self.mp_drawing_styles = mp.solutions.drawing_styles
    self.pose = self.mp_pose.Pose(static_image_mode=static_img_mode, model_complexity=complexity, smooth_landmarks=smooth_landmarks,
                                  enable_segmentation=enable_segmentation, smooth_segmentation=smooth_segmentation,
                                  min_detection_confidence=min_detection_conf, min_tracking_confidence=min_tracking_conf)
    self.landmarks_list = []

  def detect_pose(self, img, draw=True):
    img_to_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    self.results = self.pose.process(img_to_RGB)
    if self.results.pose_landmarks:
        if draw:
            condition = np.stack((self.results.segmentation_mask) * 3, axis=-1) > 0.1
            bg_image = np.zeros(img_to_RGB.shape, dtype=np.uint8)
            bg_image[:] = self.BG_COLOR
            img = np.where(condition, img, bg_image)
            self.mp_draw.draw_landmarks(img, self.results.pose_landmarks,
                                        self.mp_pose.POSE_CONNECTIONS,
                                        landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
    return img

  def find_position(self, img, draw=True):
    self.landmarks_list = []
    if self.results.pose_landmarks:
        for id, lm in enumerate(self.results.pose_landmarks.landmark):
          h, w, c = img.shape
          cx, cy = int(lm.x * w), int(lm.y * h)
          self.landmarks_list.append([id, cx, cy])
          if draw:
            cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
    return self.landmarks_list

  def find_angle(self, img, p1, p2, p3, draw=True, inverse=False):

    # Get the landmarks
    x1, y1 = self.landmarks_list[p1][1:]
    x2, y2 = self.landmarks_list[p2][1:]
    x3, y3 = self.landmarks_list[p3][1:]

    # Calculate the Angle
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    if angle < 0:
      angle += 360
    else:
      if angle > 180 and inverse:
        angle = 360 - angle

    if draw:
      cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
      cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
      cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
      cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
      cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
      cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
      cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
      cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
      cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    return angle
