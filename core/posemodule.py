import mediapipe as mp
import math
import cv2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

class poseDetector():

    def __init__(self, mode=False, upBody=False, smooth=True,
                 detectionCon=0.5, trackCon=0.5):

        # initialize pose detector settings
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode = self.mode,
                                     smooth_landmarks = self.smooth,
                                     min_detection_confidence = self.detectionCon,
                                     min_tracking_confidence = self.trackCon)
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # disance measurement for polynomial fit
        self.x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
        self.y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
        self.coefficient = np.polyfit(self.x,self.y,2)


    def findPose(self, img, demo, draw=True):
        sucess = False
        self.results = self.pose.process(img)
        if self.results.pose_landmarks is None:
            return sucess, demo

        sucess = True
        self.landmark = self.results.pose_landmarks.landmark
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(demo, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS,
                                           landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())

        return sucess, demo

    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img.astype(np.uint8), (int(cx), int(cy)), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList

    def calculate_angle(self, pointA, pointB, pointC):
        Ax, Ay = pointA[1]-pointB[1], pointA[2]-pointB[2]
        Cx, Cy = pointC[1]-pointB[1], pointC[2]-pointB[2]
        rad = math.atan2(Ay,Ax) - math.atan2(Cy,Cx)
        degree = rad * 180 / math.pi
        return 450 - degree if degree > 90 else 90 - degree

    def find_Elbow_angle(self, image, demo):
        if len(self.lmList) < 1: return image

        # resolution settings for put text
        image_h, image_w = image.shape[:2]
        x_diff = (0.65 * (image_h+image_w)//4)
        y_diff = (0.6 * (image_h+image_w)//40)
        pos_x = (int(image_w-x_diff+(x_diff/1.7)))
        pos_y = (image_h//35)+y_diff*3

        # font settings
        font_size = (0.1 * (image_h+image_w)/400)
        scale = int(0.3 * (image_h+image_w)//600)

        # calculate for left and right elblow angle
        left_shoulder, left_elbow, left_wrist = self.lmList[11], self.lmList[13], self.lmList[15]
        right_shoulder, right_elbow, right_wrist = self.lmList[12], self.lmList[14], self.lmList[16]
        left_elbow_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)

        # display on image
        image_h, image_w = image.shape[:2]
        cv2.putText(image, "{}".format(str("{:.2f}".format(left_elbow_angle))), (int(pos_x), int(pos_y)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, (255,255,255), scale, cv2.LINE_AA)
        cv2.putText(image, "{}".format(str("{:.2f}".format(right_elbow_angle))), (int(pos_x), int(pos_y+y_diff)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, (255,255,255), scale, cv2.LINE_AA)

        cv2.circle(demo, (right_elbow[1], right_elbow[2]), 15, (0, 0, 255), cv2.FILLED)
        cv2.circle(demo, (left_elbow[1], left_elbow[2]), 15, (0, 0, 255), cv2.FILLED)
        return image, demo

    def findPalm(self):
        if len(self.lmList) <1: return [],[]

        palm_sequence = [self.lmList[16][1:], self.lmList[18][1:], self.lmList[20][1:], self.lmList[22][1:]]
        min_x = min(palm_sequence, key = lambda x:x[0])[0]
        min_y = min(palm_sequence, key = lambda x:x[1])[1]
        max_x = max(palm_sequence, key = lambda x:x[0])[0]
        max_y = max(palm_sequence, key = lambda x:x[1])[1]
        cw = min_x + ((max_x - min_x) // 2)
        ch = min_y + ((max_y - min_y) // 2)
        # right_palm = [center_w, center_y, [min_x,min_y], [max_x,max_y], "right"]
        right_palm = [cw, ch, [min_x,min_y], [max_x,max_y], "right"]

        palm_sequence = [self.lmList[15][1:], self.lmList[17][1:], self.lmList[19][1:], self.lmList[21][1:]]
        min_x = min(palm_sequence, key = lambda x:x[0])[0]
        min_y = min(palm_sequence, key = lambda x:x[1])[1]
        max_x = max(palm_sequence, key = lambda x:x[0])[0]
        max_y = max(palm_sequence, key = lambda x:x[1])[1]
        cw = min_x + ((max_x - min_x) // 2)
        ch = min_y + ((max_y - min_y) // 2)
        # left_palm = [center_w, center_y, [min_x,min_y], [max_x,max_y], "left"]
        left_palm = [cw, ch, [min_x,min_y], [max_x,max_y], "left"]

        return right_palm, left_palm

    def distance_estimation(self, frame):
        if len(self.lmList) <1: return frame

        # resolution settings for put text
        image_h, image_w = frame.shape[:2]
        x_diff = (0.65 * (image_h+image_w)//4)
        y_diff = (0.6 * (image_h+image_w)//40)
        pos_x = (int(image_w-x_diff+(x_diff/1.7)))
        pos_y = (image_h//35)+y_diff*2

        # font settings
        font_size = (0.1 * (image_h+image_w)/400)
        scale = int(0.3 * (image_h+image_w)//600)

        # use left hand as a reference for measurement
        left_pinky = self.lmList[12]
        left_index = self.lmList[11]
        distance = int(math.sqrt((left_pinky[0]-left_index[0])**2 + (left_pinky[1]-left_index[1])**2))

        # scale into range from 0 - 260
        old_range = 1000
        new_range = 260
        scaled_distance = ((distance-0)/1000) * new_range + 0

        # get the coeff from polynomial fit - Ax^2 + Bx + C
        A,B,C = self.coefficient
        estimated_distance_cm = A * scaled_distance ** 2 + B * scaled_distance + C
        distance_m = estimated_distance_cm / 100

        # display the distance
        h,w = frame.shape[:2]
        cv2.putText(frame, "{}m".format(str("{:.2f}".format(distance_m))), (int(pos_x), int(pos_y)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, (255,255,255), scale, cv2.LINE_AA)

        return frame

if __name__ == '__main__':
    pose_detector = poseDetector()
    cap = cv2.VideoCapture("src/video0.mov")
    while True:
        success, frame = cap.read()
        frame = pose_detector.findPose(frame, frame)
        lmList = pose_detector.findPosition(frame, draw=False)
        frame = pose_detector.find_Elbow_angle(frame)
        frame = pose_detector.distance_estimation(frame)
        cv2.imshow("frame", frame)
        cv2.waitKey(1)
