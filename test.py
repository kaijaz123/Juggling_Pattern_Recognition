import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from core.utils import euclidean_distance

def distance_measurement():
    # Webcam
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    # Hand Detector
    detector = HandDetector(detectionCon=0.8, maxHands=1)

    # Find Function
    # x is the raw distance y is the value in cm
    x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
    y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    coff = np.polyfit(x, y, 2)  # y = Ax^2 + Bx + C

    # Loop
    while True:
        success, img = cap.read()
        hands = detector.findHands(img, draw=False)

        if hands:
            lmList = hands[0]['lmList']
            x, y, w, h = hands[0]['bbox']
            x1, y1 = lmList[5]
            x2, y2 = lmList[17]

            distance = int(math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2))
            print(distance)
            A, B, C = coff
            distanceCM = A * distance ** 2 + B * distance + C

            # print(distanceCM, distance)





            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 3)
            cvzone.putTextRect(img, f'{int(distanceCM)} cm', (x+5, y-10))

        cv2.imshow("Image", img)
        cv2.waitKey(1)

def workshop():
    import cv2
    import cvzone
    from cvzone.ColorModule import ColorFinder
    import numpy as np

    # Initialize the Video
    cap = cv2.VideoCapture('Videos/vid (4).mp4')

    # Create the color Finder object
    myColorFinder = ColorFinder(False)
    hsvVals = {'hmin': 8, 'smin': 96, 'vmin': 115, 'hmax': 14, 'smax': 255, 'vmax': 255}

    # Variables
    posListX, posListY = [], []
    xList = [item for item in range(0, 1300)]
    prediction = False

    while True:
        # Grab the image

        success, img = cap.read()
        # img = cv2.imread("Ball.png")
        img = img[0:900, :]

        # Find the Color Ball
        imgColor, mask = myColorFinder.update(img, hsvVals)
        # Find location of the Ball
        imgContours, contours = cvzone.findContours(img, mask, minArea=500)

        if contours:
            posListX.append(contours[0]['center'][0])
            posListY.append(contours[0]['center'][1])

        if posListX:
            # Polynomial Regression y = Ax^2 + Bx + C
            # Find the Coefficients
            A, B, C = np.polyfit(posListX, posListY, 2)

            for i, (posX, posY) in enumerate(zip(posListX, posListY)):
                pos = (posX, posY)
                cv2.circle(imgContours, pos, 10, (0, 255, 0), cv2.FILLED)
                if i == 0:
                    cv2.line(imgContours, pos, pos, (0, 255, 0), 5)
                else:
                    cv2.line(imgContours, pos, (posListX[i - 1], posListY[i - 1]), (0, 255, 0), 5)

            for x in xList:
                y = int(A * x ** 2 + B * x + C)
                cv2.circle(imgContours, (x, y), 2, (255, 0, 255), cv2.FILLED)

            if len(posListX) < 10:
                # Prediction
                # X values 330 to 430  Y 590
                a = A
                b = B
                c = C - 590

                x = int((-b - math.sqrt(b ** 2 - (4 * a * c))) / (2 * a))
                prediction = 330 < x < 430

            if prediction:
                cvzone.putTextRect(imgContours, "Basket", (50, 150),
                                   scale=5, thickness=5, colorR=(0, 200, 0), offset=20)
            else:
                cvzone.putTextRect(imgContours, "No Basket", (50, 150),
                                   scale=5, thickness=5, colorR=(0, 0, 200), offset=20)

        # Display
        imgContours = cv2.resize(imgContours, (0, 0), None, 0.7, 0.7)
        # cv2.imshow("Image", img)
        cv2.imshow("ImageColor", imgContours)
        cv2.waitKey(100)

def polynomial(pairA, pairB):
    degree = 2
    poslistX = np.array(pairB["trace"])[:,0][-10:]
    poslistY = np.array(pairB["trace"])[:,1][-10:]
    pred_pos = []

    A,B,C = np.polyfit(poslistX, poslistY, degree)
    mean_trace = np.mean(np.array(pairB["trace"])[:,0][:-1])
    if pairB["axis"] == 'y':
        if pairB["trace"][-1][0] < mean_trace:
            posList = list(reversed([pos for pos in range(0,int(pairB["trace"][-1][0])-2)]))
        else:
            posList = [pos for pos in range(int(pairB["trace"][-1][0])+2,1280)]

        for pos in posList[:5]:
            poly_val = A*(pos**2)+B*pos+C
            if poly_val < 0: break
            pred_pos.append([pos,poly_val])
        print(pred_pos)
        if len(pred_pos) < 1: return False
        mean_pos = np.mean(pred_pos,axis=0)
        print(mean_pos)
        distance = euclidean_distance(pairA[0],pairA[1],mean_pos[0],mean_pos[1])
        print(distance)
        if distance <= 70:
            return True
        return False

def video(vid):
    plt.style.use("dark_background")
    cam = cv2.VideoCapture(vid)
    num = 0
    while True:
        _, frame = cam.read()
        # rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imwrite("frames/{}.jpg".format(num),frame)
        num+=1
        # plt.imshow(rgb)
        # plt.show()

def merge(vid1,vid2):
    cap1 = cv2.VideoCapture(vid1)
    cap2 = cv2.VideoCapture(vid2)
    video_name = "out.mov"
    width = 640
    height = 360
    fps = int(cap2.get(cv2.CAP_PROP_FPS))
    # width, height = int(cap1.get(3)), int(cap1.get(4))
    # print(fps)
    # print(width,height)
    output_format = "MJPG"
    codec = cv2.VideoWriter_fourcc(*output_format)
    video = cv2.VideoWriter(video_name, -1, fps, (width,height))
    while True:
        suc,frame = cap1.read()
        if suc:
            video.write(frame)
        else:
            break

    while True:
        suc,frame = cap2.read()
        if suc:
            video.write(frame)
        else:
            break

def dis_video(video):
    plt.style.use("dark_background")
    cap = cv2.VideoCapture(video)
    while True:
        _,frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.imshow(frame)
        plt.show()


if __name__ == '__main__':
    dis_video("output.avi")
    # merge("src/test2.mp4","src/31.mp4")
    # video('output.avi')
    # polynomial([757.0, 211.5, (731, 186), (783, 237), 2, 'a'],{'ID': '567', 'centroid_point': [756.0, 171.0], 'p1': (730, 146),
    # 'p2': (782, 196), 'state': 'unbound', 'trace': [[730.0, 309.5], [733.5, 256.5], [735.0, 211.5], [741.0, 170.5], [744.0, 135.5],
    # [745.5, 107.0], [747.5, 86.0], [750.0, 70.0], [751.5, 58.0], [753.5, 52.5], [755.0, 52.5], [755.5, 58.5], [755.5, 70.0],
    # [756.0, 86.5], [757.0, 108.0], [758.0, 135.0], [756.0, 171.0]], 'frequency': 0, 'distance_level': 2, 'colors': (131, 242, 56),
    # 'hand_xy': [[803, 452]], 'hand_seq': ['left'], 'axis': 'y'})
























#
