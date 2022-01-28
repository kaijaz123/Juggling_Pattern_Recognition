import tkinter
import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_dashboard(frame):
    """
        make a dashboard to display the analysis of player performance
        - distance to camera of player
        - left&right elbow angle
        - ball number detected
        - ball id with ball height - distance between hand and ball
    """
    # dashboard settings
    image_h, image_w = frame.shape[:2]
    color = (255,255,255)
    x_diff = (0.65 * (image_h+image_w)//4)
    y_diff = (0.6 * (image_h+image_w)//40)
    pos_y = image_h//35

    # font settings
    font_size = (0.1 * (image_h+image_w)/400)
    scale = int(0.3 * (image_h+image_w)//600)
    font_type = cv2.FONT_HERSHEY_DUPLEX
    line_type = cv2.LINE_AA

    # create dashboard
    x_start = int(image_w-(x_diff+(x_diff*0.05)))
    y_start = int(y_diff-(y_diff*0.01))
    x_end = int(image_w-(x_diff*0.15))
    if image_h > image_w:
        y_end = int(image_h//3.75)
    else:
        y_end = int(image_h//2.3)
    cv2.rectangle(frame, (x_start,y_start), (x_end,y_end), (0,0,0), cv2.FILLED)
    cv2.rectangle(frame, (x_start,y_start), (x_end,y_end), (255,255,255), 1)

    # put in information
    cv2.putText(frame, str("Data Dashboard"), (int(x_start+(x_end-x_start)*0.15),int(pos_y+y_diff)), font_type, font_size+0.2,
                color, scale+1, line_type)
    cv2.putText(frame, str("Distance to camera: "), (int(image_w-x_diff),int(pos_y+y_diff*2)), font_type, font_size,
                color, scale, line_type)
    cv2.putText(frame, "Left Elbow Angle: ", (int(image_w-x_diff), int(pos_y+y_diff*3)), font_type, font_size,
                color, scale, line_type)
    cv2.putText(frame, "Right Elbow Angle: ", (int(image_w-x_diff), int(pos_y+y_diff*4)), font_type, font_size,
                color, scale, line_type)
    cv2.putText(frame, "Balls detected: ", (int(image_w-x_diff), int(pos_y+y_diff*5)), font_type, font_size,
                color, scale, line_type)
    cv2.putText(frame, "Height", (int(image_w-x_diff+(x_diff//1.7)),int(pos_y+y_diff*6)), font_type, font_size,
                color, scale, line_type)

    return frame

def analysis(centroids, pair_ball, frame):
    # initialize variable
    image_h, image_w = frame.shape[:2]
    color = (255,255,255)
    ball_count = len(centroids)

    # resolution settings for put text
    x_diff = (0.65 * (image_h+image_w)//4)
    y_diff = (0.6 * (image_h+image_w)//40)
    pos_x = (int(image_w-x_diff+(x_diff/1.7)))
    pos_y = (image_h//35)+y_diff*5

    # font settings
    font_size = (0.1 * (image_h+image_w)/400)
    scale = int(0.3 * (image_h+image_w)//600)
    font_type = cv2.FONT_HERSHEY_SIMPLEX
    line_type = cv2.LINE_AA

    for ball in pair_ball:
        if len(ball["trace"]) < 2 or ball["frequency"] > 1: continue

        # ball Id with height
        ball_pos = ball["centroid_point"]
        hand_pos = ball["hand_xy"]
        if len(hand_pos) == 0 : continue
        ball_ID = ball["ID"]
        ball_state = ball["state"]
        ball_distance = abs(int(ball_pos[1]) - int(hand_pos[0][1]))
        ball_height = ball_distance / image_h

        # put text with ball id and height
        cv2.putText(frame, str("Ball {}: ".format(ball_ID)), (int(image_w-x_diff), int(pos_y+y_diff*2)),
                    font_type, font_size, color, scale, line_type)
        cv2.putText(frame, str("%.2fm" % ball_height), (pos_x, int(pos_y+y_diff*2)),
                    font_type, font_size, color, scale, line_type)

        pos_y += y_diff

    # put text with ball count
    cv2.putText(frame, str(ball_count), (pos_x,int((image_h//35)+y_diff*5)), font_type, font_size,
                color, scale, line_type)

    return frame
