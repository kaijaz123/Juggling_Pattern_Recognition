import tkinter
import cv2
import numpy as np
import matplotlib.pyplot as plt

def analysis(pair_ball, screen):
    image_h, image_w = screen.shape[:2]

    radius = 20
    true_detection = 0
    for ball in pair_ball:
        if ball["frequency"] > 1: continue
        true_detection += 1
        ball_pos = ball["centroid_point"]
        hand_pos = ball["hand_xy"]
        if ball["state"] == 'unbound':
            cv2.circle(screen, (int(ball_pos[0]), int(ball_pos[1])), radius, (0,0,255), thickness = cv2.FILLED)
        else:
            cv2.circle(screen, (int(ball_pos[0]), int(ball_pos[1])), radius, (0,255,0), thickness = cv2.FILLED)

        if len(hand_pos) == 0 : continue
        distance = abs(int(ball_pos[1]) - int(hand_pos[0][1]))
        ball_height = distance / image_h
        cv2.putText(screen, str("%.2fm" % ball_height), (int(ball_pos[0]), int(ball_pos[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

    cv2.putText(screen, str("{} balls detected".format(true_detection)), (image_w - 250, 50), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0,0,255), 2, cv2.LINE_AA)

    return screen
