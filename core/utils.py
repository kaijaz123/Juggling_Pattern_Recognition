import cv2
import math
import numpy as np
import copy
from core.simulation import display_demo_ball
from core.simulation import display_demo_pattern
from core.simulation import display_demo_palm
from core.analysis import analysis
from core.pattern import pattern_recognition

def euclidean_distance(x1,y1,x2,y2):
    # euclidean distance formula
    x = x2 - x1
    y = y2 - y1
    distance = int(math.sqrt((x**2) + (y**2)))
    return distance

def update_record(ptns):
    record_window = 4 # max record can be displayed
    if len(ptns) > record_window:
        # update pattern list to display latest result
        ptns = ptns[1:]
    return ptns

def mapping(pair_ball, pair_palm, centroid_trace=False):
    threshold = 55 # distance threshold
    bound_ball = []
    hand_dis = []

    # map hands and balls
    for ball in pair_ball:
        for palm in pair_palm:
            if centroid_trace:
                distance = euclidean_distance(ball[0],ball[1],palm[0],palm[1])
            else:
                distance = euclidean_distance(ball["centroid_point"][0], ball["centroid_point"][1], palm[0], palm[1])
            hand_dis.append([distance,[palm[0],palm[1]],palm[-1]])

        # make sure there are hands detected
        if len(hand_dis) > 0:
            shortest = min(hand_dis)

            # mapping for centroid pair ball
            if centroid_trace:
                if shortest[0] <= threshold:
                    ball.append(True)
                    ball.append(shortest[-1])
                    bound_ball.append(ball)
                    continue
                bound_ball.append(ball)
                continue

            # mapping for existing pair ball
            if shortest[0] <= threshold:
                ball["state"] = "bound"
                ball["hand_xy"].append(shortest[1])
                ball['hand_seq'].append(shortest[-1])
                bound_ball.append(ball)
            else:
                if len(ball["trace"]) == 1 or len(ball["hand_xy"]) == 0:
                    ball["hand_xy"] = [shortest[1]]
                    ball['hand_seq'] = [shortest[-1]]
                ball["state"] = "unbound"

        # clear hand distance for next ball
        hand_dis = []
    return bound_ball


def classification(image, bound_ball_pair, prev_pair, pair_ball, ptn_model):
    image_h, image_w = image.shape[:2]
    results = []
    # recognition
    if len(bound_ball_pair) > 0:
        for bound_ball in bound_ball_pair:
            if bound_ball in pair_ball:
                pair_ball.remove(bound_ball)
            # recognize pattern
            results.append(pattern_recognition(bound_ball, image_w, image_h, ptn_model))

    # update previous pair
    prev_pair = copy.deepcopy(pair_ball)

    return results


def display_demo(demo, results, ptns, bound_ball_pair, pair_ball, pair_palm):
    # update record - prevent overloaded information
    ptns = update_record(ptns)

    # display pattern on demo
    if len(bound_ball_pair) < 1:
        demo = display_demo_pattern(demo,ptns)
    else:
        for result,bound_ball in zip(results,bound_ball_pair):
            if result is None:
                demo = display_demo_pattern(demo,ptns)
                continue
            elif len(result) > 0:
                result = result[0]
            ptns.append([result,bound_ball["colors"]])
            demo = display_demo_pattern(demo, ptns)

    # display ball and palm on demo
    for ball in bound_ball_pair:
        demo = display_demo_ball(demo, ball)

    for ball in pair_ball:
        if ball["frequency"] >= 2: continue
        demo = display_demo_ball(demo, ball)

    for palm in pair_palm:
        demo = display_demo_palm(demo, palm)

    return demo, ptns


def draw_bbox(image, bound_ball_pair, pair_ball, pair_palm):
    image_h, image_w = image.shape[:2]

    # draw bbox on ball
    for ball in pair_ball:
        if ball["frequency"] >= 2: continue
        cv2.rectangle(image, ball["p1"], ball["p2"], (255,0,0), 3)
        text = "ball " + str(ball["ID"]) + " " + str(ball["state"])
        cv2.putText(image, text, (int(ball["p1"][0]), int(ball["p1"][1])+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), int(0.6 * (image_h + image_w) / 600))

    # draw bbox on bound ball
    for bound_ball in bound_ball_pair:
        cv2.rectangle(image, bound_ball["p1"], bound_ball["p2"], (0,255,0), 3)
        text = "ball bound"
        cv2.putText(image, text, (int(bound_ball["p1"][0]), int(bound_ball["p1"][1])+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0),
                    int(0.6 * (image_h + image_w) / 600))

    # draw bbox on ball
    for palm in pair_palm:
        cv2.rectangle(image, (palm[2][0]-25, palm[2][1]-25), (palm[3][0]+25, palm[3][1]+25), (0,0,255), 3)
        cv2.putText(image, "palm", (int(palm[2][0]), int(palm[2][1])+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), int(0.6 * (image_h + image_w) / 600))

    return image
