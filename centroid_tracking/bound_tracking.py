import cv2
import numpy as np
import os
from core.utils import euclidean_distance
from collections import Counter

"""
    Unbound ball state - the ball is tossed into the air (siteswap notation 3 - C)
    Bound ball state - the ball is hold on hand (siteswap notation 1 - 2)

    The following tracking algorithm is customized for only bound ball tracking
    which mentioned above and applicable only to the case of siteswap notation 1 and 2
"""

def p1_track(pair_ball, ball_pair, bound_ball_pair, frame):
    distance_list = []
    predict_balls = []
    frequency_thres = 5
    avg_height = 0

    if len(ball_pair) > 0:
        for bp in ball_pair:
            avg_height += bp["trace"][-1][1]
        avg_height = avg_height // len(ball_pair)

        for pb in pair_ball:
            height = pb["trace"][-1][1]
            if abs(height - avg_height) < 30:
                bound_ball_pair.append(pb)

    for bp in ball_pair:
        for bbp in bound_ball_pair:
            distance = euclidean_distance(bbp["centroid_point"][0], bbp["centroid_point"][1],
                                          bp["trace"][-1][0], bp["trace"][-1][1])
            if distance > 70:continue
            distance_list.append([distance,bp["ID"],bbp["ID"],bbp["hand_seq"],bbp["p1"],bbp["p2"],bbp["centroid_point"]])

    for i in range(len(ball_pair)):
        if len(distance_list) < 1:continue
        shortest = min(distance_list)
        [distance_list.remove(item) for item in distance_list if item[2] == shortest[2] or item[1] == shortest[1]]
        for ball in ball_pair:
            if ball["ID"] == shortest[1]:
                ball["trace"].append(shortest[-1])
                ball["p1"] = shortest[4]
                ball["p2"] = shortest[5]
                if ball["hand_seq"] != shortest[3]:
                    ball["hand_seq"].append(shortest[3][-1])
                break
        [bound_ball_pair.remove(bbp) for bbp in bound_ball_pair[:] if bbp["ID"] == shortest[2]]

    for ball in ball_pair:
        ball["frequency"] += 1

    # make new pair in case any ball left over
    for bbp in bound_ball_pair:
        ball_pair.append(bbp)

    for bp in ball_pair[:]:
        if len(Counter(bp["hand_seq"])) > 1:
            predict_balls.append(bp)
            ball_pair.remove(bp)
            continue
        if bp["frequency"] >= frequency_thres:
            ball_pair.remove(bp)

    for ball in predict_balls:
        if len(ball["trace"]) > 3:
            ball["trace"] = ball["trace"][-3:]
        traces = np.array(ball["trace"])

        # padding to at least 3 traces
        if len(ball["trace"]) <= 3:
            pad_trace = ball["trace"][-1]
            for i in range(4-len(ball["trace"])):
                ball["trace"].append(pad_trace)

        for index,_ in enumerate(traces):
            if index == 0: continue
            thickness = index + 2
            cv2.line(frame, tuple(np.array(traces[index - 1]).astype(int)), tuple(np.array(traces[index]).astype(int)),
                     ball["colors"], thickness)

    return frame, predict_balls, ball_pair


def p2_track(location, ball_pair, bound_ball_pair, frame):
    predict_balls = []
    # trace_thres = 20
    trace_thres = 5

    for lct in location:
        for bbp in bound_ball_pair[:]:
            distance = abs(bbp["hand_xy"][0][0] - lct[0])
            if distance <= 100:
                bound_ball_pair.remove(bbp)

    # update bound ball pair
    for bp in ball_pair:
        bp["frequency"] += 1
        for bbp in bound_ball_pair[:]:
            distance = euclidean_distance(bbp["centroid_point"][0], bbp["centroid_point"][1],
                                          bp["trace"][-1][0], bp["trace"][-1][1])
            if distance <= 5:
                bp["trace"].append(bbp["centroid_point"])
                bp["p1"] = bp["p1"]
                bp["p2"] = bp["p2"]
                bound_ball_pair.remove(bbp)
                break

    for bp in ball_pair[:]:
        num_trace = len(bp["trace"])
        if num_trace == trace_thres:
            predict_balls.append(bp)
            ball_pair.remove(bp)
        if num_trace < 2:
            ball_pair.remove(bp)
        if bp['frequency'] > trace_thres:
            ball_pair.remove(bp)

    # append left over bound ball pair
    for bbp in bound_ball_pair:
        ball_pair.append(bbp)

    if len(predict_balls) > 0:
        location = []

    for ball in predict_balls:
        location.append(ball["trace"][-1])
        ball["colors"] = (67,211,255)
        cv2.circle(frame, np.array(ball["centroid_point"]).astype(int), 15, ball["colors"], thickness = cv2.FILLED)

    return frame, predict_balls, location, ball_pair
