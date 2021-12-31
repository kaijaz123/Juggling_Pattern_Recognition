import numpy as np
import cv2
from core.utils import euclidean_distance

def mapping(pair_ball, pair_palm):
    bound_ball = []
    for ball in pair_ball:
        for palm in pair_palm:
            distance = euclidean_distance(ball[0],ball[1],palm[0],palm[1])
            if distance <= 40 and ball not in bound_ball:
                bound_ball.append(ball)

    return bound_ball

def bound(centroids, pair_palm):
    bound_ball = mapping(centroids, pair_palm)

    if len(self.bound_pair_ball) <1:
        for ball in bound_ball:
            pair_ball = self.pair_ball_initialization(ball)
            self.bound_pair_ball.append(pair_ball)
        return

    pair_distances = []
    for ball in bound_ball:
        for pair_ball in self.bound_pair_ball:
            distance = euclidean_distance(pair_ball["centroid_point"][0],pair_ball["centroid_point"][1],ball[0],ball[1])
            pair_distances.append([distance,pair_ball["ID"],ball[0],ball[1],ball[2],ball[3],ball[-1],ball])

    min_distance = min(pair_distances)
    # if min_distance <= 5:
