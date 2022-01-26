import cv2
import math
import copy
import random
import numpy as np
from centroid_tracking.config import cfg
from centroid_tracking.bound_tracking import p1_track, p2_track
from core.utils import euclidean_distance
import string

"""
    Unbound ball state - the ball is tossed into the air (siteswap notation 1 - 8 except 2)
    Bound ball state - the ball is hold on hand (siteswap 2)

    The following tracking algorithm is made up of Centroid Tracking algorithm
    which track by the geometric distance between objects. Besides, the algorithm
    is also customized by using collinearity to check the object moving path
    for better tracking performance.
"""

class Tracker:
    def __init__(self):
        # initialize class name
        self.class_file = cfg.YOLO.CLASSES
        self.classes = self.read_class_names(self.class_file)

        # initialize empty pair
        self.pair_bound_ball = []
        self.pair_ball = []
        self.prev_pair_ball = []

        # bound ball tracking
        self.location = []
        self.frame_sequence = 0
        self.bound_pair_ball = []
        self.p2_balls = []
        self.p1_balls = []

        # initialize threshold for collinearity, frequency, trace length
        self.collinearity_thres = 2
        self.frequency_thres = 4
        self.trace_thres = 5


    def read_class_names(self, class_file_name):
        # load in yolo object class
        names = {}
        with open(class_file_name, 'r') as data:
            for ID, name in enumerate(data):
                names[ID] = name.strip('\n')
        return names


    def frequency_checking(self):
        # check object frequency
        for ppb in self.prev_pair_ball:
            for pb in self.pair_ball:
                if ppb["centroid_point"] == pb["centroid_point"]:
                    pb["frequency"] += 1

        # remove pair ball with exceeded threhold
        for pb in self.pair_ball:
            if pb["frequency"] >= self.frequency_thres:
                self.pair_ball.remove(pb)

        # update prev pair
        self.prev_pair_ball = copy.deepcopy(self.pair_ball)


    def polynomial(self, pairA, pairB):
        if len(pairB["trace"]) < 2: return True

        degree = 2
        poslistX = np.array(pairB["trace"])[:,0][-10:]
        poslistY = np.array(pairB["trace"])[:,1][-10:]
        pred_pos = []

        A,B,C = np.polyfit(poslistX, poslistY, degree)
        if pairB["axis"] == 'y':
            mean_trace = np.mean(np.array(pairB["trace"])[:,0][:-1])
            if pairB["trace"][-1][0] < mean_trace:
                posList = list(reversed([pos for pos in range(0,int(pairB["trace"][-1][0])-1)]))
            else:
                posList = [pos for pos in range(int(pairB["trace"][-1][0])+1,1280)]

            for pos in posList[:5]:
                poly_val = A*(pos**2)+B*pos+C
                if poly_val < 0: break
                pred_pos.append([pos,poly_val])

            if len(pred_pos) < 1: return False
            mean_pos = np.mean(pred_pos,axis=0)
            distance = euclidean_distance(pairA[0],pairA[1],mean_pos[0],mean_pos[1])
            print(distance)
            if distance <= 70:
                return True
            return False

        elif pairB["axis"] == 'x':
            mean_trace = np.mean(np.array(pairB["trace"])[:,1][:-1])
            if pairB["trace"][-1][1] < mean_trace:
                posList = [pos for pos in range(0,int(pairB["trace"][-1][1])-1)]
            else:
                posList = [pos for pos in range(int(pairB["trace"][-1][1])+1,720)]

            for pos in posList[:5]:
                poly_val = A*(pos**2)+B*pos+C
                if poly_val < 0: break
                pred_pos.append([pos,poly_val])

            if len(pred_pos) < 0: return False
            mean_pos = np.mean(pred_pos,axis=0)
            distance = euclidean_distance(pairA[0],pairA[1],mean_pos[0],mean_pos[1])
            print(distance)
            if distance <= 70:
                return True
            return False


    def axis(self, ball_pair):
        if len(ball_pair["trace"]) < 2: return
        if ball_pair["axis"] is None:
            x = abs(ball_pair["trace"][-1][0] - ball_pair["trace"][-2][0])
            y = abs(ball_pair["trace"][-1][1] - ball_pair["trace"][-2][1])
            if x>y:
                ball_pair["axis"] = "x"
            else:
                ball_pair["axis"] = "y"


    def collinearity_checking(self, pairA, pairB):
        if len(pairB["trace"]) < 2: return True

        if pairB["axis"] is None:
            x = abs(pairB["trace"][-1][0] - pairB["trace"][-2][0])
            y = abs(pairB["trace"][-1][1] - pairB["trace"][-2][1])
            if x>y:
                pairB["axis"] = "x"
            else:
                pairB["axis"] = "y"

        if pairB["axis"] == "x":
            unit = (pairB["trace"][-1][0] + pairB["trace"][-2][0])//2
        else:
            unit = (pairB["trace"][-1][1] + pairB["trace"][-2][1])//2

        (x1,y1), (x2,y2) = pairB["trace"][-2:]
        (x3,y3) = pairA[0], pairA[1]
        area = abs(round(((x1*(y2-y3)) + (x2*(y3-y1)) + (x3*(y1-y2)))/unit,1))
        # print(area)
        if area <= self.collinearity_thres:
            return True
        return False


    def ball_pair_initialization(self, centroid):
        # initialize empty pair ball
        color = tuple(np.random.randint(256, size=3))
        color = (int(color[0]), int(color[1]), int(color[2]))

        obj_pair = dict(ID = str(int(random.randint(0,1000))), centroid_point = [centroid[0],centroid[1]],
                        p1 = centroid[2], p2 = centroid[3], state = [], trace = [[centroid[0],centroid[1]]],
                        frequency = 0, distance_level = centroid[4], colors = color, hand_xy = [], hand_seq = [],
                        axis = None)
        return obj_pair


    def unbound_ball_screening(self, ball_pair):
        # remove ball from unbound state for bound state tracking
        for ball in ball_pair[:]:
            if len(ball["trace"]) > 2:
                ball_pair.remove(ball)
        return ball_pair


    def track(self, image, bboxes):
        centroids = []
        num_classes = len(self.classes)
        image_h, image_w, _ = image.shape

        out_boxes, out_scores, out_classes, num_boxes = bboxes
        for i in range(num_boxes[0]):
            if int(out_classes[0][i]) < 0 or int(out_classes[0][i]) > num_classes: continue
            coor = out_boxes[0][i]
            coor[0] = int(coor[0] * image_h)
            coor[2] = int(coor[2] * image_h)
            coor[1] = int(coor[1] * image_w)
            coor[3] = int(coor[3] * image_w)
            c1, c2 = (int(coor[1]), int(coor[0])), (int(coor[3]), int(coor[2]))
            width = int(c2[0] - c1[0])
            height = int(c2[1] - c1[1])
            cw = c1[0] + (width / 2)
            ch = c1[1] + (height / 2)

            # check distance level with recpect to ball width ratio
            distance_level = 1 if width >= 57 else 2
            centroids.append([cw,ch,c1,c2,distance_level,string.ascii_lowercase[i]])

        return centroids
        # self.object_checking(centroids)


    def object_checking(self, centroids):
        """
            opt for different tracking progress
            1. no objects detected - check for frequency
            2. if existing pair ball is empty - initialization
            3. else goes into tracking process
        """
        num_centroid = len(centroids)
        num_pair = len(self.pair_ball)

        # if no objects detected
        if num_centroid < 1:
            # make sure there are existing pair
            if num_pair > 0:
                self.frequency_checking()

        # first frame tracking - initialization
        elif num_pair < 1:
            self.ff_object_tracking(centroids)

        # else unbound ball pair tracking
        else:
            self.unbound_tracking(centroids)


    def ff_object_tracking(self, centroids):
        # first frame tracking - initialize ball pair
        for centroid in centroids:
            pair_ball = self.ball_pair_initialization(centroid)
            self.pair_ball.append(pair_ball)
        self.prev_pair_ball = copy.deepcopy(self.pair_ball)


    def unbound_tracking(self, centroids):
        print("Centroids")
        print(centroids)
        print()
        print("Pair ball")
        print(self.pair_ball)
        print()
        match_distance = []
        for centroid in centroids:
            for ball in self.pair_ball:
                # check collinearity
                axis = 1
                collinear = self.collinearity_checking(centroid,ball)

                # centroid pair ball check
                if len(centroid) > 5 and len(ball["trace"]) < 2:
                    if ball["hand_seq"][0] == centroid[-1]:
                        continue

                # pair ball track
                if collinear:
                    axis_distance = 0
                    if ball["axis"] == "x":
                        axis_distance = abs(ball["trace"][-1][1] - centroid[1])
                    elif ball["axis"] == 'y':
                        axis_distance = abs(ball["trace"][-1][0] - centroid[0])

                    if axis_distance >= 20: axis_distance = 0
                    if ball["axis"] != None: axis = 0
                    geo_distance = euclidean_distance(ball["centroid_point"][0], ball["centroid_point"][1], centroid[0], centroid[1])
                    distance = geo_distance - axis_distance
                    match_distance.append([distance, ball["ID"], centroid[0], centroid[1], centroid[2], centroid[3], centroid[4],
                                           centroid[5], centroid, axis])

        print("Match distance")
        print(match_distance)
        print()
        # update pair that has shortest distance
        for index in range(len(self.pair_ball)):
            if len(match_distance) < 1: break

            # shortest_pair = min(match_distance, key = lambda x:x[0])
            shortest_pair = sorted(match_distance, key = lambda x:(x[-1],x[0]))[0]
            # get the matched pair ball ID and centroid ID
            centroid_ID = shortest_pair[-2]
            pair_ball_ID = shortest_pair[1]
            if shortest_pair[0] > 121:
                [match_distance.remove(pair) for pair in match_distance[:] if pair[-2] == centroid_ID]
                continue

            # remove shortest pair from match_distance and centroids
            [match_distance.remove(pair) for pair in match_distance[:] if pair[-2] == centroid_ID or pair[1] == pair_ball_ID]
            centroids.remove(shortest_pair[-2])

            # update pair ball with the matched centroid
            for pair_ball in self.pair_ball:
                if pair_ball["ID"] == pair_ball_ID:
                    pair_ball["centroid_point"] = [shortest_pair[2],shortest_pair[3]]
                    pair_ball["trace"].append(pair_ball["centroid_point"])
                    pair_ball["p1"] = shortest_pair[4]
                    pair_ball["p2"] = shortest_pair[5]
                    break

        print("Update pair ball")
        print(self.pair_ball)
        print()
        # make new pair if still got object left in centroids
        for centroid in centroids:
            pair_ball = self.ball_pair_initialization(centroid)
            self.pair_ball.append(pair_ball)

        # check for frequency
        self.frequency_checking()


    def bound_tracking(self, frame, results, bound_ball_pair):
        # remove None result
        result = list(filter(None,results))

        ball_pair = copy.deepcopy(self.pair_ball)
        ball_pair = self.unbound_ball_screening(ball_pair)

        bound_ball_pair = self.unbound_ball_screening(bound_ball_pair)
        bound_ball_pair_copy = copy.deepcopy(bound_ball_pair)

        # initialize pred balls
        pred_ball_p1 = []
        pred_ball_p2 = []

        # sequence starts from notation 2 -> notation 1
        # notation 2 tracking
        if len(result) > 0:
            self.frame_sequence += 1
        if self.frame_sequence == 1:
            for bound_ball in bound_ball_pair:
                self.p2_balls.append(bound_ball)
        elif self.frame_sequence > 1:
            frame, pred_ball_p2, self.location, self.p2_balls = p2_track(self.location, self.p2_balls, bound_ball_pair, frame)

        # notation 1 tracking
        # if len(self.p1_balls) < 1:
        #     for bound_ball in bound_ball_pair_copy:
        #         self.p1_balls.append(bound_ball)
        # else:
        #     frame, pred_ball_p1, self.p1_balls = p1_track(ball_pair, self.p1_balls, bound_ball_pair_copy, frame)

        # finalized pred balls
        pred_balls = pred_ball_p1+pred_ball_p2
        return frame, pred_balls
