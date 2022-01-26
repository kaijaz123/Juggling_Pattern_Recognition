import pandas as pd
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder

class data_generator:
    def __init__(self, data_size_total):
        self.data_size = data_size_total

    def generate_distance_level(self):
        # distance level
        # 1 - very close to camera (0.5m - 0.8m)
        # 2 - normal distance to camera (1m-1.3m)
        distance_level_1 = np.array([1 for count in range(self.data_size)])
        distance_level_2 = np.array([2 for count in range(self.data_size)])

        return distance_level_1, distance_level_2

    def generate_hands_level(self):
        """
        distance between both hands
        0 means catch the ball with same hand, 1 means different hand
        """
        same_amt = self.data_size // 2
        diff_amt = self.data_size // 2
        same_hand = np.array([0 for count in range(int(same_amt))])
        diff_hand = np.array([1 for count in range(int(diff_amt))])
        hands_level = np.hstack((same_hand,diff_hand))

        return hands_level

    def generate_ball_pattern(self):
        """
        divide data size equally into 9 patterns
        data format - [hand_level,bally_distance, ballx_distance, pattern]
        * hands_level will be stacked up later from the generate_hands_level func
        """
        size = self.data_size // 8
        p1 = np.array([[random.uniform(0,0.06),random.uniform(30.0,150.0),0] for count in range(size)])
        p3 = np.array([[random.uniform(0.08,0.31),random.uniform(40.0,200.0),1] for count in range(size)])
        p4 = np.array([[random.uniform(0.14,0.40),random.uniform(0,120.0),2] for count in range(size)])
        p5 = np.array([[random.uniform(0.35,0.57),random.uniform(40.0,200.0),3] for count in range(size)])
        p6 = np.array([[random.uniform(0.48,0.65),random.uniform(0,120.0),4] for count in range(size)])
        p7 = np.array([[random.uniform(0.65,1.0),random.uniform(40.0,200.0),5] for count in range(size)])
        p8 = np.array([[random.uniform(0.7,1.0),random.uniform(0,120.0),6] for count in range(size)])
        pNone = np.array([[random.uniform(0,0.11),random.uniform(8.0,50.0),7] for count in range(size)])
        ball_patterns = np.vstack((p4,p6,p8,pNone,p1,p3,p5,p7))

        return ball_patterns

    def data_preprocess(self, data):
        # split into x and y
        X = data[:,:-1]
        y = data[:,-1]

        # apply categorical transform
        y = to_categorical(y)

        # save the data as np format file
        np.save("x.npy",X)
        np.save("y.npy",y)

        return X, y

    def generate_data(self):
        # concat them all together
        ball_pattern = self.generate_ball_pattern()
        hands_level = self.generate_hands_level()
        data = np.column_stack((hands_level,ball_pattern))

        # preprocess data and split into x and y
        X, y = self.data_preprocess(data)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, y, test_size = 0.1, random_state = 10)
        print(self.y_train.shape)

if '__main__' == __name__:
    generator = data_generator(6400)
    generator.generate_data()
