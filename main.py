import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from PIL import Image
from absl import app, flags, logging
import cv2
import numpy as np
from absl.flags import FLAGS
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.models import load_model
from core.utils import *
from centroid_tracking.tracker import Tracker
from core.analysis import analysis
from core.analysis import create_dashboard
from core.posemodule import poseDetector
import tkinter as tk

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights_ball', './checkpoints/3l4b3_ball_416',
                    'path to weights ball file')
flags.DEFINE_string('weights_palm', './checkpoints/custom-tiny-palm-416',
                    'path to weights palm file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', True, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4-tiny-3l', 'yolov3 or yolov4')
flags.DEFINE_string('video', 'src/video0.mov', 'path to input video')
flags.DEFINE_float('iou', 0.25, 'iou threshold')
flags.DEFINE_float('score', 0.30, 'score threshold')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_string('output', 'output.avi', 'path to output video')
flags.DEFINE_string('demo_output', 'demo.avi', 'path to demo output video')
flags.DEFINE_string('ptn_model', 'checkpoints/pattern_model.h5', 'path to pattern recognition model')
flags.DEFINE_boolean('gpu', True, 'activate gpu - True else False')

def main(_argv):
    # initialize all the FLAGS setting
    input_size = FLAGS.size
    video_path = FLAGS.video
    gpu = FLAGS.gpu
    weights_ball = FLAGS.weights_ball
    weights_palm = FLAGS.weights_palm
    score = FLAGS.score
    iou = FLAGS.iou
    pattern_model = FLAGS.ptn_model
    output = FLAGS.output
    demo_output = FLAGS.demo_output
    output_format = FLAGS.output_format

    if gpu:
        # set up gpu setting
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)

    # load in all the models
    saved_model_loaded_ball = tf.saved_model.load(weights_ball, tags=[tag_constants.SERVING])
    infer_ball = saved_model_loaded_ball.signatures['serving_default']
    pattern_model = load_model(pattern_model)
    pose_detector = poseDetector() # human pose estimator

    # read in the video
    print("Video from: ", video_path)
    try:
        vid = cv2.VideoCapture(int(video_path)) # 0 - real time camera access
    except:
        vid = cv2.VideoCapture(video_path) # else - video input

    # get os resolution for display purpose
    image_width, image_height = int(vid.get(3)), int(vid.get(4))
    root = tk.Tk()
    screen_height = root.winfo_screenheight()
    screen_width = root.winfo_screenwidth()
    if image_width > screen_width: # prevent oversize video
        image_width = 1280
        image_height = 720

    # initialize video writer
    if output:
        # fps = int(vid.get(cv2.CAP_PROP_FPS))
        fps = 20
        codec = cv2.VideoWriter_fourcc(*output_format)
        out = cv2.VideoWriter(output, codec, fps, (image_width,image_height))
        demo_out = cv2.VideoWriter(demo_output, codec, fps, (image_width,image_height))

    tracker = Tracker() # initialize tracker
    ptns = []

    # start capturing and detection
    while True:
        return_value, frame = vid.read()
        if return_value:
            # simulation frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (image_width,image_height))
            demo = np.zeros((image_height, image_width, 3), np.uint8)
            image = Image.fromarray(frame)
        else:
            print("Video processing complete")
            break

        # video preprocess
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        prev_time = time.time()

        # capture the detection box
        batch_data = tf.constant(image_data)
        pred_bbox_ball = infer_ball(batch_data)
        for key, value in pred_bbox_ball.items():
            boxes_ball = value[:, :, 0:4]
            pred_conf_ball = value[:, :, 4:]

        # non max suppression
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes_ball, (tf.shape(boxes_ball)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf_ball, (tf.shape(pred_conf_ball)[0], -1, tf.shape(pred_conf_ball)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=iou,
            score_threshold=score
        )

        # finalized pred bbox
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]

        # human pose estimation
        demo = pose_detector.findPose(frame, demo)
        lmList = pose_detector.findPosition(demo, draw=False)
        right_palm, left_palm = pose_detector.findPalm()

        # perform unbound tracking
        pair_ball = tracker.track(frame, pred_bbox)
        bound_ball = mapping(pair_ball, [right_palm,left_palm],True)
        tracker.object_checking(bound_ball)
        bound_ball = mapping(tracker.pair_ball, [right_palm,left_palm])
        bound_ball_copy = copy.deepcopy(bound_ball)
        unbound_results = classification(frame, bound_ball, tracker.prev_pair_ball, tracker.pair_ball, pattern_model)

        # analysis result and display on dashboard
        frame = create_dashboard(frame)
        frame = analysis(tracker.pair_ball, frame)
        frame = pose_detector.distance_estimation(frame)
        frame, dmeo = pose_detector.find_Elbow_angle(frame, demo)

        # perform bound tracking
        demo, pred_balls = tracker.bound_tracking(demo, unbound_results, bound_ball_copy)
        bound_results = classification(frame, pred_balls, tracker.prev_pair_ball, tracker.pair_ball, pattern_model)
        results = unbound_results + bound_results
        bound_ball.extend(pred_balls)

        # display result - simulation and draw bbox on frame
        demo, ptns = display_demo(demo, results, ptns, bound_ball, tracker.pair_ball, [right_palm,left_palm])
        image = draw_bbox(frame, bound_ball, tracker.pair_ball, [right_palm,left_palm])

        # display frame
        curr_time = time.time()
        exec_time = 1.0 / (curr_time - prev_time)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        re_image = cv2.resize(image, ((screen_width // 4) - 140, (screen_height // 2) - 80))
        re_demo = cv2.resize(demo, ((screen_width // 4) - 140, (screen_height // 2) - 80))
        print("FPS: %.2f" % exec_time)
        print()
        print()

        cv2.imshow("frame", image)
        cv2.imshow("simulation", demo)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

        # save the videos
        if output:
            out.write(image)

        if demo_output:
            demo_out.write(demo)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
