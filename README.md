# Juggling_Pattern_Recognition

# Introduction
The project provides juggling pattern recognition in either real-time or video format. The code was written and served
as my Final-Year-Project (FYP). This repository provides the real time balls detection and human pose estimation as well as common recognition of juggling **SITESWAP** notation from notation 1 to notation 8.

# Main Contents:
* Utilized You-Only-Look-Once (YOLO) for ball detection.
* Utilized mediapipe for human pose estimation.
* Tensorflow framework to build a model for classification of juggling siteswap notation.
* Centroid tracking algorithm for object tracker.

# Getting Started (tested on Ubuntu 18.04 & Windows)
It is recommended to have tensorflow with GPU support for faster inference time (higher framerate) while running the program. Please do follow the installation step from NVIDIA to install CUDA and CUDNN into your system.

Please install the following necessary packages in python3.7 (or higher version) virtual environment:
* tensorflow-gpu/tensorflow-cpu (2.2.0 or higher)
* keras (2.3.1 or higher)
* mediapipe (0.8.9 or higher)
* pillow
* cv2
* numpy
* tkinter
* scikit-learn
* matplotlib


# Results and details
Output | Simulation
--- | ---
![](https://user-images.githubusercontent.com/49195906/151481221-cbd6c553-73d0-4a53-a5e7-80dd992abc51.png) | ![](https://user-images.githubusercontent.com/49195906/151481300-249c3443-b74d-401a-81bc-c916823dd0f1.png)
* **Output** result shows the detected balls that attached with specific ID and palm along with the data dashboard on the top right which displays the analysis of player performance. 
* Balls with red bounding box covered is in unbound state which means its not grasped in hand. 
* Balls with green bounding box covered is in bound state which means its grasped in hand.
* **Simulation** result shows the human pose estimation along with the recognized pattern on the top left.
* Each recognized pattern is identified as the same color of the ball in the simulation result.
* Height of ball in bound state will not be shown since its grasped in hand hence the height distance will be 0 as always.
* The results will be stored in video files as output.avi and demo.avi.

## Sample video results

https://user-images.githubusercontent.com/49195906/151481530-abf78e0f-8f41-4dd6-9f14-a7667b5238a8.mp4


https://user-images.githubusercontent.com/49195906/151481551-adebd255-7827-4bd7-ae9d-bc9b99b69914.mp4

# Scripts

## centroid tracking/
1. ```tracker.py```: Tracker class. Implemented centroid tracking algorithm and customized for this project usecase.
2. ```bound_tracking.py```: Tracking algoithm for bound object (balls).
3. ```classes.names```: YOLO class name.
4. ```config.py```: YOLO properties config file.

## core/
1. ```analysis.py```: Script for dashboard creation and the analysis of player's performance.
2. ```pattern.py```: Script for recognize siteswap notation/pattern.
3. ```posemodule.py```: Script for human pose estimation.
4. ```simulation.py```: Script for displaying the simulation frames along with the human pose, balls, and recognized siteswap notation/pattern.
5. ```utils.py```: Contains functions that will call the scripts above.

## pattern_recog_model_generator/
1. ```data_generator.py```: Generate training and testing data.
2. ```model_train.py```: Build model and training script.
3. ```pattern_model.h5```: Trained model.
4. ```x.npz```: X dataset.
5. ```y.npz```: Y dataset.

## src/
1. Contains all the testing videos used in this project.

## ./
1. ```main.py```: Main program script.
2. ```output.avi```: Output video result.
3. ```demo.avi```: Simulation video result. 

# Instructions 
1. A high quality camera is recommended to be used for capturing. You may also use phone camera or any external camera for better image quality hence improve the performance.
2. Please stand in a reasonable distance to camera, recommended 1-1.2 meter. The estimated distance is displayed in the data dashboard in output video.
3. Currently supports only balls juggling and not more than 3 balls.
4. Currently supports only siteswap notation 1 - 8 and some basic form of siteswap patterns, which variant of patterns are not recommended.
5. Please prevent same color between clothes, balls, and background (also light background).
