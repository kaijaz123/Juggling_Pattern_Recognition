# Juggling_Pattern_Recognition

# Introduction
The project provides juggling pattern recognition in either real-time or video format. The code was written and serves
as my Final-Year-Project (FYP). This repository provides the real time balls detection and human pose estimation as well as common classification of juggling **SITESWAP** notation from notation 1 to notation 8.

# Contents:
* 

# Getting Started (tested on Ubuntu 18.04 & Windows)
It is recommended to have tensorflow with GPU support for faster inference time (higher framerate) while running the program. Please do follow the installation step from NVIDIA to install CUDA and CUDNN into your system.

Please install the following necessary packages in python3.7 or higher virtual environment:
* tensorflow-gpu/tensorflow-cpu (2.2 or higher)
* keras (2.3.1 or higher)
* mediapipe (0.8.9 or higher)
* pillow
* cv2
* numpy
* tkinter
* scikit-learn
* matplotlib


# Result and details
Output | Simulation
--- | ---
![](https://user-images.githubusercontent.com/49195906/151332629-1c43b810-8451-4a01-9552-7c1a19804311.png) | ![](https://user-images.githubusercontent.com/49195906/151332456-56745ab9-e879-466c-aacc-f5eb204439d9.png)
* **Output** image shows the detected balls that attached with specific ID and palm along with the data dashboard on the top right which displays the analysis of player performance. 
* Balls with red bounding box covered is in unbound state which means its not grapsed in hand. 
* Balls with green bounding box covered is in bound state which means its grapsed in hand.
* **Simulation** image shows the human pose estimation along with the recognized pattern on the top left.
* Each recognized pattern is identified as the same colors of the balls.

## Sample video results

https://user-images.githubusercontent.com/49195906/151335153-601dd7ab-f737-49ef-8171-a5d36c248f02.mp4


https://user-images.githubusercontent.com/49195906/151335388-9a8c4243-baa5-4e90-9a27-bd0aa51315c3.mp4

# Scripts
## centroid tracking/
1. ```tracker.py```: Tracker class. Implemented centroid tracking algorithm and customized for this project usecase.
2. ```bound_tracking.py```: Tracking algoithm for bound object (balls).

## core/
1. ```analysis.py```: Dashboard creation and the analysis of player's performance.
2. ```pattern.py```: Recognize siteswap notation/pattern.
3. ```posemodule.py```: Script for human pose estimation.
4. ```simulation.py```: Display the simulation frames along with the human pose, balls, and recognized siteswap notation/pattern.
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
2. ```output.avi```: Output video.
3. ```demo.avi```: Simulation video. 

# Instructions 
1. Please stand in a reasonable distance to laptop camera (or any other connected camera), recommended 1-1.2 meters. The estimated distance is displayed in the data dashboard in output video.
2. Currently supports only balls juggling and not more than 3 balls juggling.
3. Currently supports only siteswap notation 1 - 8 and some basic form of siteswap patterns, which variant of patterns are not recommended.
4. Please prevent same color between clothes, balls, and background (also light background).
