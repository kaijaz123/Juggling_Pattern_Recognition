# Juggling_Pattern_Recognition

# Introduction
The project provides juggling pattern recognition in either real-time or video format. The code was written and serves
as my Final-Year-Project (FYP). This repository provides the real time balls detection and human pose estimation as well as common classification of juggling **SITESWAP** notation from notation 1 to notation 8.

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


# Output and details
Output | Simulation
--- | ---
![](https://user-images.githubusercontent.com/49195906/151332629-1c43b810-8451-4a01-9552-7c1a19804311.png) | ![](https://user-images.githubusercontent.com/49195906/151332456-56745ab9-e879-466c-aacc-f5eb204439d9.png)

![image](https://user-images.githubusercontent.com/49195906/151332629-1c43b810-8451-4a01-9552-7c1a19804311.png)


![image](https://user-images.githubusercontent.com/49195906/151332456-56745ab9-e879-466c-aacc-f5eb204439d9.png)





# Scripts
## centroid tracking
1. ```tracker.py```: Tracker class. Implemented centroid tracking algorithm and customized for this project usecase.
2. ```bound_tracking.py```: Tracking algoithm for bound object (balls).
