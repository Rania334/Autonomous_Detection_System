# Autonomous_Detection_System

## Description

This repository contains an object detection system built using **Robot Operating System (ROS)** and **Python**, which utilizes **TensorFlow** and the **YOLO (You Only Look Once)** model for real-time detection of cones and pedestrians in video streams. The system processes images from a camera topic, enabling immediate identification and tracking of objects within the environment.

### Key Features

- **Real-time Object Detection**: Leverages the YOLO model for fast and accurate identification of cones and humans.
- **ROS Integration**: Developed within the ROS framework, facilitating seamless communication between nodes and easy integration with other robotic systems.
- **CoppeliaSim Compatibility**: Designed to work with CoppeliaSim, allowing for simulation and testing in a virtual environment.
- **User-friendly Visualization**: Displays processed video with bounding boxes and confidence scores, providing clear visualization of detection results.

### Use Cases

This project serves as a foundational tool for developing autonomous systems that require reliable object detection in dynamic environments, such as self-driving vehicles, robotics research, and automated monitoring systems.

## Demo

### Sample Outputs
![All](https://github.com/user-attachments/assets/1c6669e6-dc7a-46ce-83bc-e5bd7c9e2591)
![Car](https://github.com/user-attachments/assets/2dd5d1cd-e05a-4a04-8c73-57183a56730f)
![Cone](https://github.com/user-attachments/assets/0d1e4b88-157d-4362-b85e-8461aaeeea48)
![HumanCar2](https://github.com/user-attachments/assets/05eb7f0a-d596-4268-b342-ba5dfde6d386)

## Requirements

Before you begin, ensure you have the following installed:

- **Python 3.x**: The project is developed using Python 3. Install Python from [python.org](https://www.python.org/).
- **ROS**: Follow the [ROS installation guide](http://wiki.ros.org/ROS/Installation) to install the appropriate version of ROS for your system.
- **TensorFlow**: Install TensorFlow for Python using pip:
  ```bash
  pip install tensorflow

## Download the Trained Model
To use the YOLO model for object detection, you need to download the pre-trained model (bestcone.pt).

Download the trained YOLO model from the following link:
[Download Trained Model](https://drive.google.com/file/d/10MSY-K8sMaXu0xdKcFZHxpcTRBHfAB9G/view?usp=sharing)

