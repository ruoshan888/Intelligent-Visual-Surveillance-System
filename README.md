# Intelligent Visual Surveillance System with Moving Object Recognition

This project implements an intelligent visual surveillance system using classical image processing techniques to detect and recognize moving objects in video.
This project was developed for the course ARC6144 Machine Vision Image Process.

## Features

- Moving object detection  
- Background subtraction  
- Morphological filtering  
- Connected component analysis  
- Object classification  
- Bounding box annotation  

## Object Classes

The system recognizes:

- Person  
- Motorcycle  
- Car  

## Techniques Used

- Python  
- OpenCV  
- BackgroundSubtractorMOG2  
- Otsu Thresholding  
- Morphological Operations  
- Connected Component Labelling  

## Description

The system processes indoor and outdoor video clips to detect moving objects and classify them based on simple geometric features such as blob size and shape.

Detected objects are labelled and highlighted using bounding boxes.
