# Sign Language Detection Project

## Overview
This project involves a series of Python scripts designed to recognize and interpret sign language. Specifically, it aims to detect each letter in the sign language alphabet using image processing and machine learning techniques. The project encompasses the entire workflow from image collection and dataset creation to training a classifier and running real-time inference for sign language interpretation.

## Scripts Description

### 1. `collect_images_for_dataset.py`
This script collects images using a webcam, primarily for gathering data for the dataset. It captures frames which are likely used to represent different sign language letters.

### 2. `create_dataset.py`
Processes images (likely collected from `collect_images_for_dataset.py`) for machine learning. It includes image processing and hand pose detection using MediaPipe, preparing and labeling data for different sign language letters.

### 3. `train_classifier.py`
Used for training a machine learning classifier, employing the RandomForest algorithm from scikit-learn. This script processes the prepared data, splits it into training and testing sets, and evaluates the model's performance.

### 4. `inference_classifier.py`
Runs inference using the trained classifier, likely using a model trained by `train_classifier.py` to predict sign language letters in real-time, possibly from a webcam feed.

## Installation and Usage

### Prerequisites
- Python 3.x
- Libraries: OpenCV, scikit-learn, MediaPipe, matplotlib, numpy, pickle

### Setup and Running
- Ensure all required libraries are installed.
- Follow this sequence for a complete workflow:
  1. `collect_images_for_dataset.py` - to collect sign language images.
  2. `create_dataset.py` - to process and prepare the sign language dataset.
  3. `train_classifier.py` - to train the classifier on the sign language dataset.
  4. `inference_classifier.py` - for real-time sign language letter detection.
