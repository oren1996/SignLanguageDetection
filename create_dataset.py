import os
import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import pickle # to save the model

Data_dir = './labels_'

mp_hands = mp.solutions.hands # load the hand pose model (i.e. the model that will detect the landmarks)
mp_drawing = mp.solutions.drawing_utils # utility function to draw landmarks on the image (point de reperes)
mp_drawing_styles = mp.solutions.drawing_styles # utility function to draw connections between landmarks (traits entre les points de reperes)

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3) # load the model with some parameters
data = []
labels = []


for dir_ in os.listdir(Data_dir):
    files = sorted(os.listdir(os.path.join(Data_dir, dir_))) # sort files in ascending order by name (i.e. image_1.jpg, image_2.jpg, etc.)
    for img_path in files:
        data_aux = [] # create an empty list to store the landmarks coordinates of the image
        img = cv2.imread(os.path.join(Data_dir, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if len(results.multi_hand_landmarks) == 1: # if the model detects only one hand in the image
        # if results.multi_hand_landmarks: # if the model detects a hand or more in the image. It will iterate over each hand detected in the image
            for hand_landmarks in results.multi_hand_landmarks: # iterate over each hand detected in the image
                ### DRAW LANDMARKS ###

                # mp_drawing.draw_landmarks(
                #     img_rgb,  # image to draw on it (in this case, the webcam frame)
                #     hand_landmarks,  # the image we draw on the webcam frame
                #     mp_hands.HAND_CONNECTIONS,  # hand connections (i.e. the lines connecting the landmarks)
                #     mp_drawing_styles.get_default_hand_landmarks_style(), # style of the landmarks
                #     mp_drawing_styles.get_default_hand_connections_style()) # style of the connections
                # plt.figure()
                # plt.imshow(img_rgb)
                # plt.show()

                ### GET LANDMARKS COORDINATES ###
                for i in range(len(hand_landmarks.landmark)): # iterate over each landmark detected in the image (i.e. 21 landmarks)
                    x = hand_landmarks.landmark[i].x # get the x coordinate of the first landmark (i.e. the wrist)
                    y = hand_landmarks.landmark[i].y # get the y coordinate of the first landmark (i.e. the wrist)
                    data_aux.append(x) # append the x coordinate of the landmark i to the list
                    data_aux.append(y) # append the y coordinate of the landmark i to the list

                data.append(data_aux) # append the list of landmarks coordinates of the image to the data list
                labels.append(dir_) # append the label of the image to the labels list

f = open('data.pickle', 'wb') # open a file to save the model
pickle.dump({'data': data, 'labels': labels}, f) # save the model
f.close() # close the file