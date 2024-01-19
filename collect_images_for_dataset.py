import cv2
import os
import time

webcam = cv2.VideoCapture(0)

for i in range(26):
    count = 1
    while count <= 50:
        ret, frame = webcam.read()

        if not ret:
            break  # Break if frame not captured properly

        cv2.imshow('frame', frame)
        cv2.waitKey(40)
        s = 'P'
        directory = os.path.join('.', 'labels_', str(chr(65 + i))) # create a directory to save the images
        if not os.path.exists(directory):
            os.makedirs(directory) # make the directory

        file_path = os.path.join(directory, f'image_{count}.jpg') # create the file path to save the image

        cv2.imwrite(file_path, frame) # save the image

        print(f'image_{count}.jpg saved')
        count += 1
print("Preparing for next letter")
time.sleep(5)
print("Starting next letter")


webcam.release() # release the memory allocated for the webcam
cv2.destroyAllWindows() # destroy the window in which the webcam was vizualized
