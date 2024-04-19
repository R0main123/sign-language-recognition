# Sign language Recognition

This repository was my first ever AI project. It was initially a school project but I was so interested that I added improvments.
## The project
The project consists in the training of a neural network for sign-language recognition. It contains a python file that runs the webcam and performs a live-recognition of sign language.
## Live recognition
The live recognition is performed only within the green square drawn on screen. The model also raises its prediction only when there is a 80% confidence or more on the prediction.


## File structure
The project includes 5 files:
- A main.py that runs the traning using Tensorflow. 
  - This file find the data, preprocess it, creates the model and save it in a file "model.h5"
- A main_depreciated.py
  - This file was the first version of the training. It uses Pytorch and a personalized training loop.
- cnn_model.py Contains an alternative version of the model
- hand_gesture_recognition.py handles the recognition of the signs through the camera.
- Finally, model100.py contains a model found on kaggle that performs 100% on the dataset.
