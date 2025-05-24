# Sign language Recognition

This repository is my first ever AI project, so please be kind :D. It was initially a school project but I was so interested that I added improvments.
## The project
The project consists in the training of a neural network for sign-language recognition. It contains a python file that runs the webcam and performs a live-recognition of sign language.
## Live recognition
The live recognition is performed only within the green square drawn on screen. The model also raises its prediction only when there is a 80% confidence or more on the prediction.


## File structure
The project includes 4 files:
- A Jupyter notebook train.ipynb that runs the training using Tensorflow. 
  - This file find the data, preprocess it, creates the model and save it in a file "hand_recognition_model.h5"
- hand_recognition_model.h5 is the saved version of the model.
- main.py handles the recognition of the signs through the camera.
- Finally, requirements.txt allows you to run the training and the live recognition, and change the model to run your own tests.

## How to use it?
### Build your own structure
To change the structure of the neural network, simply open train.ipynb and change the structure of the network in the 6th cell, then run the whole notebook.

### Run the live recognition
Run main.py. You may allow the use of your webcam. Ensure to be at a good distance from your camera, and place your hand in the green square. The prediction is shown in the upper left corner of the window.

