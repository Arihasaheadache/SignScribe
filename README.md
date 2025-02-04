# SignScribe
A hand sign translation model using Neural Networks

## How does it work?

The program is divided into three parts:

* Data Collection
* Model Training
* Main Program

## Data Collection

To collect data for teaching the model new signs or adding better quality data to already present signs, you need to run `data.py`. The program uses your live camera feed to capture hand signs, and allows for label modification and shows the frame and sequence numbers for better understanding of how to collect data. Once the program is finished running, it gives a folder of sub-folders of the various words. At the smallest unit, we will be left with numpy files of arrays containing values that we captured from the live feed.

## Training model

Once data is collected, we can run `model.py`. We are using Dense and LSTM as our layers to teach the model how to identify signs on the basis of the given data. Note: Depending on how your data has been collected, you may have to make slight changes to the code, the same is shown in the code itself.

After the model is done processing the data, it will be stored as an .h5 file for ease of use. You can easily change it to work with tensorflow lite depending on your needs by converting it into a .tflite file (Alternatively, simply saving it as a keras model file with the .keras extension also works).

## Running the main file!

If everything is functional, then you can finally run the `main.py` file! The model will load and input will be the live feed of your camera. By default I have kept Mediapipe Support off, which means you will not see keypoints on your live feed. To change that you can invoke the mediapipe function `draw_landmarks()` via the `functions.py` file. 

The sign language will be translated to text and printed on the live feed view. As usual you can commit any changes to the code to print to console or anything else. 
[By default accuracy is kept really high as to shoot for a more accurate result, but for cases of overfitting its best to lower the accuracy]


