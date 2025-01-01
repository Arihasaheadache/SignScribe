# SignScribe
A hand sign translation model using Neural Networks

## Why are you re-inventing the wheel?

Because I wanna learn how to make wheels, duh... All things aside, my fascination with sign language has been evident from the very start. Also, now that AI/ML is becoming a hot topic, I felt like building my own neural networks and ml models and replacing the ones I use here would be a nice learning curve.

## How does it work?

For now, the program simply collects your data (because why should the big tech guys have all your information)

## Data Collection

When running collecting_data.py your camera feed opens and instructions are given to record various hand signs. Once all actions are completed, it exits gracefully

## Training model

When running model.py, the compiled data is used to train the model and save the weights for further usage. This means your trained data is stored in a file which can be used wherever necessary

## Running the main file!

When running main.py, the trained model will be loaded and will be used to predict the hand signs captured by the camera during the live feed, and the appropriate word(s) will show up on the live feed itself!

### Note

I'm hoping to add better explanations here soon
