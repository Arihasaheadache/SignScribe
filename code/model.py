#%

#Importing dependencies

from functions import *
import os
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

#Specifically the tf models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM

PATH = os.path.join('data')
actions = np.array(os.listdir(PATH))

sequences = 30
frames = 15

#Making a label map for our actions
label_map = {label:num for num, label in enumerate(actions)}

landmarks, labels = [], []

for action in actions:
    for sequence in range(sequences):
        temp = []
        for frame in range(frames):
            npath = np.load(os.path.join(PATH, action, str(sequence), str(frame) + '.npy'))
            temp.append(npath)
        landmarks.append(temp)
        labels.append(label_map[action])

X, Y = np.array(landmarks), to_categorical(labels).astype(int)

#Splitting our training and testing data
X_train, X_test, Y_train,Y_test = train_test_split(X,Y, test_size=0.05,random_state=34,stratify=Y)

#Layers which the model will be trained using
model = Sequential()
model.add(LSTM(32, return_sequences=True, activation='relu', input_shape=(15,126)))
model.add(LSTM(64, return_sequences=True, activation='relu'))
model.add(LSTM(32, return_sequences=False, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

#Compiling data and training model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train,Y_train,epochs=100)

#You can remove this, it's just the summary no actual purpose other than learning
print(model.summary())

#Saving that data
model.save('model.h5')

