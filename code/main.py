#&

#Importing dependencies

import keyboard
from functions import *
from tensorflow.keras.models import load_model
import os

#Variables that we will use

PATH = os.path.join('data')
actions = np.array(os.listdir(PATH))
model = load_model('model.h5') #Loading our model of trained weights
sentence, kp, prev_prediction = [],[],[] #Specifically our sentence, the key points we extract and the previous prediction made by our model
cap = cv2.VideoCapture(0)

with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        
        #Same as before
        _,frame = cap.read()
        results = image_process(frame, holistic)
        kp.append(extract_kp(results))
        draw_landmarks(frame,results)
        
        #Since my shape was (None, 15, 126) I have set the key point length to 15, I recommend checking your shape in model.py and adjusting the value as needed  
        if len(kp) == 15:
            kp = np.array(kp)
            prediction = model.predict(kp[np.newaxis,:,:])
            kp = []
            
            #Checking if our accuracy is greater than 0.9 (on a scale of 0 to 1)
            if np.amax(prediction) > 0.9:
                if prev_prediction != actions[np.argmax(prediction)]:
                    sentence.append(actions[np.argmax(prediction)])
                    prev_prediction = actions[np.argmax(prediction)]

        #If the length of our sentence exceeds 7 words, we remove the first seven words. Can be adjusted depending on your needs
        if len(sentence) > 7:
            sentence = sentence[-7:]
         #Clear everything key
        if keyboard.is_pressed(' '):
            sentence,kp,prev_prediction = [],[],[]

        #Clear last word key
        if keyboard.is_pressed('shift'):
            if len(sentence) > 0:
                sentence = sentence[:-1]

        #Ensuring live feed is right and has the sentence displayed on our feed
        image = cv2.flip(frame, 1)
        cv2.putText(image, ' '.join(sentence),  (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
        cv2.imshow('Live Feed', image)
        cv2.waitKey(1)

        #Check if close (X) button is pressed on the window
        if cv2.getWindowProperty('Live Feed',cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()

