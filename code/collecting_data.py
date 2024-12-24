#%

#Importing dependencies
import keyboard
import os
from functions import *

#Camera check
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera offline/ not accessible")
    exit()

actions = np.array(['hello','thanks']) #actions to be recorded
PATH = os.path.join('data')

#Number of sequences and frames respectively
sequences = 30
frames = 15

#Making folders for data collection
for action in actions:
    for sequence in range(sequences):
        try:
            os.makedirs(os.path.join(PATH, action, str(sequence)))
        except:
            pass

#Main process
with mp.solutions.holistic.Holistic(min_tracking_confidence=0.5, min_detection_confidence=0.5) as holistic:
    for action in actions:
        for sequence in range(sequences):
            for frame in range(frames):
                if frame == 0:
                    #For the first frame of every action to be recorded, give the user an option to start using space key 
                    while True:
                        if keyboard.is_pressed(' '):
                            break
                        _, image = cap.read()

                        results = image_process(image, holistic)
                        draw_landmarks(image, results)

                        image = cv2.flip(image, 1)
                        cv2.putText(image, 'Recording action {} sequence {}'.format(action, sequence), (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)
                        cv2.putText(image, 'Paused', (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                        cv2.putText(image, 'Press spacebar to start', (20, 420),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                        cv2.imshow('Live Feed',image)
                        cv2.waitKey(1)

                else:
                    _, image = cap.read()

                    results = image_process(image, holistic)
                    draw_landmarks(image, results)

                    image = cv2.flip(image, 1)
                    cv2.putText(image, 'Recording...', (20, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                    cv2.imshow('Live Feed', image)
                    cv2.waitKey(1)

                kp = extract_kp(results)
                path = os.path.join(PATH,action,str(sequence),str(frame))
                np.save(path,kp)

    #Closing camera after recording 
    cap.release()
    cv2.destroyAllWindows()



