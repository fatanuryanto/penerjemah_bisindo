import cv2
import time
import joblib
import streamlit as st
import pandas as pd
import mediapipe as mp 
import numpy as np


mp_drawing = mp.solutions.drawing_utils 
mp_holistic = mp.solutions.holistic
word=""
sentence=""
last_word_time = time.time()
RFC = joblib.load("RFC_model.sav")



with st.container():
    st.title("Yuk Ngobrol")
    st.subheader("Penerjemah Bahasa Isyarat")
    st.write("Silakan gunakan bahasa isyarat anda perhuruf")
    frame_placeholder = st.empty()
    text_word=st.empty()
    text_sentence=st.empty()

with st.container():
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.3,min_tracking_confidence=0.3) as holistic:
        while cap.isOpened():
            curr_word_time = time.time()

            # Read Camera Input
            ret, frame=cap.read()
            
            # BGR 2 RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Flip on horizontal
            image = cv2.flip(image, 1)

            # Set flag
            image.flags.writeable = False

            # Detections
            results = holistic.process(image)
            
            # Set flag to true
            image.flags.writeable = True

            # Draw landmark
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            
            if curr_word_time - last_word_time >= 2.0 and (results.right_hand_landmarks or results.left_hand_landmarks): # it has been at least 2 seconds
                # Get Landmark Coordinate
                lh = list(np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3))
                rh = list(np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3))

                # Satukan baris
                row = lh+rh

                # Predict
                X = pd.DataFrame([row])
                hand_class = RFC.predict(X)[0]
                hand_prob=RFC.predict_proba(X)[0]

                word=word+hand_class
                text_word.write(word)
                last_word_time = curr_word_time

            elif  curr_word_time - last_word_time >= 2.0 and (results.right_hand_landmarks or results.left_hand_landmarks) is None:
                    
                    if sentence.endswith(" "):
                        sentence=""
                        text_sentence.write(sentence)
                        word=""
                        text_word.write(word)
                        last_word_time = curr_word_time

                    else: 
                        sentence=sentence+" "+word
                        text_sentence.write(sentence)
                        word=""
                        text_word.write(word)
                        last_word_time = curr_word_time
                            

            frame_placeholder.image(image,channels="RGB")
            if cv2.waitKey(1) & 0xFF == ord('x'):
                break
            
    cap.release()
    cv2.destroyAllWindows()
    st.write("---")