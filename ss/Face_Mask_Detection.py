import cv2
import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

faceCascade = cv2.CascadeClassifier("/Users/ngames/Desktop/CityU Project/ss/cv2classifier.xml")
nose_cascade = cv2.CascadeClassifier("/Users/ngames/Desktop/CityU Project/ss/CasCade/nose.xml")
handCascade = cv2.CascadeClassifier("/Users/ngames/Desktop/CityU Project/ss/CasCade/hand.xml")
model = load_model("/Users/ngames/Desktop/CityU Project/ss/mask_recog.h5")
vid = cv2.VideoCapture(0)

def face_mask_detector(frame):
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=5,
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    faces_list = []
    preds = []
    nose_rects = nose_cascade.detectMultiScale(gray, 1.3, 5)
    hand_rects = handCascade.detectMultiScale(gray, 1.3, 5)
    # 
    #     
    for (x, y, w, h) in faces:
        face_frame = frame[y : y + h, x : x + w]
        face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
        face_frame = cv2.resize(face_frame, (224, 224))
        face_frame = img_to_array(face_frame)
        face_frame = np.expand_dims(face_frame, axis=0)
        face_frame = preprocess_input(face_frame)
        faces_list.append(face_frame)
        #faces_list.
        if len(faces_list) > 0:
            preds = model.predict(faces_list[0])
        for pred in preds:
            (mask, withoutMask) = pred
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        noselen = len(nose_rects)
        color = (0, 0, 255) if noselen > 0 else (0, 255, 0)
        color = (0, 0, 255) if len(hand_rects) > 0 else (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)  

        # cv2.rectangle(frame, (xn,yn), (xn+wn,yn+hn), (0,255,0), 3)
        # break
    return frame

def procimg(image):
    print()



i = 0
ret1, frame1 = vid.read()
cv2.imshow('frame', face_mask_detector(frame1))
while True:  
    ret, frame = vid.read()
    cv2.imshow('frame', face_mask_detector(frame))  
    if cv2.waitKey(1) & 0xFF == ord('c'):
        break
vid.release()
cv2.destroyAllWindows()
