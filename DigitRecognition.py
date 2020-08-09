import cv2

from tensorflow.keras.preprocessing.image import img_to_array
import imutils

from tensorflow.keras.models import load_model
import numpy as np



def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x


digit_classifier = load_model('model.h5')

values = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]




cv2.namedWindow('cam')
cap = cv2.VideoCapture(0)

while True:
    
    ret, frame = cap.read()
    #frame = imutils.resize(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
   

    canvas = np.zeros((250, 300, 3), dtype = "uint8")
    frameClone = frame.copy()

    #val = input("Click Picture")

    cv2.imshow('cam', frameClone)

    if (True):

        roi = cv2.resize(gray, (28, 28))
        
        roi = roi.astype("float32")/255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis = 0)


        pred = digit_classifier.predict(roi)[0]
        emotion_probability = np.max(pred)
        label = values[pred.argmax()]
        print(label)

        #font = cv2.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 3, 8  # Creates a font
        x = 100  # position of text
        y = 200  # position of text
        cv2.putText(frameClone, label, (x, y), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0, 0, 255), lineType=cv2.LINE_AA)



    cv2.imshow('cam', frameClone)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
