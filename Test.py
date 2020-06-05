import cv2
import tensorflow as tf
import keras
import numpy as np

from keras import backend as K
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for physical_device in physical_devices: tf.config.experimental.set_memory_growth(physical_device, True)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

labels_dict={0:'without_mask',1:'with_mask'}


model = keras.models.load_model('mask_recog.h5')

video_capture=cv2.VideoCapture(0)

while True:
    _,img=video_capture.read()
    img=cv2.flip(img,1,1)
    mini=cv2.resize(img,(img.shape[1],img.shape[0]))
    features=face_cascade.detectMultiScale(mini)




    color_dict = {0: (0, 0, 255), 1: (0, 255, 0)}


    coords = []
    for (x, y, w, h) in features:

        face=img[y:y+h,x:x+w]
        new_face=cv2.resize(face,(224,224))
        normalize=new_face/255.0
        resize_face=np.reshape(normalize,(1,224,224,3))
        resize_face=np.vstack([resize_face])
        predict = model.predict(resize_face)

        if (predict[0][0] > predict[0][1]):
            color, text = color_dict[1], "Mask"
        else:
            color, text = color_dict[0], "No_Mask"

        cv2.rectangle(img, (x, y), (x + w, y + h), color,thickness=2)
        cv2.putText(img, text, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


    cv2.imshow("DetectFACE",img)
    key=cv2.waitKey(10)
    if key==10:
        break;



video_capture.release()
cv2.destroyAllWindows()