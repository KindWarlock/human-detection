import cv2
import matplotlib.pyplot as plt
import numpy as np

from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input, Dropout, Activation, Reshape
#from sklearn.metrics import classification_report
from PIL import Image


import tensorflow as tf
from PyQt5.QtCore import QLibraryInfo


def create_model():
    model = Sequential()
    model.add(Conv2D(8, (5, 5), input_shape=(150, 150, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(16, (5, 5)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    model.summary()

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

model = create_model()
train_x = []
train_y = []

cam = cv2.VideoCapture(0)

ret, frame = cam.read()
f = open("img_count.txt", "r")
human_cnt, non_human_cnt = map(int, f.readline().split(' '))
f.close()
# TODO fit starting data

font                   = cv2.FONT_HERSHEY_SIMPLEX
pos                    = (10,100)
fontScale              = 1
fontColor              = (0,0,0)
thickness              = 3
lineType               = 2

predict = ''
while True:
    ret, frame = cam.read()
    if not ret:
        break

    img = cv2.resize(frame, (150, 150))

    cv2.putText(frame, predict, 
        pos, 
        font, 
        fontScale,
        fontColor,
        thickness,
        lineType)
    cv2.imshow('frame', frame)

    k = cv2.waitKey(10)
    if k > 0:
        if chr(k) == 'q':
            f = open("img_count.txt", "w")
            f.write(f"{human_cnt} {non_human_cnt}")
            f.close()
            break

        if chr(k) == 'y':
            train_x.append(img)
            train_y.append(1)
            im = Image.fromarray(img)
            im.save(f'my_set/human/{human_cnt}.png')
            # cv2.imwrite(f'my_set/human/{human_cnt}')
            human_cnt += 1

        if chr(k) == 'n':
            train_x.append(img)
            train_y.append(0)
            im = Image.fromarray(img)
            im.save(f'my_set/non-human/{non_human_cnt}.png')
            # cv2.imwrite(f'my_set/human/{non_human_cnt}')
            non_human_cnt += 1

        if chr(k) == 'p':
            img = img[np.newaxis, ...]
            predict = str(model.predict(img / .255, verbose=0))



    if len(train_x) >= 10:
        model.fit(np.array(train_x), np.array(train_y))
        print(f"Human imgs:{human_cnt}, non-numan imgs: {non_human_cnt}")
        train_x = []
        train_y = []

