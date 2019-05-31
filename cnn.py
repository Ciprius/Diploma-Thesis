import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import preprocessing
from glob import glob
import cv2
from keras.optimizers import Adam
from keras.utils import to_categorical

conv_1 = 16
conv_1_drop = 0.2
conv_2 = 32
conv_2_drop = 0.2
dense_1_n = 1024
dense_1_n_drop = 0.2
dense_2_n = 512
dense_2_n_drop = 0.2
lr = 0.001

epochs = 65
color_channels = 1
batch_size = 128
class_name = ['focused', 'not_focused']
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')

model = Sequential()
model.add(Conv2D(conv_1, kernel_size=(3, 3), activation='relu', input_shape=(60, 60, color_channels)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(conv_1_drop))

model.add(Conv2D(conv_2, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(conv_2_drop))

model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(dense_1_n, activation='relu'))
model.add(Dropout(dense_1_n_drop))
model.add(Dense(dense_2_n, activation='relu'))
model.add(Dropout(dense_2_n_drop))
model.add(Dense(len(class_name), activation='softmax'))

model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy', metrics=['accuracy'])

image_type_1 = []
image_type_2 = []
count = 0

for image_path in glob('D:/Faculta/licenta/Manually_Annotated_Images/Training/focused/*.*'):
    image1 = cv2.imread(image_path)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    face1 = face_cascade.detectMultiScale(image1, scaleFactor=1.5, minNeighbors=5)
    roi_eyes1 = ''
    count += 1
    print(1, " ", count, " ", len(face1))

    if len(face1) != 0:
        for (x, y, w, h) in face1:
            roi1 = image1[y:y + h, x:x + w]
            eyes1 = eye_cascade.detectMultiScale(roi1)
            for (ex, ey, ew, eh) in eyes1:
                roi_eyes1 = roi1[ey: ey + eh, ex: ex + ew]
                roi_eyes1 = cv2.resize(roi_eyes1, (60, 60))
        if len(roi_eyes1) != 0:
            image_type_1.append(preprocessing.image.img_to_array(roi_eyes1))
    else:
        eyes1 = eye_cascade.detectMultiScale(image1)
        if len(eyes1) != 0:
            for (ex, ey, ew, eh) in eyes1:
                roi_eyes1 = image1[ey: ey + eh, ex: ex + ew]
                roi_eyes1 = cv2.resize(roi_eyes1, (60, 60))
            image_type_1.append(preprocessing.image.img_to_array(roi_eyes1))

count = 0
for image_path in glob('D:/Faculta/licenta/Manually_Annotated_Images/Training/not_focused/*.*'):
    image2 = cv2.imread(image_path)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    face2 = face_cascade.detectMultiScale(image2, scaleFactor=1.5, minNeighbors=5)
    roi_eyes2 = ''
    count += 1
    print(2, " ", count, " ", len(face2))

    if len(face2) != 0:
        for (x, y, w, h) in face2:
            roi2 = image2[y:y + h, x:x + w]
            eyes2 = eye_cascade.detectMultiScale(roi2)
            for (ex, ey, ew, eh) in eyes2:
                roi_eyes2 = roi2[ey: ey + eh, ex: ex + ew]
                roi_eyes2 = cv2.resize(roi_eyes2, (60, 60))
        if len(roi_eyes2) != 0:
            image_type_2.append(preprocessing.image.img_to_array(roi_eyes2))
    else:
        eyes2 = eye_cascade.detectMultiScale(image2)
        if len(eyes2) != 0:
            for (ex, ey, ew, eh) in eyes2:
                roi_eyes2 = image2[ey: ey + eh, ex: ex + ew]
                roi_eyes2 = cv2.resize(roi_eyes2, (60, 60))
            image_type_2.append(preprocessing.image.img_to_array(roi_eyes2))


x_type_focused = np.array(image_type_1)
x_type_not_focused = np.array(image_type_2)

X = np.concatenate((x_type_focused, x_type_not_focused), axis=0)
X = X / 255.

y_type_focused = [0 for item in enumerate(x_type_focused)]
y_type_not_focused = [1 for item in enumerate(x_type_not_focused)]
y = np.concatenate((y_type_focused, y_type_not_focused), axis=0)
y = to_categorical(y, num_classes=len(class_name))

model.fit(X, y, epochs=epochs, batch_size=batch_size)
model.summary()
model.save('focus_detector_model4.h5')
