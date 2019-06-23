from threading import Thread
from imutils import face_utils
import cv2
from keras import preprocessing

from Utils import sound_alarm, eye_aspect_ratio, get_face_utils, get_haar_cascades
from keras.models import load_model
import numpy as np

# variables for the alarm
ALARM_ON_FOCUS = False
ALARM_ON_Drowsiness = False
ALARM_ON_Predict = False

# variables to count the how many times the driver is not focused/tired
COUNTER_PREDICT = 0
COUNTER_FOCUS = 0
COUNTER = 0

# thresholds that signal danger
THRESH_PREDICT = 20
THRESH_COUNT = 30
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 15

SPEED = 0
class_name = ['focused', 'not_focused']
result = ''
model = load_model('focus_detector_model1.h5')

# getting the the face/eye haar cascades and the utils for the face
detector, predictor, (lStart, lEnd), (rStart, rEnd) = get_face_utils()
face_cascade, eye_cascade = get_haar_cascades()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # get each frame from the webcam
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # make the image gray
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)  # detect the face
    rects = detector(gray, 0)

    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        # compute the convex hull for the left and right eye, then visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        if ear < EYE_AR_THRESH:
            COUNTER += 1

            # if the eyes were closed for a sufficient number of
            # then sound the alarm
            if COUNTER >= EYE_AR_CONSEC_FRAMES and SPEED != 0:
                # if the alarm is not on, turn it on
                if not ALARM_ON_Drowsiness:
                    # make a thread to show the alert
                    ALARM_ON_FOCUS = True
                    ALARM_ON_Drowsiness = True
                    t = Thread(target=sound_alarm, args=("alarm.wav",))
                    t.daemon = True
                    t.start()
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:  # reset the counter an alarm
            COUNTER = 0
            ALARM_ON_Drowsiness = False

        for (x, y, w, h) in faces:  # if face detected reset the counter for focus and the alarm
            COUNTER_FOCUS = 0
            ALARM_ON_FOCUS = False
            # make a region of interest for the frame if face detected
            roi_gray = gray[y: y + h, x: x + w]
            roi_color = frame[y: y + h, x: x + w]
            cv2.imwrite("my-image.png", roi_color)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # setting the rectangle for face detection
            eyes = eye_cascade.detectMultiScale(roi_color)  # detect the eye

            for (ex, ey, ew, eh) in eyes:
                # show the eye by setting the region on interest
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                roi_gray1 = roi_gray[ey: ey + eh, ex: ex + ew]

                # prepare the image for prediction, setting the right dimension, color spectrum(gray) and the number
                # of images
                roi_predict = cv2.resize(roi_gray1, (60, 60))
                roi_predict = preprocessing.image.img_to_array(roi_predict)
                roi_predict = np.expand_dims(roi_predict, axis=0)

                # predict and select the correct label for that image
                prediction = model.predict(roi_predict)
                cv2.imwrite("my-image1.png", roi_gray1)
                result = class_name[np.argmax(prediction)]

            cv2.putText(frame, result, (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            print(result)
            if result == 'not_focused':
                COUNTER_PREDICT += 1
            else:
                COUNTER_PREDICT = 0

        cv2.putText(frame, "EAR: {:.2f}".format(ear), (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # if no face detected than increment the counter and make a thread to show the alert
    # if is greater than the thresh hold
    COUNTER_FOCUS += 1
    if COUNTER_FOCUS >= THRESH_COUNT or COUNTER_PREDICT >= THRESH_PREDICT and SPEED != 0:
        if not ALARM_ON_FOCUS:
            ALARM_ON_FOCUS = True
            ALARM_ON_Drowsiness = True
            t = Thread(target=sound_alarm, args=("alarm.wav",))
            t.daemon = True
            t.start()
        cv2.putText(frame, "FOCUS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(frame, str(SPEED) + " Km/h", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if cv2.waitKey(20) & 0xFF == ord('w'):  # simulates when the driver presses on the throttle
        SPEED += 10
    elif cv2.waitKey(20) & 0xFF == ord('s') and SPEED > 0:  # simulates when the driver presses on the brake
        SPEED -= 10

    if SPEED == 0:  # when the speed is 0 reset the counters
        COUNTER_PREDICT = 0
        COUNTER_FOCUS = 0
        COUNTER = 0

    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
