from threading import Thread

from scipy.spatial import distance as dist
import playsound
import dlib
from imutils import face_utils
import cv2


def sound_alarm(path):
    playsound.playsound(path)  # play an alarm sound


def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of vertical eye landmarks (x, y)-coordinates
    a = dist.euclidean(eye[1], eye[5])
    b = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal eye landmark (x, y)-coordinates
    c = dist.euclidean(eye[0], eye[3])

    ear = (a + b) / (2.0 * c)  # compute the eye aspect ratio
    return ear


def get_face_utils():
    # initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
    # grab the indexes of the facial landmarks for the left and right eye, respectively
    return dlib.get_frontal_face_detector(), \
           dlib.shape_predictor("shape_predictor_68_face_landmarks.dat"), \
           face_utils.FACIAL_LANDMARKS_IDXS["left_eye"], \
           face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


def get_haar_cascades():
    # init the face and eye haarcascade detectors from openCv
    return cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml'), \
           cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')


def fire_alarm(counter, threshold, alarm, speed):
    if counter >= threshold and speed != 0:
        # if the alarm is not on, turn it on
        if not alarm:
            # make a thread to show the alert
            alarm = True
            t = Thread(target=sound_alarm, args=("alarm.wav",))
            t.daemon = True
            t.start()
    return alarm
