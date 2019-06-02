import cv2


face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')


def test_face_true():
    image1 = cv2.imread('img/train/focused/0.png')
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    face1 = face_cascade.detectMultiScale(image1, scaleFactor=1.5, minNeighbors=5)

    assert len(face1) != 0


def test_face_false():
    image2 = cv2.imread('img/train/not_focused/0.png')
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    face2 = face_cascade.detectMultiScale(image2, scaleFactor=1.5, minNeighbors=5)

    assert len(face2) == 0


def test_eye_true():
    image = cv2.imread('img/train/focused/0.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(image, scaleFactor=1.5, minNeighbors=5)
    eyes = ''

    for (x, y, w, h) in face:
        roi1 = image[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi1)

    assert len(eyes) != 0


def test_ReLU():
    assert max(0, 1) == 1
    assert max(0, -1) == 0
    assert max(0, 103) == 103
    assert max(0, -3) == 0


test_face_false()
test_face_true()
test_eye_true()
test_ReLU()
