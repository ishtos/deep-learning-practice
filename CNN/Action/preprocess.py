import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture('sample.mov')
ret, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray)

x = faces[0][0]
y = faces[0][1]
w = faces[0][2]
h = faces[0][3]
margin = 100

i = 0
while(cap.isOpened()):
    i += 1
    cv2.rectangle(frame, (x-margin, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('frame', frame)
    cv2.imwrite('./img/img_{:05}.png'.format(i), frame[y:y+h, x-margin:x+w])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()
