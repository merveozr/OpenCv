# videodan yüz algılama

import cv2 
# vid = cv2.VideoCapture('C:\\Users\\Merve\\Desktop\\openCv\\hear_cascade\\faces.mp4')
vid = cv2.VideoCapture(0) # Webcam üzerinden görüntüyü işlemek.
face_cascade = cv2.CascadeClassifier('C:\\Users\\Merve\\Desktop\\openCv\\hear_cascade\\frontalface.xml')


#kaç tane frame olduğunu bilmiyoruz
while 1:
    #_,frame = vid.read()
    ret,frame = vid.read()
    frame = cv2.flip(frame,1)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.4,4)

    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow("image",frame)

    if cv2.waitKey(5) & 0xFF ==ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
