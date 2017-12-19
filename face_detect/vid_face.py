import cv2
import time

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
roi=cv2.imread('friends.jpg')
count = 0

#cascPath = "haarcascade_frontalface_default.xml"
cascPath = "haarcascade_frontalface_alt.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)
t0=time.time()
print 't0=',t0


while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=0)

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi = frame[y:y+h, x:x+w]

    # Display the resulting frame
    cv2.imshow('Video', frame)

    print(int(time.time()-t0))

    if int(time.time()-t0)%5 == 0:
        cv2.imwrite("frame%d.jpg" % count, frame)
        cv2.imwrite("face%d.jpg" % count, roi)
        print "clicked"
        time.sleep(1)
        count += 1

    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture

out.release()
video_capture.release()
cv2.destroyAllWindows()