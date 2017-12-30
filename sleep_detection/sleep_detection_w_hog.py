from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import imutils
import time
import dlib
import cv2

ASPECT_RATIO_THRESHOLD = 0.3  # If aspect ratio falls below this value, eye is closed
MAX_CONSEC_FRAMES = 20  # Max. no. of consecutive frames for which eye can stay shut without triggering alarms


def wakeUp(path):  # Plays sound
    global is_alarm_on  # To enable changing the value
    playsound.playsound(path)
    is_alarm_on = False


def aspectRatio(eye):
    a = np.linalg.norm(eye[1] - eye[5])  # Vertical Distance between the eyelids
    b = np.linalg.norm(eye[2] - eye[4])  # Vertical Distance between the eyelids
    c = np.linalg.norm(eye[0] - eye[3])  # Horizontal Distance of the eyeball
    # Computing the aspect ratio, this dips when the eyes shut due to smaller a and b
    ratio = (a + b) / (2.0 * c)
    return ratio


counter = 0  # To count number of consec frames eyes are closed
is_alarm_on = False
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
(leftStart, leftEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]  # Gets left eye indices
(rightStart, rightEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]  # Gets right eye indices
vs = VideoStream(0).start()
time.sleep(1.0)  # pause for feed

while True:
    frame = vs.read()  # Get frame from video
    # frame = imutils.resize(frame, width=450)  # Resize the frame
    # frame = cv2.resize(frame,(frame.height,450))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    faces = detector(gray, 0)  # Detect faces in image
    for face in faces:
        shape = predictor(gray, face)  # Get facial coordinates
        shape = face_utils.shape_to_np(shape)  # Convert facial cartesian coordinates to a numpy array
        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[leftStart:leftEnd]  # Slice of coordinates corresponding to left eye
        rightEye = shape[rightStart:rightEnd]  # Slice of coordinates corresponding to right eye
        leftAspectRatio = aspectRatio(leftEye)  # Get aspect ratio of left eye
        rightAspectRatio = aspectRatio(rightEye)  # Get aspect ratio of left eye
        avgAspectRatio = (leftAspectRatio + rightAspectRatio) / 2.0  # Average of both eyes)
        file_write = open('output.txt', 'a')
        file_write.write(str(avgAspectRatio) + '\n')
        cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)  # Drawing contour around the left eye
        cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)  # Drawing contour around the right eye
        if avgAspectRatio < ASPECT_RATIO_THRESHOLD:  # If Aspect ratio dips below threshold
            counter += 1
            print(counter)
            if counter >= MAX_CONSEC_FRAMES:  # If eyes are closed for long enough
                # if the alarm is not on, turn it on
                if not is_alarm_on:
                    is_alarm_on = True
                    t = Thread(target=wakeUp, args=("weewoo.wav",))  # Separate thread so scanning isn't interrupted
                    t.daemon = True  # Daemon threads kill themselves automatically so we don't have to bother
                    t.start()
                # Write alert on the screen
                font = cv2.FONT_HERSHEY_SIMPLEX
                text = "YOU'RE DOZING!"
                textsize = cv2.getTextSize(text, font, 2.5, 12)[0]
                textX = int((frame.shape[1] - textsize[0]) / 2)
                textY = int((frame.shape[0] + textsize[1]) / 2)
                cv2.putText(frame, text, (textX, textY), font, 2.5, (0, 0, 255), 12)
        else:  # If not, reset counter and turn off the alarm
            counter = 0
            is_alarm_on = False
    cv2.imshow("Frame", frame)  # Display the frame
    key = cv2.waitKey(1) & 0xFF  # Bitmask 0b11111111 applied
    if key == ord("q"):  # If input is 'q'
        break

# Clean up the act
cv2.destroyAllWindows()
vs.stop()
