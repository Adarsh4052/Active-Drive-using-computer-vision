#importing packages
from scipy.spatial import distance as dist
# threading for multiple threads can run together
from threading import Thread
import playsound

#imutils for image processing functions that can be used with opencv
import imutils
from imutils.video import WebcamVideoStream
from imutils import face_utils

import time

#import dlib
import dlib

#import twilio package for message sending
from twilio.rest import Client

# import opencv
import cv2

###Definitions of the variables/constants

#define shape predictor file name into a variable
shape_predictor_file = "shape_predictor_68_face_landmarks.dat"

#define alert-sound file name into a variable
alert_sound_file = "funny-alarm.mp3"

#EAR limit is the limit of distance b/w eye lids after this person would be considered sleeping
EAR_limit = 0.25

#40 Frames should be on or below EAR_LIMIT to consider person is sleeping
EAR_UNDER_LIMIT_FRAMES = 25

# defining a counter for the frames and set initial value to zero
FRAME_COUNTER = 0
# defining a flag ans setting it to False for alarm sound is off, True for alarm sound on
ALERT = False



#Calculation of Eye Aspect Reatios
def ear_calc(i):
    # euclidean distances calculation for vertical landmarks
    v1 = dist.euclidean(i[1], i[5])
    v2 = dist.euclidean(i[2], i[4])

    # euclidean distances calculation for horizontal landmarks
    h = dist.euclidean(i[0], i[3])

    # compute the eye aspect ratio
    ratio = (v1 + v2) / (2.0 * h)

    # return the eye aspect ratio
    return ratio



## function to play sound parallely
def play_audio(file):
    # play the Mp3 file
    playsound.playsound(file)


# The trained model named shape_predictor_68_face_landmarks.dat we are using was an trained model which uses
#   Histogram of Oriented Gradients feature combined with a linear classifier, an image pyramid,
#   and sliding window detection scheme
# Ref: http://dlib.net/face_landmark_detection.py.html
# Ref: https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/


# loading the DLIB's face detector/predictor
dlib_detector = dlib.get_frontal_face_detector()
print("Logs: DLIB Face Landmark Predictor has been loaded...")

# creating a variable for face landmark prediction, and load the pre-trained model on facial landmarks
dlib_predictor = dlib.shape_predictor(shape_predictor_file)

#Fetching the landmarks for the eyes and storing the beginning and end of the left and right eyes
(left_beg, left_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(right_beg, right_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# initializing the camera for live detection
cam_stream = WebcamVideoStream(src=0).start()
print("Logs: Camera started ~ 📷 ~ ")



# Starting while loop for the frames of the camera streams
while True:

    #Read the camera stream frame by frame
    stream_frame = cam_stream.read()
    #print("Logs: Reading the stream frame by frame...")

    #Resizing the frame width to 600
    stream_frame = imutils.resize(stream_frame, width=600)

    # convert the frames into grayscale
    gray_scale = cv2.cvtColor(stream_frame, cv2.COLOR_BGR2GRAY)

    # From the frame, detect the face, facial landmarks
    face_detect = dlib_detector(gray_scale, 0)

    # Exit the app, press x
    #cv2.putText(stream_frame, "Press button x to exit the App", (300, 410),
     #           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2 )


    # iterate over the face landmarks
    for f_land in face_detect:

        #detect the facial landmarks
        shape_of_land = dlib_predictor(gray_scale, f_land)
        shape_of_land = face_utils.shape_to_np(shape_of_land)

        left_eye = shape_of_land[left_beg:left_end]
        right_eye = shape_of_land[right_beg:right_end]

        #Calculate the eye aspect ratios for left and right eyes
        left_aspect_ratio = ear_calc(left_eye)
        right_aspect_ratio = ear_calc(right_eye)

        # Take average of the ratios
        eye_aspect_ratio = (left_aspect_ratio + right_aspect_ratio) / 2.0

        #Using the opencv to detect and draw boundries around the eyes
        left_eye_hull = cv2.convexHull(left_eye)
        right_eye_hull = cv2.convexHull(right_eye)
        cv2.drawContours(stream_frame, [left_eye_hull], -1, (255, 0, 0), 2)
        cv2.drawContours(stream_frame, [right_eye_hull], -1, (255, 0, 0), 2)

        # If eye aspect ratio is less than the EAR limit/ threshold value,
        #that means person is blinking or sleeping
        #increase the counter
        if eye_aspect_ratio < EAR_limit:
            FRAME_COUNTER += 1

            #if the eyes were closed for >=40 frames alert the driver by playing alert sound
            if FRAME_COUNTER >= EAR_UNDER_LIMIT_FRAMES:
                print("Logs: The driver was found sleepy!")

                if not ALERT:
                    ALERT = True

                    #start the thread the play the sound in background
                    if alert_sound_file != "":
                        thread = Thread(target=play_audio,
                                   args=(alert_sound_file,))
                        thread.deamon = True
                        #starting the thread
                        thread.start()
                        print("Logs: Played the Alert sound to wake up the driver...")

                        ###Twilio message sending
                        ###Twilio message sending
                        #send the message to a friend/family
                        # Account sid provided by Twilio
                        account_sid = ''
                        #Authentcation token provided by Twilio
                        auth_token = ''

                        # Phone number of the friend where we need to send the text message
                        friendsPhone = ''
                        # Phone number given by Twilio service
                        TwilioNumber = ''

                        client = Client(account_sid, auth_token)

                        client.messages.create(
                            to=friendsPhone,
                            from_=TwilioNumber,
                            body='Hey there! Your friend is sleeping while driving. Talk to him and save his life! ' + u'\U0001f680')
                        print("Logs: A message has been sent to the friend of the driver")

                #use open cv for showing text "sleeping alert" on  the frame
                cv2.putText(stream_frame, "Sleeping alert!!!", (100, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)



        #if the counter did not reach the LIMIT = 40 then reset the counter
        else:
            FRAME_COUNTER = 0
            ALERT = False

        #with opencv show eye aspect ratio on the screen
        cv2.putText(stream_frame, "Eye Aspect Ratio: {:.2f}".format(eye_aspect_ratio), (300, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # with opencv show eye counter on the screen
        cv2.putText(stream_frame, "Sleeping Frame Counter: {:.2f}".format(FRAME_COUNTER), (300, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 2)

    # show the frame
    cv2.imshow("Active-Drive                      Press button x to exit the App", stream_frame)
    key = cv2.waitKey(1) & 0xFF

    #if user wakes up from the alarm and presses the "x" key break the while loop and close the camera
    if key == ord("x"):
        break

# if x was pressed then close all the windows and stop the camera stream
cv2.destroyAllWindows()
cam_stream.stop()
