# Based on tutorial https://google.github.io/mediapipe/solutions/pose.html
# Keypoints at Fig. 4

import sys
import mediapipe as mp
import numpy as np
import cv2
import pandas as pd

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic
LANDMARK_ENUM = mp.python.solutions.pose.PoseLandmark
LANDMARKS = list(map(lambda x: x.name, LANDMARK_ENUM._member_map_.values()))

def run_vid(VIDEO_NAME="00016.mts", PLAY_ANNOT=True):

    cap = cv2.VideoCapture("00016.mts")  # assuming files will have .mts format

    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # https://stackoverflow.com/a/49080967/5060208
    out = cv2.VideoWriter("ANNOT_" + VIDEO_NAME[:-4] + ".avi", fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
    keypoints = pd.DataFrame()


    frame_num = 0
    keypoints = pd.DataFrame()

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            frame_num += 1
            if not success or not image.any():
                print("ignore empty frame or black image")
                break

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image.flags.writeable = False
            results = pose.process(image)

            for idx_, data_point in enumerate(results.pose_landmarks.landmark):
                keypoints = keypoints.append({
                                    'X': data_point.x,
                                    'Y': data_point.y,
                                    'Z': data_point.z,
                                    'Visibility': data_point.visibility,
                                    'Landmark' : LANDMARKS[idx_],
                                    'Frame' : frame_num
                                    }, ignore_index=True)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
             # for watching the video
            if PLAY_ANNOT:
                cv2.imshow('pose', image)
                # necessary, otherwise it would not be displayed https://stackoverflow.com/questions/21810452/cv2-imshow-command-doesnt-work-properly-in-opencv-python
                cv2.waitKey(1)
            out.write(image)


    cap.release()
    out.release()

    keypoints.to_csv("ANNOT_" + VIDEO_NAME[:-4] +".csv")

if __name__ == "__main__":

    # check if a video file was passed

    #run_vid('00016.MTS', True)
    if len(sys.argv)>1:
        run_vid(sys.argv[1], sys.argv[2])
    else:
        run_vid()
