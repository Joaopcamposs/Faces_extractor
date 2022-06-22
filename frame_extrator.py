import cv2
import os

dir = os.getcwd()
dir2 = f'{os.getcwd()}\\videos'


def FrameCapture(path):
    vidObj = cv2.VideoCapture(path)

    count = 0

    success = 1

    while success:
        success, image = vidObj.read()
        cv2.imwrite(f'{dir}\\images\\frame{count}.jpg', image)
        count += 1


FrameCapture(f'{dir2}\\video.mp4')
