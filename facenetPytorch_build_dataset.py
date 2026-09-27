'''
1. 本程式適合用來建立個人臉部辨識資料集（每次執行僅針對一個人） This script is used to build a custom face recognition dataset.
2. 資料集會儲存在資料夾中（每個人一個資料夾） The dataset is stored in a folder.
3. 圖片是使用電腦的攝影機拍攝的 The images are captured using the webcam of the computer.
4. 本程式使用 OpenCV 的 Haar cascade 來偵測影片串流中的人臉 The script uses OpenCV's Haar cascade to detect faces in the video stream.
5. 當按下 'k' 鍵時，程式會將原始影格儲存到磁碟 The script saves the original frame to disk when the 'k' key is pressed.
6. 當按下 'q' 鍵時，程式會退出 The script exits when the 'q' key is pressed.
7. 程式會列印已儲存的人臉影像總數，並清理影片串流與視窗 The script prints the total number of face images stored and 
	cleans up the video stream and windows.
8. 本程式可以使用以下指令從終端機執行 The script can be run from the terminal with the following command:
python facenetPytorch_build_dataset.py --cascade haarcascade_frontalface_default.xml --output pictures/wang

note: The haarcascade_frontalface_default.xml file can be downloaded from the OpenCV GitHub repository:
https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml

目前儲存完整照片，之後可以再進行裁切，或是直接在這裡進行裁切，儲存臉部照片。

Source: https://pyimagesearch.com/2018/06/11/how-to-build-a-custom-face-recognition-dataset/

'''

# import the necessary packages
import os
import cv2
import time
import imutils
import argparse
from pathlib import Path
from imutils.video import VideoStream

# read the cascade file: a pre-trained Haar Cascade face-detection model.
cascadePath = Path(__file__).resolve().parent / "haarcascade_frontalface_default.xml"
detector = cv2.CascadeClassifier(str(cascadePath))
if detector.empty():
    raise FileNotFoundError(f"Cannot load cascade file: {cascadePath}")

username = input("Enter the userID for the face: ")
# 修改 outputPath 為你想要儲存圖片的資料夾路徑，資料夾名稱可以改成你想要的名字
outputPath = f'facenet_dataset/{username}' # folder to store the images
# create a folder to store the images if it does not exist
if not os.path.exists(outputPath):
	os.makedirs(outputPath)

# initialize the video stream, allow the camera sensor to warm up,
# and initialize the total number of example faces written to disk
print("[INFO] starting video stream...")
vs = VideoStream(src=1).start() # 對內攝影機, src=1 對外攝影機
time.sleep(2.0)
total = 0

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream, clone it, (just
	# in case we want to write it to disk), and then resize the frame
	# so we can apply face detection faster
	frame = vs.read()
	orig = frame.copy()
	frame = imutils.resize(frame, width=400) # resize the frame to width of 400 pixels for faster processing
	# detect faces in the grayscale frame
	rects = detector.detectMultiScale(
		cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, 
		minNeighbors=5, minSize=(30, 30))
	# loop over the face detections and draw them on the frame
	for (x, y, w, h) in rects:
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `k` key was pressed, write the *original* frame to disk
	# so we can later process it and use it for face recognition
	if key == ord("k"):
		p = os.path.sep.join([outputPath, "{}.png".format(
			str(total).zfill(5))])
		cv2.imwrite(p, orig)
		total += 1
	# if the `q` key was pressed, break from the loop
	elif key == ord("q"):
		break
	# print the total faces saved and do a bit of cleanup
print("[INFO] {} face images stored".format(total))
print("[INFO] cleaning up...")
cv2.destroyAllWindows()
vs.stop()