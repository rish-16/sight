# pip install sightseer

from sightseer.sightseer import Sightseer
from sightseer.zoo import YOLOv3Client
from pprint import pprint
import cv2

yolo = YOLOv3Client()
yolo.load_model() # downloads weights

# loading image from local system
ss = Sightseer()
frames = ss.load_vidsource("./test_data/img/tony.mp4")

# # getting labels, confidence scores, and bounding box data
preds, new_frames = yolo.framewise_predict(frames, return_vid=True)
pprint (preds)

fps = 25
out = cv2.VideoWriter("./test_data/img/tony_pred.avi", cv2.VideoWriter_fourcc(*'DIVX'), fps, (new_frames[0].shape[:2]))

for i in range(len(new_frames)):
	out.write(new_frames[i])
out.release()