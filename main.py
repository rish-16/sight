# pip install sightseer

from sightseer import Sightseer
from sightseer.zoo import YOLOv3Client
from pprint import pprint

yolo = YOLOv3Client()
yolo.load_model() # downloads weights

# loading image from local system
ss = Sightseer()
frames = ss.load_vidsource("./test_data/img/london.mp4")

# # getting labels, confidence scores, and bounding box data
preds, pred_frames  = yolo.framewise_predict(frames)

out = cv2.VideoWriter("./test_data/img/london_pred.avi", cv2.VideoWriter_fourcc(*'DIVX'), 25, pred_frames[0].shape[:2])

for i in range(len(pred_frames)):
	out.write(pred_frames[i])
out.release()