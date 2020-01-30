from sightseer import Sightseer
from sightseer.zoo import YOLOv3Client

yolo = YOLOv3Client()
yolo.load_model() # downloads weights

# loading image from local system
ss = Sightseer()
frames = ss.load_vidsource("./test_data/img/london_1sec.mp4")
print (frames.shape)

# # getting labels, confidence scores, and bounding box data
preds, pred_frames = yolo.framewise_predict(frames)
ss.render_footage(pred_frames)