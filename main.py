from sightseer import Sightseer
from sightseer.zoo import YOLOv3Client

yolo = YOLOv3Client()
yolo.load_model()

ss = Sightseer()
frames = ss.load_vidsource("./test_data/img/london.mp4")
print (frames.shape)

preds, det_frames = yolo.framewise_predict(frames, stride=10, verbose=False)
ss.render_footage(det_frames)