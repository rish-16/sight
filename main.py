from sightseer import Sightseer
from sightseer.zoo import YOLOv3Client

yolo = YOLOv3Client()
yolo.load_model()

ss = Sightseer()
image = ss.load_image("./test_data/img/street.jpg")

preds, det_image = yolo.predict(image, return_img=True)
ss.render_image(det_image)