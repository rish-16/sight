from sight.sight import Sightseer
from sight.zoo import YOLOv3Client

# downloading and configuring weights and hyperparams
yolo = YOLOv3Client()
yolo.load_model()

# preprocessing image to fit YOLO9000 specs
ss = Sightseer("./test_data/img/street3.jpg")
image = ss.load_image()

# obtaining bounding boxes and displaying image
preds, pred_img = yolo.predict(image, return_img=True, verbose=True)
ss.render_image(pred_img)