from sight import Sightseer
from zoo import YOLOv3Client
import matplotlib.pyplot as plt
from pprint import pprint

# downloading and configuring weights and hyperparams
yolo = YOLOv3Client()
yolo.load_model()

# preprocessing image to fit YOLO9000 specs
ss = Sightseer("./test_data/img/street.jpg")
image = ss.load_image()

# Getting bounding boxes and displaying image
preds, pred_img = yolo.get_predictions(image)
pprint (preds)
plt.imshow(pred_img)
plt.show()