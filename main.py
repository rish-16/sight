from sight import Sightseer
from zoo import YOLOv3Client
import matplotlib.pyplot as plt

# downloading and configuring weights and hyperparams
yolo = YOLOv3Client()
yolo.load_model()

# preprocessing image to fit YOLO9000 specs
ss = Sightseer("./test_data/img/street2.jpeg")
image = ss.load_image()
new_image = yolo.preprocess(image)

# Getting bounding boxes and displaying image
preds, det_image = yolo.get_predictions(new_image)
plt.imshow(det_image)
plt.show()