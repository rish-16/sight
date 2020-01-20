from sight import Sightseer
from zoo import YOLO9000Client
from pprint import pprint
import matplotlib.pyplot as plt

# downloading and configuring weights and hyperparams
yolo = YOLO9000Client()
yolo.load_model(verbose=False)

# preprocessing the image to fit YOLO9000 specs
ss = Sightseer("./test_data/img/street2.jpeg")
image = ss.load_image()
new_image = yolo.preprocess(image)

# Getting bounding boxes and displaying image
preds, pred_image = yolo.get_predictions(new_image, return_image=True)
pprint (preds)
ss.render_image(pred_image)