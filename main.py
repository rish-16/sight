from sight import Sightseer
from zoo import YOLO9000Client
from pprint import pprint

# downloading and configuring weights and hyperparams
yolo = YOLO9000Client()
yolo.load_model()

# preprocessing image to fit YOLO9000 specs
ss = Sightseer("./test_data/img/fruits.jpg")
image = ss.load_image()
new_image = yolo.preprocess(image)

# Getting bounding boxes and displaying image
preds = yolo.get_predictions(new_image)
pprint (preds)
ss.render_image(new_image, preds) # experimental display method