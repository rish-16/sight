from sight import Sightseer
from zoo import YOLO9000Client

yolo = YOLO9000Client()
yolo.load_model() # downloading and configuring weights into model

ss = Sightseer()
img = ss.load_image("./test_data/img/street.jpg")
new_img = yolo.preprocess(img)

preds = yolo.get_predictions(new_img)
ss.render_image(new_img, preds) # renders image on pyplot