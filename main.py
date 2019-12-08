from sight import Sightseer
from zoo import YOLO9000Client
# from dataproc import DataAnnotator

# ss = Sightseer()
# data = ss.open_vidsource(set_gray=True, write_data=True)

yolonet = YOLO9000Client()
yolonet.download_model()

# preds = yolonet.get_predictions(save_data=True, render=True)

# print (preds)