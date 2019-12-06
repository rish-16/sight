from sight import Sightseer
from zoo import YOLOClient
from dataproc import DataAnnotator

ss = Sightseer()
data = ss.open_vidsource(set_gray=True, write_data=True)

yolonet = YOLOClient(data)
preds = yolonet.get_predictions(save_data=True, render=True)

print (preds)