from sight import Sightseer
from zoo import YOLO9000Client

yolonet = YOLO9000Client()
yolonet.download_model()

ss = Sightseer()
data = ss.open_vidsource(set_gray=True, write_data=True)

yolonet.get_predictions(data)