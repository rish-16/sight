from sightseer import Sightseer
from sightseer.zoo import MaskRCNNClient

rcnn = MaskRCNNClient()
rcnn.load_model() # downloads weights