from sight import Sightseer
from sight.zoo import YOLO9000

ss = Sightseer()
data = ss.load_vidsource(filepath="./tony.mp4", set_gray=True)

print (data.shape)