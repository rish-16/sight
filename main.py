from sight import Sightseer
from zoo import YOLOClient
from dataproc import DataAnnotator

ss = Sightseer()
data = ss.open_vidsource(set_gray=True, write_data=True)

for i in data:
	print (i.shape)

print (data.shape)