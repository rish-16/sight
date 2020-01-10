from sight import Sightseer
import proc

ss = Sightseer()
data = ss.load_webcam(set_gray=False, return_data=True)

proc.xml_to_csv("./train/xml/", "/train/csv/")

print (data.shape)