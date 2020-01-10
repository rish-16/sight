from sight import Sightseer

ss = Sightseer()
data = ss.load_webcam(set_gray=False, return_data=True)

print (data.shape)