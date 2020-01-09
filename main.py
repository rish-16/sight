from sight import Sightseer

ss = Sightseer()
# data = ss.screen_grab(set_gray=False)
data = ss.webcam(set_gray=False)

print (data)