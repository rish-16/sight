from sight import Sightseer

ss = Sightseer()
# data = ss.screen_grab(set_gray=False)
data = ss.load_vidsource(filepath="./tony.mp4", set_gray=False)

print (data.shape)