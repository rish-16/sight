import tensorflow as tf
import numpy as np
import cv2

class Sightseer(object):
	def __init__(self, weights_path):
		self.weights_path = weights_path

	def open_vidsource(self, write_data=False, set_gray=True, kill_key="q"):

		cap = cv2.VideoCapture(0)
		cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
		cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)		

		if write_data:
			self.write_vidsource()

		while True:
			ret, frame = cap.read()

			if set_gray:
				new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				cv2.imshow('frame', new_frame)
			else:
				cv2.imshow('frame', frame)

			if cv2.waitKey(1) & 0xFF == ord(kill_key):
				break

		cap.release()
		cv2.destroyAllWindows()				
			
	def write_vidsource(self):
		pass

class YOLOClient(object):
	def __init__(self):
		pass
	
class 