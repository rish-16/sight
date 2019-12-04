import tensorflow as tf
import numpy as np
import cv2

class Sightseer(object):
	def __init__(self, weights_path):
		self.weights_path = weights_path

	def open_vidsource(self, write_data=False, set_gray=True, kill_key="q", width=160, height=120):

		cap = cv2.VideoCapture(0)
		cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
		cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

		frames = []

		while True:
			ret, frame = cap.read()

			if set_gray:
				new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				frames.append(new_frame)
				cv2.imshow('frame', new_frame)
			else:
				cv2.imshow('frame', frame)
				frames.append(frame)

			if cv2.waitKey(1) & 0xFF == ord(kill_key):
				break

		cap.release()
		cv2.destroyAllWindows()	

		if write_data:
			return frames