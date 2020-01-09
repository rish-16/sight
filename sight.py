import tensorflow as tf
import numpy as np
from PIL import ImageGrab
import cv2

class Sightseer(object):
	def webcam(self, return_data=False, set_gray=True, kill_key="q", width=160, height=120):

		cap = cv2.VideoCapture(0)
		cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
		cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

		frames = []

		while True:
			ret, frame = cap.read()

			if set_gray:
				new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				new_frame = cv2.flip(new_frame, 1)
				frames.append(new_frame)
				cv2.imshow('frame', new_frame)
			else:
				frame = cv2.flip(frame, 1)
				cv2.imshow('frame', frame)
				frames.append(frame)

			if cv2.waitKey(1) & 0xFF == ord(kill_key):
				frames = np.array(frames)

				if return_data:
					return frames

				break

		cap.release()
		cv2.destroyAllWindows()	
		
	def screen_grab(self, set_gray=True, write_data=True, return_data=True, kill_key="q", filename='output.avi', width=400, height=400):
		fourcc = cv2.VideoWriter_fourcc(*'XVID')
		out = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))

		frames = []

		while True:
			img = ImageGrab.grab(bbox=(0, 0, width, height))
			imcv = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

			if write_data:
				out.write(imcv)

			if set_gray:
				new_frame = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2GRAY)
				frames.append(new_frame)
				cv2.imshow('frame', new_frame)
			else:
				cv2.imshow('frame', imcv)
				frames.append(imcv)

			if cv2.waitKey(1) & 0xFF == 113:
				break

		out.release()
		cv2.destroyAllWindows()

		frames = np.array(frames)

		if return_data:
			return frames

	def load_vidsource(self, filepath, set_gray=True, kill_key="q"):
		vidcap = cv2.VideoCapture(filepath)
		frame, image = vidcap.read()
		frames = []
		while frame:
			frame, image = vidcap.read()
			print (image.shape)

			if set_gray:
				new_frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2GRAY)
				frames.append(new_frame)
				cv2.imshow('frame', new_frame)
			else:
				cv2.imshow('frame', image)
				frames.append(image)

			if cv2.waitKey(1) & 0xFF == ord(kill_key):
				break
		
		vidcap.release()
		cv2.destroyAllWindows()

		return frames
			