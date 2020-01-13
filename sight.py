import tensorflow as tf
import numpy as np
from PIL import ImageGrab
import cv2

class Sightseer(object):
	def render_grayscale(self, frame):
		gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		return gray_frame

	def load_webcam(self, return_data=True, set_gray=True, kill_key="q", width=160, height=120):

		cap = cv2.VideoCapture(0)
		cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
		cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

		frames = []

		while True:
			ret, frame = cap.read()
			print (frame.shape)

			if set_gray:
				frame = self.render_grayscale(frame)
			
			frame = cv2.flip(frame, 1) # prevent lateral inversion
			cv2.imshow('frame', frame)
			frames.append(frame)

			if cv2.waitKey(1) & 0xFF == ord(kill_key):
				break

		cap.release()
		cv2.destroyAllWindows()	

		if return_data:
			frames = np.array(frames)
			return frames
		
	def screen_grab(self, set_gray=True, write_data=True, return_data=True, kill_key="q", filename='output.avi', width=400, height=400):
		fourcc = cv2.VideoWriter_fourcc(*'XVID')
		out = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))

		frames = []

		while True:
			img = np.array(ImageGrab.grab(bbox=(0, 0, width, height)))
			frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

			if write_data:
				out.write(imcv)

			if set_gray:
				frame = self.render_grayscale(img)

			cv2.imshow('frame', frame)
			frames.append(frame)

			if cv2.waitKey(1) & 0xFF == ord(kill_key):
				break

		out.release()
		cv2.destroyAllWindows()

		if return_data:
			frames = np.array(frames)
			return frames

	def load_source(self, filepath, return_data=True, set_gray=True, kill_key="q"):
		vidcap = cv2.VideoCapture(filepath)
		
		frame_exists, frame = vidcap.read()
		frames = []

		while frame_exists:
			frame_exists, frame = vidcap.read()
			print (frame.shape)

			if set_gray:
				frame = self.render_grayscale(frame)

			cv2.imshow('frame', frame)
			frames.append(frame)

			if cv2.waitKey(1) & 0xFF == ord(kill_key):
				break
		
		vidcap.release()
		cv2.destroyAllWindows()
		
		if return_data:
			frames = np.array(frames)
			return frames
			
	def stream(self, conn, port, buffer):
		pass