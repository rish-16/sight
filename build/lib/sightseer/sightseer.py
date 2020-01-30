import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# need to find Linux-friendly alternative
# from PIL import ImageGrab

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.animation as animation

class Sightseer(object):
	def __init__(self):
		self.filepath = None

	def render_grayscale(self, frame):
		gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		return gray_frame

	# Experimental
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
		
	# Experimental
	# def screen_grab(self, set_gray=True, write_data=True, return_data=True, kill_key="q", filename='output.avi', width=400, height=400):
	# 	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	# 	out = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))

	# 	frames = []

	# 	while True:
	# 		img = np.array(ImageGrab.grab(bbox=(0, 0, width, height)))
	# 		frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

	# 		if write_data:
	# 			out.write(imcv)

	# 		if set_gray:
	# 			frame = self.render_grayscale(img)

	# 		cv2.imshow('frame', frame)
	# 		frames.append(frame)

	# 		if cv2.waitKey(1) & 0xFF == ord(kill_key):
	# 			break

	# 	out.release()
	# 	cv2.destroyAllWindows()

	# 	if return_data:
	# 		frames = np.array(frames)
	# 		return frames

	def load_vidsource(self, filepath, return_data=True, set_gray=False):
		self.filepath = filepath
		vidcap = cv2.VideoCapture(filepath)

		print ("Extracting frames from video...")
		
		frames = []

		while vidcap.isOpened():
			frame_exists, frame = vidcap.read()

			if frame_exists == False:
				break

			if set_gray:
				frame = self.render_grayscale(frame)

			frames.append(frame)
		
		vidcap.release()
		cv2.destroyAllWindows()
		
		if return_data:
			frames = np.array(frames)
			return frames

	def load_image(self, filepath):
		self.filepath = filepath
		try:
			img = cv2.imread(filepath)
			return img
		except:
			raise FileExistsError ("File does not exist. You may want to check the filepath again.")

	def get_final_filepath(self, image_path):
		image_path = image_path.split('/')
		img_name = image_path[-1]
		img_name = img_name.split('.')
		img_name = img_name[0] + "_preds." + img_name[-1]
		image_path = "/".join(image_path[:-1]) + "/" + img_name

		return image_path	
	
	def render_image(self, image, save_image=False):
		plt.imshow(image)
		plt.show()

		if save_image:
			new_filepath = self.get_final_filepath(self.filepath)
			plt.savefig(new_filepath)
	
	def render_footage(self, frames):
		fig = plt.figure()
		final_frames = []

		for i in range(len(frames)):
			final_frames.append([plt.imshow(frames[i], animated=True)])
		
		ani = animation.ArtistAnimation(fig, final_frames, interval=50, blit=True, repeat_delay=1000)
		final_filename = self.get_final_filepath(self.filepath)
		ani.save(final_filename)

		plt.show()