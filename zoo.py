import tensorflow as tf
import tqdm
import numpy as np
import urllib.request
import cv2

class YOLO9000Client(object):
	# def __init__(self):
		# self.data = data

	def download_model(self):
		"""
		Downloads the weights and checkpoints from 
		online and saves them locally
		"""
		
		weights_url = "https://pjreddie.com/media/files/yolov3.weights"
		config_url = "https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg"

		# try:
		urllib.request.urlretrieve(weights_url, "./yolov3.weights")
		urllib.request.urlretrieve(config_url, "./yolov3.cfg")

		print ("Weights anc config file downloaded successfully!")
		# except:
			# print ("Oh no! Something went wrong with the download. Please try again.")

	def load_model(self, weights_path):
		"""
		Search for weights and load into model
		"""
		pass

	def get_predictions(self, save_data=True, render=True):
		"""
		Returns array of dictionaries of all probable classes
		with bounding boxes and confidence scores.

		Each element in the array is for each input image.
		"""
		pass

class SSDClient(object):
	def __init__(self, data):
		self.data = data

	def load_weights(self, weights_path):
		"""
		Search for weights and load into model
		"""
		pass

	def get_predictions(self, save_data=True, render=True):
		"""
		Returns array of dictionaries of all probable classes
		with bounding boxes and confidence scores.

		Each element in the array is for each input image.
		"""
		pass

class MaskRCNNClient(object):
	def __init__(self, data):
		self.data = data

	def load_weights(self, weights_path):
		"""
		Search for weights and load into model
		"""
		pass

	def get_predictions(self, save_data=True, render=True):
		"""
		Returns array of dictionaries of all probable classes
		with bounding boxes and confidence scores.

		Each element in the array is for each input image.
		"""
		pass

