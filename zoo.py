import tensorflow as tf
import numpy as np
import cv2

class YOLO9000_Client(object):
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

class SSD_Client(object):
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

class MaskRCNN_Client(object):
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

