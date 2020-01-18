import os
import wget
import shutil

import cv2
import tensorflow as tf
import numpy as np

class YOLO9000Client(object):
	def __init__(self):
		self.data = None	

	def load_model(self):
		"""
		Downloads the weights and checkpoints from 
		online and saves them locally
		"""

		if os.path.exists("./bin/yolov3.weights") and os.path.exists("./bin/yolov3.cfg"):
			print ("Weights and model config already exist. Proceeding to load YOLO9000Client...")
		else:
			print ("Downloading weights and model config. This may may take a moment...")
			weights_url = "https://pjreddie.com/media/files/yolov3.weights"
			config_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg"
	
			wget.download(weights_url, os.getcwd() + "/yolov3.weights")
			wget.download(config_url, os.getcwd() + "/yolov3.cfg")

			os.mkdir("./bin", 0o755)
			shutil.move("./yolov3.weights", "./bin/yolov3.weights")
			shutil.move("./yolov3.cfg", "./bin/yolov3.cfg")

			print ("\n\nWeights and config file downloaded successfully!")

	def run_single_frame(self, frame):
		"""
		Runs YOLO prediction on a single frame
		"""
		pass

	def get_predictions(self, data, save_data=True, render=True):
		"""
		Returns array of dictionaries of all probable classes
		with bounding boxes and confidence scores.

		Each element in the array is for each input image.
		"""
		for frame in data:
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