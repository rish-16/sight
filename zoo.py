import os
import wget
from termcolor import colored
from clint.textui import progress

import cv2
import tensorflow as tf
import numpy as np

class YOLO9000Client(object):
	def __init__(self):
		self.data = None

	def download_file(self, url, name):
		"""
		Download helper function to download 
		single file from URL
		"""

		r = requests.get(url, stream=True)
		with open(name, 'wb') as f:
			total_length = int(r.headers.get('content-length'))
			print ("Downloading {}...".format(colored(name, 'cyan')))
			for chunk in progress.bar(r.iter_content(chunk_size=8192), expected_size=(total_length/8192) + 1): 
				if chunk:
					f.write(chunk)
					f.flush()

			return local_filename		

	def download_model(self):
		"""
		Downloads the weights and checkpoints from 
		online and saves them locally
		"""
		
		weights_url = "https://pjreddie.com/media/files/yolov3.weights"
		config_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg"

		wget.download(weights_url, os.getcwd() + "/yolov3.weights")
		wget.download(config_url, os.getcwd() + "/yolov3.cfg")

		print ("\n\nWeights and config file downloaded successfully!")

	def load_model(self, weights_path):
		"""
		Search for weights and load into model
		"""
		pass

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