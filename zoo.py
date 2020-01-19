import os
import wget
import shutil

import cv2
import tensorflow as tf
import numpy as np

from blocks import Layer

class SightLoader():
	def __init__(self, weights_path):
		with open(weights_path, 'rb') as wf:
			major, = struct.unpack('i', wf.read(4))
			minor, = struct.unpack('i', wf.read(4))
			revision, = struct.unpack('i', wf.read(4))

            if (major*10+ minor) >= 2 and major < 1000 and minor < 1000:
                wf.read(8)
            else:
                wf.read(4)

            transpose = (major > 1000) or (minor > 1000)

            binary = wf.read()

		self.offset = 0
		self.all_weights = np.frombuffer(binary, dtype="float32")

	def read_bytes(self, chunk_size):
		self.offset = self.offset + chunk_size
		return self.all_weights[self.offset - chunk_size:self.offset]

	def load_weights(self, model, verbose=True):
		for i in range(106): # standard darknet layer count
            try:
                conv_layer = model.get_layer("conv_" + str(i))
                if verbose:
                    print ("Loading Convolution #{}".format(i))

                if i not in [81, 93, 105]:
                    norm_layer = model.get_layer("bnorm_" + str(i))

                    size = np.prod(norm_layer.get_weights()[0].shape)

                    beta = self.read_bytes(size)
                    gamma = self.read_bytes(size)
                    mean = self.read_bytes(size)
                    var = self.read_bytes(size)

                    weights = norm_layer.set_weights([gamma, beta, mean, var])

                if len(conv_layer.get_weights()) > 1:
                    bias = self.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
                    kernel = self.read_bytes(np.prod(conv_layer.get_weights()[0].shape))

                    kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
                    kernel = kernel.transpose([2, 3, 1, 0])
                    conv_layer.set_weights([kernel, bias])
                else:
                    kernel = self.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
                    kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
                    kernel = kernel.transpose([2, 3, 1, 0])
                    conv_layer.set_weights([kernel])

            except ValueError:
                if verbose:
                    print ("No Convolution {}".format(i))
                else:
                    pass

        if verbose:
            print ("Finished loading weights into model. Predicting on input data...")

	def reset_offset(self):
		self.offset = 0

class BoundingBox(object):
	def __init__(self, xmin, ymin, xmax, ymax, objness=None, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

        self.objness = objness
        self.classes = classes

        self.label = -1
        self.confidence = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        return self.label

    def get_confidence(self):
        if self.confidence == -1:
            self.confidence = self.classes[self.get_label()]
        return self.confidence

class YOLO9000Client(object):
	def __init__(self):
		self.data = None	
		self.labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
						"boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
						"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
						"backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
						"sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
						"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
						"apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
						"chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
						"remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
						"book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

	def download_model(self):
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

			os.mkdir("./bin", 0o755) # configuring admin rights
			shutil.move("./yolov3.weights", "./bin/yolov3.weights")
			shutil.move("./yolov3.cfg", "./bin/yolov3.cfg")

			print ("\n\nWeights and config file downloaded successfully!")	

	def non_maximum_suppression(bboxes, nms_thresh):
		if len(bboxes) > 0:
			nb_class = len(bboxes[0].classes)
		else:
			return
			
		for c in range(nb_class):
			sorted_indices = np.argsort([-box.classes[c] for box in bboxes])

			for i in range(len(sorted_indices)):
				index_i = sorted_indices[i]

				if bboxes[index_i].classes[c] == 0: continue

				for j in range(i+1, len(sorted_indices)):
					index_j = sorted_indices[j]

					if bbox_iou(bboxes[index_i], bboxes[index_j]) >= nms_thresh:
						bboxes[index_j].classes[c] = 0		

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