import os
import wget
import struct
import shutil

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, UpSampling2D, concatenate
from tensorflow.keras.models import Model

from .blocks import ConvBlock, BoundingBox, SightLoader

class YOLO9000Client(object):
	def __init__(self):
		self.data = None	
		self.all_labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
						"boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
						"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
						"backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
						"sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
						"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
						"apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
						"chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
						"remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
						"book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
		self.nms_threshold = 0.45
		self.obj_threshold = 0.5
		self.net_h, self.net_w = 416, 416
		self.bbox_anchors = [[116, 90, 156, 198, 373, 326], [30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]]
		self.yolo_model = None

	def download_weights(self):
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

	def load_architecture(self):
		"""
		Returns tf.keras.models.Model instance
		"""
		inp_img = Input(shape=[None, None, 3])

		x = ConvBlock.get_conv_block(inp_img, [{'filter': 32, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 0},
										{'filter': 64, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 1},
										{'filter': 32, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 2},
										{'filter': 64, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 3}])

		x = ConvBlock.get_conv_block(x, [{'filter': 128, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 5},
							{'filter':  64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 6},
							{'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 7}])

		x = ConvBlock.get_conv_block(x, [{'filter':  64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 9},
							{'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 10}])

		x = ConvBlock.get_conv_block(x, [{'filter': 256, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 12},
							{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 13},
							{'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 14}])

		for i in range(7):
			x = ConvBlock.get_conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 16+i*3},
								{'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 17+i*3}])
			
		skip_36 = x
			
		x = ConvBlock.get_conv_block(x, [{'filter': 512, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 37},
							{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 38},
							{'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 39}])

		for i in range(7):
			x = ConvBlock.get_conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 41+i*3},
								{'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 42+i*3}])
			
		skip_61 = x
			
		x = ConvBlock.get_conv_block(x, [{'filter': 1024, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 62},
							{'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 63},
							{'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 64}])

		for i in range(3):
			x = ConvBlock.get_conv_block(x, [{'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 66+i*3},
								{'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 67+i*3}])
			
		x = ConvBlock.get_conv_block(x, [{'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 75},
							{'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 76},
							{'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 77},
							{'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 78},
							{'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 79}], skip=False)

		yolo_82 = ConvBlock.get_conv_block(x, [{'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 80},
									{'filter':  255, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'layer_idx': 81}], skip=False)

		x = ConvBlock.get_conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 84}], skip=False)
		x = UpSampling2D(2)(x)
		x = concatenate([x, skip_61])

		x = ConvBlock.get_conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 87},
							{'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 88},
							{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 89},
							{'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 90},
							{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 91}], skip=False)

		yolo_94 = ConvBlock.get_conv_block(x, [{'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 92},
									{'filter': 255, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'layer_idx': 93}], skip=False)

		x = ConvBlock.get_conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True,   'layer_idx': 96}], skip=False)
		x = UpSampling2D(2)(x)
		x = concatenate([x, skip_36])

		yolo_106 = ConvBlock.get_conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 99},
									{'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 100},
									{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 101},
									{'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 102},
									{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 103},
									{'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 104},
									{'filter': 255, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'layer_idx': 105}], skip=False)

		model = Model(inp_img, [yolo_82, yolo_94, yolo_106])    
		return model

	def non_maximum_suppression(bboxes):
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

					if bbox_iou(bboxes[index_i], bboxes[index_j]) >= self.nms_threshold:
						bboxes[index_j].classes[c] = 0

	def 

	def load_model(self, default_path="./bin/yolov3.weights"):
		"""
		Downloads weights and config, loads checkpoints into architecture
		"""
		self.download_weights() # downloading weights from online
		loader = SightLoader(default_path)
		
		self.yolo_model = self.load_architecture() # loading weights into model
		loader.load_weights(yolo_model)

	def get_predictions(self, image):
		img_h, img_w = image.shape[:2]

		preds = self.yolo_model.predict(image)
		boxes = []

		for i in range(len(preds)):
			boxes += self.decode_output(preds[i][0])

		self.rectify_bboxes(boxes)
		self.non_maximum_suppression(boxes)