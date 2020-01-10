import json
import glob
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd

class DataAnnotator(object):
	def __init__(self, classes):
		self.classes = classes # array of class labels

	def class_to_int(self, label):
		for i in range(len(self.classes)):
			if self.classes[i] == label:
				return i + 1
			else:
				return None
		
	def __list_to_csv(self.annotations, outfile):
		columns = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
		xml_df = pd.DataFrame(annotations, columns=columns)
		xml_df.to_csv(outfile, index=None)

	def xml_to_csv(self, xmlpath, csvpath):
		annotations = []
		for xml_file in glob.glob(xml_dir + '/*.xml'):
			tree = ET.parse(xml_file)
			root = tree.getroot()
			for member in root.findall('object'):
				value = (root.find('filename').text,
						 int(root.find('size')[0].text), int(root.find('size')[1].text), 
						 member[0].text,
						 int(member[4][0].text), int(member[4][1].text),
						 int(member[4][2].text), int(member[4][3].text))
				annotations.append(value)

		self.__list_to_csv(annotations, csvpath)

	def json_to_csv(self, jsonpath, csvpath):
		with open(jsonpath) as f:
			images = json.load(f)

		annotations = []

		for entry in images:
			filename = images[entry]['filename']
			for region in images[entry]['regions']:
				c = region['region_attributes']['class']
				xmin = region['shape_attributes']['x']
				ymin = region['shape_attributes']['y']
				xmax = xmin + region['shape_attributes']['width']
				ymax = ymin + region['shape_attributes']['height']
				width = 0
				height = 0

				value = (filename, width, height, c, xmin, ymin, xmax, ymax)
				annotations.append(value)

		self.__list_to_csv(annotations, csvpath)