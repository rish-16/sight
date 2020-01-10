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

	def xml_to_csv(self, xmlpath):
		xml_list = []
		for xml_file in glob.glob(path + '/*.xml'):
			tree = ET.parse(xml_file)
			root = tree.getroot()
			for member in root.findall('object'):
				value = (root.find('filename').text,
							int(root.find('size')[0].text),
							int(root.find('size')[1].text),
							member[0].text,
							int(member[4][0].text),
							int(member[4][1].text),
							int(member[4][2].text),
							int(member[4][3].text)
							)
				xml_list.append(value)
		column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'] # usual 
		xml_df = pd.DataFrame(xml_list, columns=column_name)
		return xml_df

	def xml_to_tfrecord(self, data):
		pass