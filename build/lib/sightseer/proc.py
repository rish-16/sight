import io
import json
import glob
import os
from PIL import Image
import xml.etree.ElementTree as ET
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd

class DataAnnotator(object):
	def __init__(self, classes):
		self.classes = classes # array of class labels
		
	def list_to_csv(self, annotations, outfile):
		columns = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
		xml_df = pd.DataFrame(annotations, columns=columns)
		xml_df.to_csv(outfile, index=None)

	def class_to_int(self, class_label):
		for i in range(len(self.classes)):
			if self.classes[i] == class_label:
				return i + 1
			else:
				return None

	def xml_to_csv(self, xml_path, csv_path):
		annotations = []
		for xml_file in glob.glob(xml_path + '*.xml'):
			tree = ET.parse(xml_file)
			root = tree.getroot()
			for member in root.findall('object'):
				value = (root.find('filename').text,
						 int(root.find('size')[0].text), int(root.find('size')[1].text), 
						 member[0].text,
						 int(member[4][0].text), int(member[4][1].text),
						 int(member[4][2].text), int(member[4][3].text))
				annotations.append(value)

		self.list_to_csv(annotations, csv_path)

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

		self.list_to_csv(annotations, csvpath)

	def generate_tfexample(self, group, path):
		with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
			encoded_jpg = fid.read()
		encoded_jpg_io = io.BytesIO(encoded_jpg)
		image = Image.open(encoded_jpg_io)
		width, height = image.size

		filename = group.filename.encode('utf8')
		image_format = b'jpg'
		xmins = []
		xmaxs = []
		ymins = []
		ymaxs = []
		classes_text = []
		classes = []

		for index, row in group.object.iterrows():
			xmins.append(row['xmin'] / width)
			xmaxs.append(row['xmax'] / width)
			ymins.append(row['ymin'] / height)
			ymaxs.append(row['ymax'] / height)
			classes_text.append(row['class'].encode('utf8'))
			classes.append(self.class_to_int(row['class']))

		tf_example = tf.train.Example(features=tf.train.Features(feature={
			'image/height': dataset_util.int64_feature(height),
			'image/width': dataset_util.int64_feature(width),
			'image/filename': dataset_util.bytes_feature(filename),
			'image/source_id': dataset_util.bytes_feature(filename),
			'image/encoded': dataset_util.bytes_feature(encoded_jpg),
			'image/format': dataset_util.bytes_feature(image_format),
			'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
			'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
			'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
			'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
			'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
			'image/object/class/label': dataset_util.int64_list_feature(classes),
		}))
		
		return tf_example		

	def csv_to_tfrecord(self, csvpath, filename, tfrpath):
		csv = pd.read_csv(csvpath).values
