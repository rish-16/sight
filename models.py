import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Dropout, LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import numpy as np

class YOLO9000(object):
	def build_model(self):
		model = Sequential()
		model.add(Conv2D(32, (3,3), stride=(2,2), input_size=(240, 160, 3), activation=LeakyReLU(alpha=0.3)))
		model.add(MaxPool2D(pool_size=(2,2), stride=(2,2)))
		model.add(Conv2D(64, (3,3), activation=LeakyReLU(alpha=0.3)))
		model.add(MaxPool2D(pool_size=(2,2), stride=(2,2)))
		model.add(Conv2D())