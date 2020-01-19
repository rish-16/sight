import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU, Input, Conv2D, ZeroPadding2D, UpSampling2D, BatchNormalization, add, concatenate
from tensorflow.keras.models import Model
import struct

class Layer():
	def get_block(inp, convs, skip=True):
		x = inp
		count = 0

		for conv in convs:
			if count == (len(convs) - 2) and skip:
				skip_conn = x
			count += 1

			if conv['stride'] > 1: x = ZeroPadding2D(((1,0),(1,0)))(x)
			
			x = Conv2D(conv['filter'], 
					conv['kernel'], 
					strides=conv['stride'], 
					padding="valid" if conv['stride']>1 else "same", 
					name="conv_"+str(conv['layer_idx']), 
					use_bias=False if conv['bnorm'] else True)(x)
			
			if conv['bnorm']: x = BatchNormalization(epsilon=0.001, name="bnorm_"+str(conv['layer_idx']))(x)
			
			if conv['leaky']: x = LeakyReLU(alpha=0.1, name="leaky_"+str(conv['layer_idx']))(x)

		return add([skip_conn, x]) if skip else x		