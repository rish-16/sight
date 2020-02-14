import struct
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, BatchNormalization, LeakyReLU, add	

class BoundingBox(object):
    def __init__(self, xmin, ymin, xmax, ymax, objectness=None, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

        self.objectness = objectness
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

class SightLoader():
    def __init__(self, weights_path):
        """
        Weights loading framework for all Sight models
        """
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
                    print ("No Convolution #{}".format(i))
                else:
                    pass

        if verbose:
            print ("Finished loading weights into model. Predicting on input data...")

    def reset_offset(self):
        self.offset = 0		

class Layer():
	def get_conv_block(inp, convs, skip=True):
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