import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU, Input, Conv2D, ZeroPadding2D, UpSampling2D, BatchNormalization, add, concatenate
from tensorflow.keras.models import Model
import struct
import cv2
import matplotlib.pyplot as plt

class WeightsLoader():
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

    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset - size:self.offset]

    def load_weights(self, model, verbose=True):
        for i in range(106): # standard darknet layer count
            try:
                conv_layer = model.get_layer("conv_" + str(i))
                if verbose:
                    print ("Loading weights for convolution #{}".format(i))

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
                    print ("No convolution #{}".format(i))
                else:
                    pass

        if verbose:
            print ("Finished loading weights into model. Predicting on input data...")
    
    def reset(self):
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
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
        return self.score

def ConvBlock(inp, convs, skip=True):
    x = inp
    count = 0

    for conv in convs:
        if count == (len(convs) - 2) and skip:
            skip_conn = x
        count += 1

        if conv['stride'] > 1: x = ZeroPadding2D(((1,0),(1,0)))(x)
        x = Conv2D(conv['filter'], conv['kernel'], strides=conv['stride'], padding="valid" if conv['stride']>1 else "same", name="conv_"+str(conv['layer_idx']), use_bias=False if conv['bnorm'] else True)(x)
        if conv['bnorm']: x = BatchNormalization(epsilon=0.001, name="bnorm_"+str(conv['layer_idx']))(x)
        if conv['leaky']: x = LeakyReLU(alpha=0.1, name="leaky_"+str(conv['layer_idx']))(x)

    return add([skip_conn, x]) if skip else x

def interval_overlap(int_a, int_b):
    x1, x2 = int_a
    x3, x4 = int_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3
        
def sigmoid(z):
    return 1. / (1. + np.exp(-z))

def bbox_iou(box1, box2):
    int_w = interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    int_h = interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])

    intersect = int_w * int_h

    w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
    w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin

    union = w1*h1 + w2*h2 - intersect

    return float(intersect) / union

def YOLO9000():
    inp_img = Input(shape=[None, None, 3])

    x = ConvBlock(inp_img, [{'filter': 32, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 0},
                                    {'filter': 64, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 1},
                                    {'filter': 32, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 2},
                                    {'filter': 64, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 3}])

    x = ConvBlock(x, [{'filter': 128, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 5},
                        {'filter':  64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 6},
                        {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 7}])

    x = ConvBlock(x, [{'filter':  64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 9},
                        {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 10}])

    x = ConvBlock(x, [{'filter': 256, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 12},
                        {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 13},
                        {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 14}])

    for i in range(7):
        x = ConvBlock(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 16+i*3},
                            {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 17+i*3}])
        
    skip_36 = x
        
    x = ConvBlock(x, [{'filter': 512, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 37},
                        {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 38},
                        {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 39}])

    for i in range(7):
        x = ConvBlock(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 41+i*3},
                            {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 42+i*3}])
        
    skip_61 = x
        
    x = ConvBlock(x, [{'filter': 1024, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 62},
                        {'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 63},
                        {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 64}])

    for i in range(3):
        x = ConvBlock(x, [{'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 66+i*3},
                            {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 67+i*3}])
        
    x = ConvBlock(x, [{'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 75},
                        {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 76},
                        {'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 77},
                        {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 78},
                        {'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 79}], skip=False)

    yolo_82 = ConvBlock(x, [{'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 80},
                                {'filter':  255, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'layer_idx': 81}], skip=False)

    x = ConvBlock(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 84}], skip=False)
    x = UpSampling2D(2)(x)
    x = concatenate([x, skip_61])

    x = ConvBlock(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 87},
                        {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 88},
                        {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 89},
                        {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 90},
                        {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 91}], skip=False)

    yolo_94 = ConvBlock(x, [{'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 92},
                                {'filter': 255, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'layer_idx': 93}], skip=False)

    x = ConvBlock(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True,   'layer_idx': 96}], skip=False)
    x = UpSampling2D(2)(x)
    x = concatenate([x, skip_36])

    yolo_106 = ConvBlock(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 99},
                                {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 100},
                                {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 101},
                                {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 102},
                                {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 103},
                                {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 104},
                                {'filter': 255, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'layer_idx': 105}], skip=False)

    model = Model(inp_img, [yolo_82, yolo_94, yolo_106])    
    return model

def preprocess(img, net_w, net_h):
    new_h, new_w, _ = img.shape

    # determine the new size of the image
    # if (float(net_w)/new_w) < (float(net_h)/new_h):
    #     new_h = (new_h * net_w)/new_w
    #     new_w = net_w
    # else:
    #     new_w = (new_w * net_h)/new_h
    #     new_h = net_h

    if (float(net_w)/new_w) < (float(net_h)/new_h):
        new_h = (new_h * net_w)//new_w
        new_w = net_w
    else:
        new_w = (new_w * net_h)//new_h
        new_h = net_h        

    # resize the image to the new size
    resized = cv2.resize(img[:, :, ::-1]/255., (int(new_w), int(new_h)))

    # embed the image into the standard letter box
    new_img = np.ones((net_h, net_w, 3)) * 0.5
    new_img[int((net_h-new_h)//2):int((net_h+new_h)//2), int((net_w-new_w)//2):int((net_w+new_w)//2), :] = resized
    new_img = np.expand_dims(new_img, 0)

    return new_img

def decode_netout(netout, anchors, obj_thresh, nms_thresh, neth, netw):
    gridh, gridw = netout.shape[:2]
    nb_box = 3
    netout = netout.reshape([gridh, gridw, nb_box, -1])
    nb_class = netout.shape[-1] - 5

    boxes = []
    
    netout[..., :2]  = sigmoid(netout[..., :2])
    netout[..., 4:]  = sigmoid(netout[..., 4:])
    netout[..., 5:]  = netout[..., 4][..., np.newaxis] * netout[..., 5:]
    netout[..., 5:] *= netout[..., 5:] > obj_thresh

    for i in range(gridh * gridw):
        row = i / gridw
        col = i % gridw

        for b in range(nb_box):
            objectness = netout[int(row)][int(col)][b][4]

            if(objectness.all() <= obj_thresh): continue

            x, y, w, h = netout[int(row)][int(col)][b][:4]

            x = (col + x) / gridw 
            y = (row + y) / gridh 
            w = anchors[2 * b + 0] * np.exp(w) / netw
            h = anchors[2 * b + 1] * np.exp(h) / neth

            classes = netout[int(row)][col][b][5:]

            box = BoundingBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)

            boxes.append(box)    

    return boxes

def rectify_yolo_boxes(boxes, img_h, img_w, neth, netw):
    if (float(net_w)/img_w) < (float(net_h)/img_h):
        neww = netw
        newh = (img_h * netw)/ img_w
    else:
        newh = netw
        neww = (img_w * neth) / img_h
        
    for i in range(len(boxes)):
        x_offset, x_scale = (netw - neww)/2./netw, float(neww)/netw
        y_offset, y_scale = (neth - newh)/2./neth, float(newh)/neth
        
        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * img_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * img_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * img_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * img_h)
                
def non_maximum_suppresion(boxes, nms_thresh):
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return
        
    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            if boxes[index_i].classes[c] == 0: continue

            for j in range(i+1, len(sorted_indices)):
                index_j = sorted_indices[j]

                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0

def render_boxes(img, boxes, labels, obj_thresh):
    for box in boxes:
        label_str = ""
        label = -1

        for i in range(len(labels)):
            if box.classes[i] > obj_thresh:
                label_str += labels[i]
                label = i
                print ("{}: {:.4f}%".format(labels[i], box.classes[i]*100))

        if label >= 0:
            cv2.rectangle(img, (box.xmin, box.ymin), (box.xmax, box.ymax), (0, 255, 3), 3)
            cv2.putText(img, '{} {:.3f}'.format(label_str, box.get_score()), (box.xmax, box.ymin - 13), cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * img.shape[0], (0, 255, 0), 2)

    return img

def write_img(image, image_path):
    image_path = image_path.split('/')
    img_name = image_path[-1]
    img_name = img_name.split('.')
    img_name = img_name[0] + "_detected." + img_name[1]
    image_path = "/".join(image_path[:-1]) + "/" + img_name

    cv2.imwrite(image_path, (image).astype('uint8'))

weights_path = "./bin/yolov3.weights"
image_path   = "./test_data/img/fruits.jpg"

net_h, net_w = 416, 416
obj_thresh, nms_thresh = 0.5, 0.45
anchors = [[116, 90, 156, 198, 373, 326], [30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]]
labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", 
            "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", 
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", 
            "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", 
            "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", 
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", 
            "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", 
            "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", 
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", 
            "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

yolov3 = YOLO9000()

weight_loader = WeightsLoader(weights_path)
weight_loader.load_weights(yolov3)

img = cv2.imread(image_path)
img_h, img_w = img.shape[:2]
new_img = preprocess(img, net_h, net_w)

yolos = yolov3.predict(new_img)
boxes = []

for i in range(len(yolos)):
    boxes += decode_netout(yolos[i][0], anchors[i], obj_thresh, nms_thresh, net_h, net_w)

rectify_yolo_boxes(boxes, img_h, img_w, net_h, net_w)    
non_maximum_suppresion(boxes, nms_thresh)

bbox_img = render_boxes(img, boxes, labels, obj_thresh)
plt.imshow(bbox_img)
plt.show()

write_img(bbox_img, image_path)