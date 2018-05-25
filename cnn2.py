import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cPickle
import time 
from os import walk
from FDDB_loader import load_dataset
import random
from backprop2 import *

class ConvolutionLayer(object):
    def __init__(self, input_shape, filter_size, stride, num_filters, padding = 0):
        self.depth, self.height_input, self.width_input = input_shape
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.num_filters = num_filters

        self.weights = np.random.randn(self.num_filters, self.depth, self.filter_size, self.filter_size)
        self.biases = np.random.randn(self.num_filters,1)

        self.height_output = (self.height_input - self.filter_size + 2*self.padding)/self.stride + 1 
        self.width_output = (self.width_input - self.filter_size + 2*self.padding)/self.stride + 1 

        self.z_values = np.zeros((self.num_filters,self.height_output, self.width_output))
        print "z values shape: " + str(self.z_values.shape)
        self.output = np.zeros((self.num_filters, self.height_output, self.width_output))
        print "output shape: " + str(self.output.shape)


    def convolve(self, input_data):
        act_length1d = self.height_output * self.width_output
        self.z_values = self.z_values.reshape((self.num_filters, act_length1d))
        print "z values shape 2: " + str(self.z_values.shape)
        self.output = self.output.reshape((self.num_filters, act_length1d))
        print "output shape 2: " + str(self.output.shape)

        
        print "act_length1d: " + str(act_length1d)

        for j in range(self.num_filters):
            slide = 0
            row = 0
            print "filter number %d" % j
            for i in range(act_length1d):
                self.z_values[j][i] = np.sum(input_data[:,row:self.filter_size+row, slide:self.filter_size+slide] * self.weights[j]) + self.biases[j]
                self.output[j][i] = relu(self.z_values[j][i])
                slide += self.stride

                if(self.filter_size+slide) - self.stride >= self.width_input:
                    slide= 0
                    row += self.stride
            

        self.z_values = self.z_values.reshape((self.num_filters, self.height_output, self.width_output))
        self.output = self.output.reshape((self.num_filters, self.height_output, self.width_output))
        print "z values shape 3: " +str(self.z_values.shape)
            
            
        #plt.imshow(self.z_values[19])
        #plt.show()
            
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f, -1)
        logger.info('save model to path %s' % path)
        return None

class PoolingLayer(object):

    def __init__(self, input_shape, pool_size = (2,2)):
        self.depth, self.height_input, self.width_input = input_shape
        self.pool_size = pool_size
        self.height_output = (self.height_input - self.pool_size[0]) / self.pool_size[1] + 1
        self.width_output = (self.width_input - self.pool_size[0])/self.pool_size[1] + 1

        print "height output: " + str(self.height_output)
        print "width output: " + str(self.width_output)

        self.output = np.zeros((self.depth, self.height_output, self.width_output))
        print "output shape pooling: " + str(self.output.shape)
        self.max_indices = np.zeros((self.depth, self.height_output, self.width_output,2))

        
    
    def pool(self,input_image):
        pool_length1d = self.height_output * self.width_output

        self.output = self.output.reshape((self.depth, pool_length1d))
        print "output shape pooling 2: " + str(self.output.shape)
        self.max_indices = self.max_indices.reshape((self.depth, pool_length1d, 2))
        
        # for each filter map
        for j in range(self.depth):
            row = 0
            slide = 0
            
            for i in range(pool_length1d):
                toPool = input_image[j][row:self.pool_size[0] + row, slide:self.pool_size[0] + slide]

                self.output[j][i] = np.amax(toPool)                # calculate the max activation
                index = zip(*np.where(np.max(toPool) == toPool))           # save the index of the max
                if len(index) > 1:
                    index = [index[0]]
                index = index[0][0]+ row, index[0][1] + slide
                self.max_indices[j][i] = index

                slide += self.pool_size[1]

                # modify this if stride != filter for poolsize 
                if slide >= self.width_input:
                    slide = 0
                    row += self.pool_size[1]

        self.output = self.output.reshape((self.depth, self.height_output, self.width_output))
        print self.output.shape
        self.max_indices = self.max_indices.reshape((self.depth, self.height_output, self.width_output, 2))


class FullyConnectedLayer(object):
    def __init__(self, input_shape, num_output):
        self.output = np.zeros((num_output,1))
        self.z_values = np.zeros((num_output,1))
        self.depth, self.height_input, self.width_input = input_shape
        self.num_output = num_output

        self.weights = np.random.randn(self.num_output, self.depth,self.height_input,self.width_input)
        self.biases = np.random.randn(self.num_output,1)


    def feedforward(self,x):
        self.weights= np.reshape(self.weights,((self.num_output, self.depth*self.height_input*self.width_input)))
        x = np.reshape(x, (self.depth*self.height_input*self.width_input,1))
        self.z_values = np.dot(self.weights,x) + self.biases
        self.output = tanh(self.z_values)
        self.weights = self.weights.reshape((self.num_output,self.depth,self.height_input,self.width_input))



class ClassifyLayer(object):
    def __init__(self,num_inputs, num_classes):
        self.output= np.zeros((num_classes,1))
        self.z_values = np.zeros((num_classes,1))
        self.num_classes = num_classes
        num_inputs, cols = num_inputs
        self.weights = np.random.randn(self.num_classes,num_inputs)
        self.biases = np.random.randn(self.num_classes,1)

    def classify(self, x):
        self.z_values = np.dot(self.weights,x) + self.biases
        self.output = tanh(self.z_values)
        print "guess is : " +str(self.output)



class Network(object):
    
    layer_type_map = {
        'fc_layer': FullyConnectedLayer,
        'final_layer': ClassifyLayer,
        'conv_layer': ConvolutionLayer,
        'pool_layer': PoolingLayer
    }

    def __init__(self,input_shape, layer_config):
        '''
        :param layer_config: list of dicts, outer key is 
        Valid Layer Types:
        Convolutional Layer: shape of input, filter_size, stride, padding, num_filters
        Pooling Layer: shape of input(depth, height_in, width_in), poolsize
        Fully Connected Layer: shape_of_input, num_output, classify = True/False, num_classes (if classify True)
        Gradient Descent: training data, batch_size, eta, num_epochs, lambda, test_data
        '''

        self.input_shape = input_shape
        self.initialize_layers(layer_config)
        self.layer_weight_shapes = [l.weights.shape for l in self.layers if not isinstance(l,PoolingLayer)]
        self.layer_biases_shapes = [l.biases.shape for l in self.layers if not isinstance(l,PoolingLayer)]


    def initialize_layers(self,layer_config):
        """
        Sets the net's <layer> attribute
        to be a list of Layers (classes from layer_type_map)
        """
        layers = []
        input_shape = self.input_shape
        for layer_spec in layer_config:
            # handle the spec format: {'type': {kwargs}}
            layer_class = self.layer_type_map[layer_spec.keys()[0]]
            layer_kwargs = layer_spec.values()[0]
            layer = layer_class(input_shape, **layer_kwargs)
            input_shape = layer.output.shape
            layers.append(layer)
        self.layers = layers


    def _get_layer_transition(self, inner_ix, outer_ix):
        inner, outer = self.layers[inner_ix], self.layers[outer_ix]
        # either input to FC or pool to FC -> going from 3d matrix to 1d
        if (
            (inner_ix < 0 or isinstance(inner, PoolingLayer)) and 
            isinstance(outer, FullyConnectedLayer)
            ):
            return '3d_to_1d'
        # going from 3d to 3d matrix -> either input to conv or conv to conv
        if (
            (inner_ix < 0 or isinstance(inner, ConvolutionLayer)) and 
            isinstance(outer, ConvolutionLayer)
            ):
            return 'to_conv'
        if (
            isinstance(inner, FullyConnectedLayer) and
            (isinstance(outer, ClassifyLayer) or isinstance(outer, FullyConnectedLayer))
            ):
            return '1d_to_1d'
        if (
            isinstance(inner,ConvolutionLayer) and
            isinstance(outer, PoolingLayer)
            ):
            return 'conv_to_pool'

    def feedforward(self, image):
        prev_activation = image

        #forward prop
        for layer in self.layers:
            input_to_feed = prev_activation

            if isinstance(layer,ConvolutionLayer):
                layer.convolve(input_to_feed)
                for i in xrange(layer.output.shape[0]):
                    plt.imsave('images/conv_pic%d.jpg'%i, layer.output[i])
            
            elif isinstance(layer, PoolingLayer):
                layer.pool(input_to_feed)
                for i in xrange(layer.output.shape[0]):
                    plt.imsave('images/pool_pic%d.jpg'%i,layer.output[i])

            elif isinstance(layer,FullyConnectedLayer):
                layer.feedforward(input_to_feed)

            elif isinstance(layer,ClassifyLayer):
                layer.classify(input_to_feed)

            else:
                raise NotImplementedError

            prev_activation = layer.output
        final_activation = prev_activation
        return final_activation

    def backprop(self, image, label):
        nabla_w = [np.zeros(s) for s in self.layer_weight_shapes]
        nabla_b = [np.zeros(s) for s in self.layer_biases_shapes]

        # set first params on the final layer
        final_output = self.layers[-1].output
        last_delta = (label - final_output) * dtanh(self.layers[-1].z_values)
        last_weights = None
        final=True

        num_layers = len(self.layers)
        # import ipdb;ipdb.set_trace()

        for l in xrange(num_layers - 1, -1, -1):
            # the "outer" layer is closer to classification
            # the "inner" layer is closer to input
            inner_layer_ix = l - 1
            if (l-1) <0:
                inner_layer_ix = 0
            outer_layer_ix = l

            layer = self.layers[outer_layer_ix]
            activation = self.layers[inner_layer_ix].output if inner_layer_ix >= 0 else image

            transition = self._get_layer_transition(
                inner_layer_ix, outer_layer_ix
            )

            # inputfc = poolfc
            # fc to fc = fc to final
            # conv to conv -> input to conv
            # conv to pool -> unique

            if transition == '1d_to_1d':   # final to fc, fc to fc
                db, dw, last_delta = backprop_1d_to_1d(
                    delta = last_delta,
                    prev_weights=last_weights,
                    prev_activations=activation,
                    z_vals=layer.z_values,
                    final=final)
                final = False

            elif transition == '3d_to_1d':
                if l==0:
                    activation = image
                # calc delta on the first final layer
                db, dw, last_delta = backprop_1d_to_3d(
                    delta=last_delta,
                    prev_weights=last_weights,    # shape (10,100) this is the weights from the next layer
                    prev_activations=activation,  #(28,28)
                    z_vals=layer.z_values)    # (100,1)
                # layer.weights = layer.weights.reshape((layer.num_output, layer.depth, layer.height_in, layer.width_in))

            # pool to conv layer
            elif transition == 'conv_to_pool':
                # no update for dw,db => only backprops the error            
                last_delta = backprop_pool_to_conv(
                    delta = last_delta,
                    prev_weights = last_weights,
                    input_from_conv = activation,
                    max_indices = layer.max_indices,
                    poolsize = layer.pool_size,
                    pool_output = layer.output)

            # conv to conv layer
            elif transition == 'to_conv':
                # weights passed in are the ones between conv to conv
                # update the weights and biases
                activation = image
                last_weights = layer.weights
                db,dw = backprop_to_conv(
                    delta = last_delta,
                    weight_filters = last_weights,
                    stride = layer.stride,
                    input_to_conv = activation,
                    prev_z_vals = layer.z_values)
            else:
                pass

            if transition != 'conv_to_pool':
                # print 'nablasb, db,nabldw, dw, DELTA', nabla_b[inner_layer_ix].shape, db.shape, nabla_w[inner_layer_ix].shape, dw.shape, last_delta.shape
                nabla_b[inner_layer_ix], nabla_w[inner_layer_ix] = db, dw
                last_weights = layer.weights

        return self.layers[-1].output, nabla_b, nabla_w

    def training(self, training_data , batch_size, learning_rate, num_epochs):
        
        training_size = len(training_data)

        mean_error = []
        correct_res = []
        
        for epoch in xrange(num_epochs):
            print "Starting epochs %d" % epoch
            start = time.time()
            random.shuffle(training_data)
            batches = [training_data[k:k + batch_size] for k in xrange(0, training_size, batch_size)]
            losses = 0
            counter = 0
            for batch in batches:
                print "batch %d" % counter
                counter+=1
                loss = self.update_mini_batch(batch, learning_rate)
                losses+=loss
            mean_error.append(round(losses/batch_size,2))
            print mean_error
        
    def update_mini_batch(self, batch, learning_rate):
        nabla_w = [np.zeros(s) for s in self.layer_weight_shapes]
        nabla_b = [np.zeros(s) for s in self.layer_biases_shapes]

        batch_size = len(batch)

        for image in batch:
            image = image.reshape((1,image.shape[1],image.shape[2]))
            _ = self.feedforward(image)
            final_res, delta_b, delta_w = self.backprop(image, 1)

            nabla_b = [nb + db for nb, db in zip(nabla_b, delta_b)]
            nabla_w = [nw + dw for nw, dw in zip(nabla_w, delta_w)]

        ################## print LOSS ############
        error = loss(label, final_res)
        print "error %f" %error
        num =0

        weight_index = []
        for layer in self.layers:
            if not isinstance(layer,PoolingLayer):
                weight_index.append(num)
            num+=1

        for ix, (layer_nabla_w, layer_nabla_b) in enumerate(zip(nabla_w, nabla_b)):
            layer = self.layers[weight_index[ix]]
            layer.weights -= learning_rate * layer_nabla_w / batch_size
            layer.biases -= learning_rate * layer_nabla_b / batch_size
        return error
'''
    def validate(self,data):
        data = [(im.reshape((1,28,28)),y) for im,y in data]
        test_results = [(np.argmax(self.feedforward(x)),y) for x, y in data]
        return sum(int(x == y) for x, y in test_results) 
'''

def tanh(x):
    return np.tanh(x)

def dtanh(x):
    return 1. - x * x

def relu(x):
    return x*(x>0)

def dReLU(x):
    return 1. * (x > 0)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def save(model,filename):
    with open(filename, 'wb') as fid:
        cPickle.dump(model, fid)  

def load(filename):
    with open(filename, 'rb') as fid:
        model_loaded = cPickle.load(fid)
    return model_loaded



img = Image.open('img_18.jpg')
img = np.array(img)
gray = rgb2gray(img)
gray= gray.reshape((1,gray.shape[0],gray.shape[1]))
input_shape = gray.shape
net = Network(input_shape,
            layer_config = [
                {'conv_layer': {
                    'filter_size' : 5,
                    'stride' : 1,
                    'num_filters' : 20}},
                {'pool_layer': {
                    'pool_size' : (2,2)}},
                {'fc_layer': {
                    'num_output' : 30}},
                {'final_layer': {
                    'num_classes' : 1}}
            ])


training_data = load_dataset()
training_data1 = []
print training_data1
for face in xrange(len(training_data)):
    img = training_data[face][0]
    try:
        img = img.reshape((500,500,3))
        gray = rgb2gray(img)
        gray = gray.reshape((1,500,500))
        training_data1.append(gray)
    except:
        continue
net.training(training_data1,10,0.1,1)
net.save()