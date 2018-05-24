import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


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



        '''checking convolution'''
        img = Image.open('img_18.jpg')
        img = np.array(img)
        print "img shape: " + str(img.shape)
        img = img.reshape((3,450,348))
        self.convolve(img)
        
        a= PoolingLayer((3,446,344))
        a.pool(self.output)

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
                self.output[j][i] = sigmoid(self.z_values[j][i])
                slide += self.stride

                if(self.filter_size+slide) - self.stride >= self.width_input:
                    slide= 0
                    row += self.stride
            

        self.z_values = self.z_values.reshape((self.num_filters, self.height_output, self.width_output))
        self.output = self.output.reshape((self.num_filters, self.height_output, self.width_output))
        for j in xrange(self.num_filters):
            plt.imsave('images/cat_conv%d.jpg'%j, self.z_values[j])
        print "z values shape 3: " +str(self.z_values.shape)
            
            
        #plt.imshow(self.z_values[19])
        #plt.show()
            


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
        self.max_indices = self.max_indices.reshape((self.depth, self.height_output, self.width_output, 2))

def sigmoid(x):
    return 1/1+np.exp(-x)



ConvolutionLayer((3,450,348), 5, 1, 20)

#PoolingLayer((3,450,348))