'''
usage: python gen_diff.py -h
'''

from __future__ import print_function

import argparse

from keras.datasets import mnist
from keras.layers import Input
from scipy.misc import imsave

from Model1 import Model1
from Model2 import Model2
from Model3 import Model3
from configs import bcolors
from utils import *

import pickle

random.seed(4172306)


# input image dimensions
img_rows, img_cols = 28, 28
# the data, shuffled and split between train and test sets
(x_train, _), (x_test, _) = mnist.load_data()

x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)

input_shape = (img_rows, img_cols, 1)

x_test = x_test.astype('float32')
x_train = x_train.astype('float32')
x_test /= 255
x_train /= 255

# define input tensor as a placeholder
input_tensor = Input(shape=input_shape)

# load multiple models sharing same input tensor
model1 = Model1(input_tensor=input_tensor)
model2 = Model2(input_tensor=input_tensor)
model3 = Model3(input_tensor=input_tensor)

# init coverage table
model_layer_dict1, model_layer_dict2, model_layer_dict3 = init_coverage_tables(model1, model2, model3)

def update_neuron_bounds(input_data, model, model_layer_dict):
    layer_names = [layer.name for layer in model.layers if
                   'flatten' not in layer.name and 'input' not in layer.name]

    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
    intermediate_layer_outputs = intermediate_layer_model.predict(input_data)

    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        layer = intermediate_layer_output[0]
        for neuron in xrange(num_neurons(layer.shape)): # index through every single (indiv) neuron
            v = layer[np.unravel_index(neuron, layer.shape)]

            if not model_layer_dict[(layer_names[i], neuron)]: # get rid of mean
                model_layer_dict[(layer_names[i], neuron)] = (v, v)
            else:
                (lower,upper) = model_layer_dict[(layer_names[i], neuron)]
                if v > upper:
                    model_layer_dict[(layer_names[i], neuron)] = (lower, v)
                elif v < lower:
                    model_layer_dict[(layer_names[i], neuron)] = (v, upper)

# ==============================================================================================
# start gen inputs
i = 0
training_set = random.sample(x_train, 2000)

for train_img in training_set:
    gen_img = np.expand_dims(train_img, axis=0)
    update_neuron_bounds(gen_img, model1, model_layer_dict1)
    update_neuron_bounds(gen_img, model2, model_layer_dict2)
    update_neuron_bounds(gen_img, model3, model_layer_dict3)
    print(i)
    i += 1
    

pickle.dump(model_layer_dict1, open("m1.p", "wb"))
pickle.dump(model_layer_dict2, open("m2.p", "wb"))
pickle.dump(model_layer_dict3, open("m3.p", "wb"))
