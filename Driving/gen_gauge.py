'''
usage: python gen_diff.py -h
'''

from __future__ import print_function

import argparse

from scipy.misc import imsave

from driving_models import *
from utils import *

import pickle

parser = argparse.ArgumentParser(
    description='Main function for difference-inducing input generation in Driving dataset')
parser.add_argument('model', help="model to generate highs and lows", type=int)

args = parser.parse_args()

random.seed(4172306)

# input image dimensions
img_rows, img_cols = 100, 100
input_shape = (img_rows, img_cols, 3)

# define input tensor as a placeholder
input_tensor = Input(shape=input_shape)

# load multiple models sharing same input tensor
K.set_learning_phase(0)
model1 = Dave_orig(input_tensor=input_tensor, load_weights=True)
model2 = Dave_norminit(input_tensor=input_tensor, load_weights=True)
model3 = Dave_dropout(input_tensor=input_tensor, load_weights=True)
# init coverage table
model_layer_dict1, model_layer_dict2, model_layer_dict3 = init_coverage_tables(model1, model2, model3)

# partition data into a training and testing set
img_paths = image.list_pictures('./testing/center', ext='jpg')
random.shuffle(img_paths)
testing_set = img_paths[:2000]
training_set = img_paths[2000:]

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
# start gen deepgauge bounds
i = 0

if args.model == 1:
    for train_img in training_set:
        gen_img = preprocess_image(train_img)
        update_neuron_bounds(gen_img, model1, model_layer_dict1)
        print(i)
        i += 1
    pickle.dump(model_layer_dict1, open("m1.p", "wb"))

if args.model == 2:
    for train_img in training_set:
        gen_img = preprocess_image(train_img)
        update_neuron_bounds(gen_img, model2, model_layer_dict2)
        print(i)
        i += 1
    pickle.dump(model_layer_dict2, open("m2.p", "wb"))

if args.model == 3:
    for train_img in training_set:
        gen_img = preprocess_image(train_img)
        update_neuron_bounds(gen_img, model3, model_layer_dict3)
        print(i)
        i += 1
    pickle.dump(model_layer_dict3, open("m3.p", "wb"))
    
