'''
usage: python gen_diff.py -h
'''

from __future__ import print_function

import argparse

from mimicus.tools.featureedit import FeatureDescriptor
from scipy.misc import imsave

from pdf_models import *
from utils import *

import pickle

parser = argparse.ArgumentParser(
    description='Main function for difference-inducing input generation in Driving dataset')
parser.add_argument('model', help="model to generate highs and lows", type=int)

args = parser.parse_args()

random.seed(4172306)

# load data
X_train, _, names = datasets.csv2numpy('./dataset/train.csv')
X_train = X_train.astype('float32')
num_features = X_train.shape[1]
feat_names = FeatureDescriptor.get_feature_names()
incre_idx, incre_decre_idx = init_feature_constraints(feat_names)

# define input tensor as a placeholder
input_tensor = Input(shape=(num_features,))

# load multiple models sharing same input tensor
K.set_learning_phase(0)
model1 = Model1(input_tensor=input_tensor, load_weights=True)
model2 = Model2(input_tensor=input_tensor, load_weights=True)
model3 = Model3(input_tensor=input_tensor, load_weights=True)

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
# start gen deepgauge bounds
i = 0

if args.model == 1:
    for pdf in X_train:
        gen_pdf = np.expand_dims(pdf, axis=0)
        update_neuron_bounds(gen_pdf, model1, model_layer_dict1)
        print(i)
        i += 1
    pickle.dump(model_layer_dict1, open("m1.p", "wb"))

if args.model == 2:
    for pdf in X_train:
        gen_pdf = np.expand_dims(pdf, axis=0)
        update_neuron_bounds(gen_pdf, model2, model_layer_dict2)
        print(i)
        i += 1
    pickle.dump(model_layer_dict2, open("m2.p", "wb"))

if args.model == 3:
    for pdf in X_train:
        gen_pdf = np.expand_dims(pdf, axis=0)
        update_neuron_bounds(gen_pdf, model3, model_layer_dict3)
        print(i)
        i += 1
    pickle.dump(model_layer_dict3, open("m3.p", "wb"))
    
