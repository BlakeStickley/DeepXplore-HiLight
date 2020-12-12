import random
from collections import defaultdict

import numpy as np
from keras import backend as K
from keras.models import Model


# util function to convert a tensor into a valid image
def deprocess_image(x):
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x.reshape(x.shape[1], x.shape[2])  # original shape (1,img_rows, img_cols,1)


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def constraint_occl(gradients, start_point, rect_shape):
    new_grads = np.zeros_like(gradients)
    new_grads[:, start_point[0]:start_point[0] + rect_shape[0],
    start_point[1]:start_point[1] + rect_shape[1]] = gradients[:, start_point[0]:start_point[0] + rect_shape[0],
                                                     start_point[1]:start_point[1] + rect_shape[1]]
    return new_grads


def constraint_light(gradients):
    new_grads = np.ones_like(gradients)
    grad_mean = np.mean(gradients)
    return grad_mean * new_grads


def constraint_black(gradients, rect_shape=(6, 6)):
    start_point = (
        random.randint(0, gradients.shape[1] - rect_shape[0]), random.randint(0, gradients.shape[2] - rect_shape[1]))
    new_grads = np.zeros_like(gradients)
    patch = gradients[:, start_point[0]:start_point[0] + rect_shape[0], start_point[1]:start_point[1] + rect_shape[1]]
    if np.mean(patch) < 0:
        new_grads[:, start_point[0]:start_point[0] + rect_shape[0],
        start_point[1]:start_point[1] + rect_shape[1]] = -np.ones_like(patch)
    return new_grads

def constraint_black_mod(gradients, iters, rect_shape=(2, 2)):
    if iters < 10:
    	start_point = (random.randint(6, 18), random.randint(0, gradients.shape[2] - rect_shape[1]))
    else:
	start_point = (
        	random.randint(0, gradients.shape[1] - rect_shape[0]), random.randint(0, gradients.shape[2] - rect_shape[1]))
    new_grads = np.zeros_like(gradients)
    patch = gradients[:, start_point[0]:start_point[0] + rect_shape[0], start_point[1]:start_point[1] + rect_shape[1]]
    if np.mean(patch) < 0:
        new_grads[:, start_point[0]:start_point[0] + rect_shape[0],
        start_point[1]:start_point[1] + rect_shape[1]] = -np.ones_like(patch)
    return new_grads

def constraint_black_hilight(gradients, image, rect_shape=(8, 8)):
    # calculate the start point to cover the brightest pixel in the image
    brightestPixel = 0.00
    brightestXY = [0,0]
    max_x = len(image[0][0]) - rect_shape[0]
    max_y = len(image[0]) - rect_shape[1]
    for y in range(0, max_y):
        for x in range (0, max_x):
            if image[0][y][x][0] > brightestPixel:
                brightestPixel = image[0][y][x][0]
                brightestXY = [x, y]
    start_point = brightestXY
    new_grads = np.zeros_like(gradients)
    patch = gradients[:, start_point[0]:start_point[0] + rect_shape[0], start_point[1]:start_point[1] + rect_shape[1]]
    #if np.mean(patch) < 0:
    new_grads[:, start_point[0]:start_point[0] + rect_shape[0],start_point[1]:start_point[1] + rect_shape[1]] = -np.ones_like(patch)
    return new_grads

def constraint_black_darkest(gradients, image, rect_shape=(8, 8)):
    # calculate the start point to cover the darkest pixel in the image
    darkPixel = 100.00
    darkXY = [0,0]
    max_x = len(image[0][0]) - rect_shape[0]
    max_y = len(image[0]) - rect_shape[1]
    for y in range(0, max_y):
        for x in range (0, max_x):
            if image[0][y][x][0] < darkPixel:
                darkPixel = image[0][y][x][0]
                darkXY = [x, y]
    start_point = darkXY
    new_grads = np.zeros_like(gradients)
    patch = gradients[:, start_point[0]:start_point[0] + rect_shape[0], start_point[1]:start_point[1] + rect_shape[1]]
    #if np.mean(patch) < 0:
    new_grads[:, start_point[0]:start_point[0] + rect_shape[0],start_point[1]:start_point[1] + rect_shape[1]] = -np.ones_like(patch)
    return new_grads

def constraint_black_mean(gradients, image, rect_shape=(8, 8)):
    # calculate the start point to cover the pixel with the value closest to the average pixel value in the image
    avgPixel = 0.00
    avgXY = [0,0]
    max_x = len(image[0][0]) - rect_shape[0]
    max_y = len(image[0]) - rect_shape[1]
    avg_value = 0.00
    num_pixels = 0
    for y in range(0, max_y):
        for x in range (0, max_x):
            avg_value += image[0][y][x][0]
            num_pixels += 1
    avg_value = avg_value / num_pixels
    for y in range(0, max_y):
        for x in range (0, max_x):
            if abs(image[0][y][x][0] - avg_value) < abs(avgPixel - avg_value):
                avgPixel = image[0][y][x][0]
                avgXY = [x, y]
    start_point = avgXY
    new_grads = np.zeros_like(gradients)
    patch = gradients[:, start_point[0]:start_point[0] + rect_shape[0], start_point[1]:start_point[1] + rect_shape[1]]
    #if np.mean(patch) < 0:
    new_grads[:, start_point[0]:start_point[0] + rect_shape[0],start_point[1]:start_point[1] + rect_shape[1]] = -np.ones_like(patch)
    return new_grads

def init_coverage_tables(model1, model2, model3):
    model_layer_dict1 = defaultdict(bool)
    model_layer_dict2 = defaultdict(bool)
    model_layer_dict3 = defaultdict(bool)
    init_dict(model1, model_layer_dict1)
    init_dict(model2, model_layer_dict2)
    init_dict(model3, model_layer_dict3)
    return model_layer_dict1, model_layer_dict2, model_layer_dict3


def init_dict(model, model_layer_dict):
    for layer in model.layers:
        if 'flatten' in layer.name or 'input' in layer.name:
            continue
        for index in range(num_neurons(layer.output_shape)): # product of dims
            model_layer_dict[(layer.name, index)] = False


def neuron_to_cover(model_layer_dict):
    not_covered = [(layer_name, index) for (layer_name, index), v in model_layer_dict.items() if not v]
    if not_covered:
        layer_name, index = random.choice(not_covered)
    else:
        layer_name, index = random.choice(model_layer_dict.keys())
    return layer_name, index


def neuron_covered(model_layer_dict):
    covered_neurons = len([v for v in model_layer_dict.values() if v])
    total_neurons = len(model_layer_dict)
    return covered_neurons, total_neurons, covered_neurons / float(total_neurons)


def update_coverage(input_data, model, model_layer_dict, model_layer_hl_dict, test_only=False, threshold=0):
    snac_dict, nc_dict = {}, {}
    if test_only:
        snac_dict = model_layer_dict["snac_test"]
        nc_dict = model_layer_dict["nc_test"]
    else:
        snac_dict = model_layer_dict["snac"]
        nc_dict = model_layer_dict["nc"]

    layer_names = [layer.name for layer in model.layers if
                   'flatten' not in layer.name and 'input' not in layer.name]

    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
    intermediate_layer_outputs = intermediate_layer_model.predict(input_data)

    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        layer = intermediate_layer_output[0]
        for neuron in xrange(num_neurons(layer.shape)): # index through every single (indiv) neuron
            _,high = model_layer_hl_dict[(layer_names[i], neuron)]

            # evaluate snac criteria
            if  layer[np.unravel_index(neuron, layer.shape)] > high and not snac_dict[(layer_names[i], neuron)]:
                snac_dict[(layer_names[i], neuron)] = True

            # evaluate nc criteria
            if  layer[np.unravel_index(neuron, layer.shape)] > threshold and not nc_dict[(layer_names[i], neuron)]: 
                nc_dict[(layer_names[i], neuron)] = True


def num_neurons(shape):
    return reduce(lambda x,y: x*y, filter(lambda x : x != None, shape))

def full_coverage(model_layer_dict):
    if False in model_layer_dict.values():
        return False
    return True


def scale(intermediate_layer_output, rmax=1, rmin=0):
    X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
        intermediate_layer_output.max() - intermediate_layer_output.min())
    X_scaled = X_std * (rmax - rmin) + rmin
    return X_scaled


def fired(model, layer_name, index, input_data, threshold=0):
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    intermediate_layer_output = intermediate_layer_model.predict(input_data)[0]
    scaled = scale(intermediate_layer_output)
    if np.mean(scaled[..., index]) > threshold:
        return True
    return False


def diverged(predictions1, predictions2, predictions3, target):
    #     if predictions2 == predictions3 == target and predictions1 != target:
    if not predictions1 == predictions2 == predictions3:
        return True
    return False
