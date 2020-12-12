'''
usage: python gen_diff.py -h
'''

from __future__ import print_function

import argparse
import datetime

from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.layers import Input
from scipy.misc import imsave

from configs import bcolors
from utils import *

GEN_INPUTS_DIR='../generated_inputs/ImageNet/'

# read the parameter
# argument parsing
parser = argparse.ArgumentParser(
    description='Main function for difference-inducing input generation in ImageNet dataset')
parser.add_argument('transformation', help="realistic transformation type", choices=['light', 'occl', 'blackout'])
parser.add_argument('weight_diff', help="weight hyperparm to control differential behavior", type=float)
parser.add_argument('weight_nc', help="weight hyperparm to control neuron coverage", type=float)
parser.add_argument('step', help="step size of gradient descent", type=float)
parser.add_argument('seeds', help="number of seeds of input", type=int)
parser.add_argument('grad_iterations', help="number of iterations of gradient descent", type=int)
parser.add_argument('threshold', help="threshold for determining neuron activated", type=float)
parser.add_argument('-t', '--target_model', help="target model that we want it predicts differently",
                    choices=[0, 1, 2], default=0, type=int)
parser.add_argument('-sp', '--start_point', help="occlusion upper left corner coordinate", default=(0, 0), type=tuple)
parser.add_argument('-occl_size', '--occlusion_size', help="occlusion size", default=(50, 50), type=tuple)

args = parser.parse_args()

# input image dimensions
img_rows, img_cols = 224, 224
input_shape = (img_rows, img_cols, 3)

# define input tensor as a placeholder
input_tensor = Input(shape=input_shape)

# load multiple models sharing same input tensor
K.set_learning_phase(0)
model1 = VGG16(input_tensor=input_tensor)
model2 = VGG19(input_tensor=input_tensor)
model3 = ResNet50(input_tensor=input_tensor)
# init coverage table
model_layer_dict1, model_layer_dict2, model_layer_dict3 = init_coverage_tables(model1, model2, model3)

# ==============================================================================================
# start gen inputs

coverageList = list()
goal_completed = False
num_blackouts = 0
differences = 0
inner_total = 0
averaged_nc = 0

#TIMING START
start_time = datetime.datetime.now()

img_paths = image.list_pictures('./seeds', ext='JPEG')
for _ in xrange(args.seeds):
    gen_img = preprocess_image(random.choice(img_paths))
    orig_img = gen_img.copy()
    # first check if input already induces differences
    pred1, pred2, pred3 = model1.predict(gen_img), model2.predict(gen_img), model3.predict(gen_img)
    label1, label2, label3 = np.argmax(pred1[0]), np.argmax(pred2[0]), np.argmax(pred3[0])
    #if not label1 == label2 == label3:
    #    print(bcolors.OKGREEN + 'input already causes different outputs: {}, {}, {}'.format(decode_label(pred1),
    #                                                                                        decode_label(pred2),
    #                                                                                        decode_label(
    #                                                                                            pred3)) + bcolors.ENDC)

    #    update_coverage(gen_img, model1, model_layer_dict1, args.threshold)
    #    update_coverage(gen_img, model2, model_layer_dict2, args.threshold)
    #    update_coverage(gen_img, model3, model_layer_dict3, args.threshold)

    #    print(bcolors.OKGREEN + 'covered neurons percentage %d neurons %.3f, %d neurons %.3f, %d neurons %.3f'
    #          % (len(model_layer_dict1), neuron_covered(model_layer_dict1)[2], len(model_layer_dict2),
    #             neuron_covered(model_layer_dict2)[2], len(model_layer_dict3),
    #             neuron_covered(model_layer_dict3)[2]) + bcolors.ENDC)
    #    averaged_nc = (neuron_covered(model_layer_dict1)[0] + neuron_covered(model_layer_dict2)[0] +
    #                   neuron_covered(model_layer_dict3)[0]) / float(
    #        neuron_covered(model_layer_dict1)[1] + neuron_covered(model_layer_dict2)[1] +
    #        neuron_covered(model_layer_dict3)[
    #            1])
    #    print(bcolors.OKGREEN + 'averaged covered neurons %.3f' % averaged_nc + bcolors.ENDC)

    #    gen_img_deprocessed = deprocess_image(gen_img)

        # save the result to disk
    #    imsave(GEN_INPUTS_DIR + 'already_differ_' + decode_label(pred1) + '_' + decode_label(
    #        pred2) + '_' + decode_label(pred3) + '.png',
    #           gen_img_deprocessed)
    #    continue

    # if all label agrees
    orig_label = label1
    layer_name1, index1 = neuron_to_cover(model_layer_dict1)
    layer_name2, index2 = neuron_to_cover(model_layer_dict2)
    layer_name3, index3 = neuron_to_cover(model_layer_dict3)

    # construct joint loss function
    if args.target_model == 0:
        loss1 = -args.weight_diff * K.mean(model1.get_layer('predictions').output[..., orig_label])
        loss2 = K.mean(model2.get_layer('predictions').output[..., orig_label])
        loss3 = K.mean(model3.get_layer('fc1000').output[..., orig_label])
    elif args.target_model == 1:
        loss1 = K.mean(model1.get_layer('predictions').output[..., orig_label])
        loss2 = -args.weight_diff * K.mean(model2.get_layer('predictions').output[..., orig_label])
        loss3 = K.mean(model3.get_layer('fc1000').output[..., orig_label])
    elif args.target_model == 2:
        loss1 = K.mean(model1.get_layer('predictions').output[..., label1])
        loss2 = K.mean(model2.get_layer('predictions').output[..., orig_label])
        loss3 = -args.weight_diff * K.mean(model3.get_layer('fc1000').output[..., orig_label])
    loss1_neuron = K.mean(model1.get_layer(layer_name1).output[..., index1])
    loss2_neuron = K.mean(model2.get_layer(layer_name2).output[..., index2])
    loss3_neuron = K.mean(model3.get_layer(layer_name3).output[..., index3])
    layer_output = (loss1 + loss2 + loss3) + args.weight_nc * (loss1_neuron + loss2_neuron + loss3_neuron)

    # for adversarial image generation
    final_loss = K.mean(layer_output)

    # we compute the gradient of the input picture wrt this loss
    grads = normalize(K.gradients(final_loss, input_tensor)[0])

    # this function returns the loss and grads given the input picture
    iterate = K.function([input_tensor], [loss1, loss2, loss3, loss1_neuron, loss2_neuron, loss3_neuron, grads])

    # we run gradient ascent for a specified number of steps
    first_mod = True
    for iters in xrange(args.grad_iterations):
        loss_value1, loss_value2, loss_value3, loss_neuron1, loss_neuron2, loss_neuron3, grads_value = iterate(
            [gen_img])
        if args.transformation == 'light':
            grads_value = constraint_light(grads_value)  # constraint the gradients value
        elif args.transformation == 'occl':
            grads_value = constraint_occl(grads_value, args.start_point,
                                          args.occlusion_size)  # constraint the gradients value
        elif args.transformation == 'blackout':
	    inner_start_time = datetime.datetime.now()
	    num_blackouts += 1
	    # Blackout Modification
	    # CHANGE THIS LINE TO USE OTHER SELECTION APPROACHES
            grads_value = constraint_black_hilight(grads_value, gen_img)  # constraint the gradients value
	    inner_end_time = datetime.datetime.now()
	    inner_diff = (inner_end_time - inner_start_time)
	    inner_total += inner_diff.total_seconds()

        gen_img += grads_value * args.step
        pred1, pred2, pred3 = model1.predict(gen_img), model2.predict(gen_img), model3.predict(gen_img)
        label1, label2, label3 = np.argmax(pred1[0]), np.argmax(pred2[0]), np.argmax(pred3[0])

        if not label1 == label2 == label3:
            update_coverage(gen_img, model1, model_layer_dict1, args.threshold)
            update_coverage(gen_img, model2, model_layer_dict2, args.threshold)
            update_coverage(gen_img, model3, model_layer_dict3, args.threshold)

            print(bcolors.OKGREEN + 'covered neurons percentage %d neurons %.3f, %d neurons %.3f, %d neurons %.3f'
                  % (len(model_layer_dict1), neuron_covered(model_layer_dict1)[2], len(model_layer_dict2),
                     neuron_covered(model_layer_dict2)[2], len(model_layer_dict3),
                     neuron_covered(model_layer_dict3)[2]) + bcolors.ENDC)
            averaged_nc = (neuron_covered(model_layer_dict1)[0] + neuron_covered(model_layer_dict2)[0] +
                           neuron_covered(model_layer_dict3)[0]) / float(
                neuron_covered(model_layer_dict1)[1] + neuron_covered(model_layer_dict2)[1] +
                neuron_covered(model_layer_dict3)[
                    1])
            print(bcolors.OKGREEN + 'averaged covered neurons %.3f' % averaged_nc + bcolors.ENDC)

            differences += 1
	    coverageList.append('%.3f' % (averaged_nc*100))
	    if(averaged_nc >= 0.95 and goal_completed == False):
		goal_completed = True
		goal_end = datetime.datetime.now()
		goal_diff = (goal_end - start_time)

            gen_img_deprocessed = deprocess_image(gen_img)
            orig_img_deprocessed = deprocess_image(orig_img)

            # save the result to disk
            imsave(GEN_INPUTS_DIR + args.transformation + '_' + decode_label(pred1) + '_' + decode_label(
                pred2) + '_' + decode_label(pred3) + '.png', gen_img_deprocessed)
            imsave(GEN_INPUTS_DIR + args.transformation + '_' + decode_label(pred1) + '_' + decode_label(
                pred2) + '_' + decode_label(pred3) + '_orig.png', orig_img_deprocessed)
            break
        else:
            if(first_mod == True):
                coverageList.append('%.3f' % (averaged_nc*100))
                first_mod = False

# TIMING END
end_time = datetime.datetime.now()
time_diff = (end_time - start_time)
execution_time = time_diff.total_seconds() * 1000
# goal_time = goal_diff.total_seconds() * 1000

print("Total differences found: %i" % differences)
print("Total execution time: ")
print(execution_time)
print("Inner execution time: ")
print(inner_total * 1000)
# print("Time to reach 95 percent NC: ")
# print(goal_time)

print("Final coverage plot: [")
print(*coverageList , sep = ", ")
print(" ]")

print("num blackouts: %d" % num_blackouts)


