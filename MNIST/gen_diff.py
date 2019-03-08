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

GEN_INPUTS_DIR='../generated_inputs/MNIST/'

# read the parameter
# argument parsing
parser = argparse.ArgumentParser(description='Main function for difference-inducing input generation in MNIST dataset')
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
parser.add_argument('-occl_size', '--occlusion_size', help="occlusion size", default=(10, 10), type=tuple)

args = parser.parse_args()

random.seed(4172306)

# input image dimensions
img_rows, img_cols = 28, 28
# the data, shuffled and split between train and test sets
(_, _), (x_test, _) = mnist.load_data()

x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_test = x_test.astype('float32')
x_test /= 255

# define input tensor as a placeholder
input_tensor = Input(shape=input_shape)

# load multiple models sharing same input tensor
model1 = Model1(input_tensor=input_tensor)
model2 = Model2(input_tensor=input_tensor)
model3 = Model3(input_tensor=input_tensor)

# init coverage table
# model_layer_dict = SNAC coverage (note: this impl uses SNAC to guide neuron selection as well)
# model_layer_dict_only_test = measuring SNAC coverage from test data only (ignoring generated inputs from applied gradients)
# model_layer_nc_dict = NC coverage (analagous to model_layer_dict - still using the same SNAC neuron selection though)
model_layer_dict1, model_layer_dict2, model_layer_dict3 = init_coverage_tables(model1, model2, model3)
model_layer_dict_only_test1, model_layer_dict_only_test2, model_layer_dict_only_test3 = init_coverage_tables(model1, model2, model3)
model_layer_nc_dict1, model_layer_nc_dict2, model_layer_nc_dict3 = init_coverage_tables(model1, model2, model3)
m1_hl = pickle.load(open("m1-10000-samples.p", "rb"))
m2_hl = pickle.load(open("m2-10000-samples.p", "rb"))
m3_hl = pickle.load(open("m3-10000-samples.p", "rb"))

# ==============================================================================================
# start gen inputs
for iter in xrange(args.seeds):
    print("Iteration " + str(iter+1))
    gen_img = np.expand_dims(random.choice(x_test), axis=0)
    orig_img = gen_img.copy()
    # first check if input already induces differences
    label1, label2, label3 = np.argmax(model1.predict(gen_img)[0]), np.argmax(model2.predict(gen_img)[0]), np.argmax(
        model3.predict(gen_img)[0])

    # measuring test-only coverage (ie don't include these only_test dictionaries when computing updated coverage
    # after applying gradients
    update_coverage(gen_img, model1, model_layer_dict_only_test1, m1_hl, args.threshold)
    update_coverage(gen_img, model2, model_layer_dict_only_test2, m2_hl, args.threshold)
    update_coverage(gen_img, model3, model_layer_dict_only_test3, m3_hl, args.threshold)

    if not label1 == label2 == label3:
        print(bcolors.OKGREEN + 'input already causes different outputs: {}, {}, {}'.format(label1, label2,
                                                                                            label3) + bcolors.ENDC)

        update_coverage(gen_img, model1, model_layer_dict1, m1_hl, args.threshold)
        update_coverage(gen_img, model2, model_layer_dict2, m2_hl, args.threshold)
        update_coverage(gen_img, model3, model_layer_dict3, m3_hl, args.threshold)
        update_nc_coverage(gen_img, model1, model_layer_nc_dict1, args.threshold)
        update_nc_coverage(gen_img, model2, model_layer_nc_dict2, args.threshold)
        update_nc_coverage(gen_img, model3, model_layer_nc_dict3, args.threshold)

        print(bcolors.OKGREEN + 'SNAC percentage %d neurons %.3f, %d neurons %.3f, %d neurons %.3f'
              % (len(model_layer_dict1), neuron_covered(model_layer_dict1)[2], len(model_layer_dict2),
                 neuron_covered(model_layer_dict2)[2], len(model_layer_dict3),
                 neuron_covered(model_layer_dict3)[2]) + bcolors.ENDC)
        averaged_snac = (neuron_covered(model_layer_dict1)[0] + neuron_covered(model_layer_dict2)[0] +
                       neuron_covered(model_layer_dict3)[0]) / float(
            neuron_covered(model_layer_dict1)[1] + neuron_covered(model_layer_dict2)[1] +
            neuron_covered(model_layer_dict3)[
                1])
        print(bcolors.OKGREEN + 'averaged SNAC %.3f' % averaged_snac + bcolors.ENDC)

        print(bcolors.OKGREEN + 'covered neurons percentage %d neurons %.3f, %d neurons %.3f, %d neurons %.3f'
              % (len(model_layer_nc_dict1), neuron_covered(model_layer_nc_dict1)[2], len(model_layer_nc_dict2),
                 neuron_covered(model_layer_nc_dict2)[2], len(model_layer_nc_dict3),
                 neuron_covered(model_layer_nc_dict3)[2]) + bcolors.ENDC)
        averaged_nc = (neuron_covered(model_layer_nc_dict1)[0] + neuron_covered(model_layer_nc_dict2)[0] +
                       neuron_covered(model_layer_nc_dict3)[0]) / float(
            neuron_covered(model_layer_nc_dict1)[1] + neuron_covered(model_layer_nc_dict2)[1] +
            neuron_covered(model_layer_nc_dict3)[
                1])
        print(bcolors.OKGREEN + 'averaged covered neurons %.3f' % averaged_nc + bcolors.ENDC)



        gen_img_deprocessed = deprocess_image(gen_img)

        # save the result to disk
        imsave(GEN_INPUTS_DIR + 'already_differ_' + str(label1) + '_' + str(
            label2) + '_' + str(label3) + '.png', gen_img_deprocessed)
        continue

    # if all label agrees
    orig_label = label1

    # construct joint loss function
    if args.target_model == 0:
        loss1 = -args.weight_diff * K.mean(model1.get_layer('before_softmax').output[..., orig_label])
        loss2 = K.mean(model2.get_layer('before_softmax').output[..., orig_label])
        loss3 = K.mean(model3.get_layer('before_softmax').output[..., orig_label])
    elif args.target_model == 1:
        loss1 = K.mean(model1.get_layer('before_softmax').output[..., orig_label])
        loss2 = -args.weight_diff * K.mean(model2.get_layer('before_softmax').output[..., orig_label])
        loss3 = K.mean(model3.get_layer('before_softmax').output[..., orig_label])
    elif args.target_model == 2:
        loss1 = K.mean(model1.get_layer('before_softmax').output[..., orig_label])
        loss2 = K.mean(model2.get_layer('before_softmax').output[..., orig_label])
        loss3 = -args.weight_diff * K.mean(model3.get_layer('before_softmax').output[..., orig_label])

    # we run gradient ascent for 20 steps
    for iters in xrange(args.grad_iterations):

        layer_name1, index1 = neuron_to_cover(model_layer_dict1)
        layer_name2, index2 = neuron_to_cover(model_layer_dict2)
        layer_name3, index3 = neuron_to_cover(model_layer_dict3)
        loss1_neuron = model1.get_layer(layer_name1).output[0][np.unravel_index(index1,list(model1.get_layer(layer_name1).output.shape)[1:])]
        loss2_neuron = model2.get_layer(layer_name2).output[0][np.unravel_index(index2,list(model2.get_layer(layer_name2).output.shape)[1:])]
        loss3_neuron = model3.get_layer(layer_name3).output[0][np.unravel_index(index3,list(model3.get_layer(layer_name3).output.shape)[1:])]
        layer_output = (loss1 + loss2 + loss3) + args.weight_nc * (loss1_neuron + loss2_neuron + loss3_neuron)

        # for adversarial image generation
        final_loss = K.mean(layer_output)

        # we compute the gradient of the input picture wrt this loss
        grads = normalize(K.gradients(final_loss, input_tensor)[0])

        # this function returns the loss and grads given the input picture
        iterate = K.function([input_tensor], [loss1, loss2, loss3, loss1_neuron, loss2_neuron, loss3_neuron, grads])

        loss_value1, loss_value2, loss_value3, loss_neuron1, loss_neuron2, loss_neuron3, grads_value = iterate(
            [gen_img])
        if args.transformation == 'light':
            grads_value = constraint_light(grads_value)  # constraint the gradients value
        elif args.transformation == 'occl':
            grads_value = constraint_occl(grads_value, args.start_point,
                                          args.occlusion_size)  # constraint the gradients value
        elif args.transformation == 'blackout':
            grads_value = constraint_black(grads_value)  # constraint the gradients value

        gen_img += grads_value * args.step
        predictions1 = np.argmax(model1.predict(gen_img)[0])
        predictions2 = np.argmax(model2.predict(gen_img)[0])
        predictions3 = np.argmax(model3.predict(gen_img)[0])

        if not predictions1 == predictions2 == predictions3:
            update_coverage(gen_img, model1, model_layer_dict1, m1_hl, args.threshold)
            update_coverage(gen_img, model2, model_layer_dict2, m2_hl, args.threshold)
            update_coverage(gen_img, model3, model_layer_dict3, m3_hl, args.threshold)
            update_nc_coverage(gen_img, model1, model_layer_nc_dict1, args.threshold)
            update_nc_coverage(gen_img, model2, model_layer_nc_dict2, args.threshold)
            update_nc_coverage(gen_img, model3, model_layer_nc_dict3, args.threshold)

            print("Found output which causes difference in models' predictions.")

            print(bcolors.OKGREEN + 'SNAC percentage %d neurons %.3f, %d neurons %.3f, %d neurons %.3f'
                  % (len(model_layer_dict1), neuron_covered(model_layer_dict1)[2], len(model_layer_dict2),
                     neuron_covered(model_layer_dict2)[2], len(model_layer_dict3),
                     neuron_covered(model_layer_dict3)[2]) + bcolors.ENDC)
            averaged_snac = (neuron_covered(model_layer_dict1)[0] + neuron_covered(model_layer_dict2)[0] +
                             neuron_covered(model_layer_dict3)[0]) / float(
                neuron_covered(model_layer_dict1)[1] + neuron_covered(model_layer_dict2)[1] +
                neuron_covered(model_layer_dict3)[
                    1])
            print(bcolors.OKGREEN + 'averaged SNAC %.3f' % averaged_snac + bcolors.ENDC)

            print(bcolors.OKGREEN + 'covered neurons percentage %d neurons %.3f, %d neurons %.3f, %d neurons %.3f'
                  % (len(model_layer_nc_dict1), neuron_covered(model_layer_nc_dict1)[2], len(model_layer_nc_dict2),
                     neuron_covered(model_layer_nc_dict2)[2], len(model_layer_nc_dict3),
                     neuron_covered(model_layer_nc_dict3)[2]) + bcolors.ENDC)
            averaged_nc = (neuron_covered(model_layer_nc_dict1)[0] + neuron_covered(model_layer_nc_dict2)[0] +
                           neuron_covered(model_layer_nc_dict3)[0]) / float(
                neuron_covered(model_layer_nc_dict1)[1] + neuron_covered(model_layer_nc_dict2)[1] +
                neuron_covered(model_layer_nc_dict3)[
                    1])
            print(bcolors.OKGREEN + 'averaged covered neurons %.3f' % averaged_nc + bcolors.ENDC)

            gen_img_deprocessed = deprocess_image(gen_img)
            orig_img_deprocessed = deprocess_image(orig_img)

            # save the result to disk
            imsave(GEN_INPUTS_DIR + args.transformation + '_' + str(predictions1) + '_' + str(
                predictions2) + '_' + str(predictions3) + '.png',
                   gen_img_deprocessed)
            imsave(GEN_INPUTS_DIR + args.transformation + '_' + str(predictions1) + '_' + str(
                predictions2) + '_' + str(predictions3) + '_orig.png',
                   orig_img_deprocessed)
            break

print("Final coverage metric from test data with adversarial example generation")


print(bcolors.OKGREEN + 'SNAC percentage %d neurons %.3f, %d neurons %.3f, %d neurons %.3f'
      % (len(model_layer_dict1), neuron_covered(model_layer_dict1)[2], len(model_layer_dict2),
         neuron_covered(model_layer_dict2)[2], len(model_layer_dict3),
         neuron_covered(model_layer_dict3)[2]) + bcolors.ENDC)
averaged_snac = (neuron_covered(model_layer_dict1)[0] + neuron_covered(model_layer_dict2)[0] +
               neuron_covered(model_layer_dict3)[0]) / float(
    neuron_covered(model_layer_dict1)[1] + neuron_covered(model_layer_dict2)[1] +
    neuron_covered(model_layer_dict3)[
        1])
print(bcolors.OKGREEN + 'averaged SNAC %.3f' % averaged_snac + bcolors.ENDC)


print(bcolors.OKGREEN + 'covered neurons percentage %d neurons %.3f, %d neurons %.3f, %d neurons %.3f'
      % (len(model_layer_nc_dict1), neuron_covered(model_layer_nc_dict1)[2], len(model_layer_nc_dict2),
         neuron_covered(model_layer_nc_dict2)[2], len(model_layer_nc_dict3),
         neuron_covered(model_layer_nc_dict3)[2]) + bcolors.ENDC)
averaged_nc = (neuron_covered(model_layer_nc_dict1)[0] + neuron_covered(model_layer_nc_dict2)[0] +
               neuron_covered(model_layer_nc_dict3)[0]) / float(
    neuron_covered(model_layer_nc_dict1)[1] + neuron_covered(model_layer_nc_dict2)[1] +
    neuron_covered(model_layer_nc_dict3)[
        1])
print(bcolors.OKGREEN + 'averaged covered neurons %.3f' % averaged_nc + bcolors.ENDC)


print("Final coverage metric solely from test data")
print(bcolors.OKGREEN + 'covered neurons percentage %d neurons %.3f, %d neurons %.3f, %d neurons %.3f'
        % (len(model_layer_dict_only_test1), neuron_covered(model_layer_dict_only_test1)[2], len(model_layer_dict_only_test2),
            neuron_covered(model_layer_dict_only_test2)[2], len(model_layer_dict_only_test3),
            neuron_covered(model_layer_dict_only_test3)[2]) + bcolors.ENDC)
averaged_nc = (neuron_covered(model_layer_dict_only_test1)[0] + neuron_covered(model_layer_dict_only_test2)[0] +
                neuron_covered(model_layer_dict_only_test3)[0]) / float(
    neuron_covered(model_layer_dict_only_test1)[1] + neuron_covered(model_layer_dict_only_test2)[1] +
    neuron_covered(model_layer_dict_only_test3)[
        1])
print(bcolors.OKGREEN + 'averaged covered neurons %.3f' % averaged_nc + bcolors.ENDC)