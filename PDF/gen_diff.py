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

random.seed(4172306)

# read the parameter
# argument parsing
parser = argparse.ArgumentParser(
    description='Main function for difference-inducing input generation in VirusTotal/Contagio dataset')
parser.add_argument('weight_diff', help="weight hyperparm to control differential behavior", type=float)
parser.add_argument('weight_nc', help="weight hyperparm to control neuron coverage", type=float)
parser.add_argument('step', help="step size of gradient descent", type=float)
parser.add_argument('seeds', help="number of seeds of input", type=int)
parser.add_argument('grad_iterations', help="number of iterations of gradient descent", type=int)
parser.add_argument('threshold', help="threshold for determining neuron activated", type=float)
parser.add_argument('coverage', help='Coverage criteria targeted', choices=["nc", "snac"])
parser.add_argument('-t', '--target_model', help="target model that we want it predicts differently",
                    choices=[0, 1, 2], default=0, type=int)

args = parser.parse_args()

X_test, _, names = datasets.csv2numpy('./dataset/test.csv')
X_test = X_test.astype('float32')
num_features = X_test.shape[1]
feat_names = FeatureDescriptor.get_feature_names()
incre_idx, incre_decre_idx = init_feature_constraints(feat_names)

output_file = "../generated_inputs/PDF/pdf.txt"

# define input tensor as a placeholder
input_tensor = Input(shape=(num_features,))

# load multiple models sharing same input tensor
K.set_learning_phase(0)
model1 = Model1(input_tensor=input_tensor, load_weights=True)
model2 = Model2(input_tensor=input_tensor, load_weights=True)
model3 = Model3(input_tensor=input_tensor, load_weights=True)

# init coverage table
m1_dict, m2_dict, m3_dict = {}, {}, {}
m1_dict["snac"], m2_dict["snac"], m3_dict["snac"] = init_coverage_tables(model1, model2, model3)
m1_dict["snac_test"], m2_dict["snac_test"], m3_dict["snac_test"] = init_coverage_tables(model1, model2, model3)
m1_dict["nc"], m2_dict["nc"], m3_dict["nc"] = init_coverage_tables(model1, model2, model3)
m1_dict["nc_test"], m2_dict["nc_test"], m3_dict["nc_test"] = init_coverage_tables(model1, model2, model3)

m1_hl = pickle.load(open("m1.p", "rb"))
m2_hl = pickle.load(open("m2.p", "rb"))
m3_hl = pickle.load(open("m3.p", "rb"))

def outputCoverage(m1, m2, m3, c):
    print(bcolors.OKGREEN + '%s percentage %d neurons %.3f, %d neurons %.3f, %d neurons %.3f'
              % (c, len(m1), neuron_covered(m1)[2], len(m2),
                 neuron_covered(m2)[2], len(m3),
                 neuron_covered(m3)[2]) + bcolors.ENDC)
    averaged_coverage = (neuron_covered(m1)[0] + neuron_covered(m2)[0] +
                       neuron_covered(m3)[0]) / float(
            neuron_covered(m1)[1] + neuron_covered(m2)[1] +
            neuron_covered(m3)[
                1])
    print(bcolors.OKGREEN + 'averaged %s %.3f' % (c, averaged_coverage) + bcolors.ENDC)


if args.coverage == "nc":
    print("\nRunning DeepXplore with coverage: Neuron Coverage")
elif args.coverage == "snac":
    print("\nRunning DeepXplore with coverage: SNAC")

# ==============================================================================================
# start gen inputs

random.shuffle(X_test)
test_data = X_test[:args.seeds]
iter = 0
differences = 0

for idx, pdf in enumerate(test_data):
    gen_pdf = np.expand_dims(pdf, axis=0)
    orig_pdf = gen_pdf.copy()
    print("\nIteration " + str(iter+1))
    iter += 1

    outputCoverage(m1_dict["snac"], m2_dict["snac"], m3_dict["snac"], "SNAC")
    outputCoverage(m1_dict["nc"], m2_dict["nc"], m3_dict["nc"], "Neuron Coverage")

    update_coverage(gen_pdf, model1, m1_dict, m1_hl, True, args.threshold)
    update_coverage(gen_pdf, model2, m2_dict, m2_hl, True, args.threshold)
    update_coverage(gen_pdf, model3, m3_dict, m3_hl, True, args.threshold)


    # first check if input already induces differences
    label1, label2, label3 = np.argmax(model1.predict(gen_pdf)[0]), np.argmax(model2.predict(gen_pdf)[0]), np.argmax(
        model3.predict(gen_pdf)[0])
    if not label1 == label2 == label3:
        print(bcolors.OKGREEN + 'input already causes different outputs: {}, {}, {}'.format(label1, label2,
                                                                                            label3) + bcolors.ENDC)

        update_coverage(gen_pdf, model1, m1_dict, m1_hl, args.threshold)
        update_coverage(gen_pdf, model2, m2_dict, m2_hl, args.threshold)
        update_coverage(gen_pdf, model3, m3_dict, m3_hl, args.threshold)

        outputCoverage(m1_dict["snac"], m2_dict["snac"], m3_dict["snac"], "SNAC")
        outputCoverage(m1_dict["nc"], m2_dict["nc"], m3_dict["nc"], "Neuron Coverage")

        # save the result to disk
        with open(output_file, 'a') as f:
            f.write(
                'Already causes differences: name: {}, label1:{}, label2: {}, label3: {}\n'.format(names[idx], label1,
                                                                                                   label2, label3))
        continue

    # construct joint loss function
    if args.target_model == 0:
        loss1 = -args.weight_diff * K.mean(model1.get_layer('before_softmax').output[..., label1])
        loss2 = K.mean(model2.get_layer('before_softmax').output[..., label1])
        loss3 = K.mean(model3.get_layer('before_softmax').output[..., label1])
    elif args.target_model == 1:
        loss1 = K.mean(model1.get_layer('before_softmax').output[..., label2])
        loss2 = -args.weight_diff * K.mean(model2.get_layer('before_softmax').output[..., label2])
        loss3 = K.mean(model3.get_layer('before_softmax').output[..., label2])
    elif args.target_model == 2:
        loss1 = K.mean(model1.get_layer('before_softmax').output[..., label3])
        loss2 = K.mean(model2.get_layer('before_softmax').output[..., label3])
        loss3 = -args.weight_diff * K.mean(model3.get_layer('before_softmax').output[..., label3])

    # we run gradient ascent for 20 steps
    for iters in xrange(args.grad_iterations):

        # if all turning angles roughly the same
        orig_label = label1
        layer_name1, index1 = neuron_to_cover(m1_dict[args.coverage])
        layer_name2, index2 = neuron_to_cover(m2_dict[args.coverage])
        layer_name3, index3 = neuron_to_cover(m3_dict[args.coverage])
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
            [gen_pdf])
        grads_value = constraint(grads_value, incre_idx, incre_decre_idx)  # constraint the gradients value

        gen_pdf += grads_value * args.step
        label1, label2, label3 = np.argmax(model1.predict(gen_pdf)[0]), np.argmax(
            model2.predict(gen_pdf)[0]), np.argmax(model3.predict(gen_pdf)[0])

        if not label1 == label2 == label3:
            update_coverage(gen_pdf, model1, m1_dict, m1_hl, args.threshold)
            update_coverage(gen_pdf, model2, m2_dict, m2_hl, args.threshold)
            update_coverage(gen_pdf, model3, m3_dict, m3_hl, args.threshold)

            print("Found new output which causes difference in models' predictions.")
            differences += 1
            outputCoverage(m1_dict["snac"], m2_dict["snac"], m3_dict["snac"], "SNAC")
            outputCoverage(m1_dict["nc"], m2_dict["nc"], m3_dict["nc"], "Neuron Coverage")

            # save the result to disk
            with open(output_file, 'a') as f:
                f.write(
                    'name: {}, label1:{}, label2: {}, label3: {}\n'.format(names[idx], label1, label2, label3))
                f.write('changed features: {}\n\n'.format(features_changed(gen_pdf, orig_pdf, feat_names)))
            break


print("Total differences found: %i" % differences)
print("Final coverage metric from test data with adversarial example generation: ")
outputCoverage(m1_dict["snac"], m2_dict["snac"], m3_dict["snac"], "SNAC")
outputCoverage(m1_dict["nc"], m2_dict["nc"], m3_dict["nc"], "Neuron Coverage")

print("Final coverage metric solely from test data: ")
outputCoverage(m1_dict["snac_test"], m2_dict["snac_test"], m3_dict["snac_test"], "SNAC")
outputCoverage(m1_dict["nc_test"], m2_dict["nc_test"], m3_dict["nc_test"], "Neuron Coverage")