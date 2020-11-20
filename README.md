# HiLight, intelligent modification selection extension to DeepXplore++
This project is an extension of DeepXplore, focusing on imporving the random selection process that is used to determine "blackout" modification placement on images. This project builds off of the DeepXplore++ repository. Please ensure that the correct versions of the needed dependencies are installed.

# DeepXplore++, An extension of DeepXplore
See the SOSP'17 paper [DeepXplore: Automated Whitebox Testing of Deep Learning Systems](http://www.cs.columbia.edu/~suman/docs/deepxplore.pdf) for more details.

DeepXplore++ fixes discrepancies in the implementation of Neuron Coverage between the description in the paper and in the implementation found on github. In addition, DeepXplore is extended to include SNAC (Strong Neuron Activation Coverage) as a coverage criteria to be tracked as well as targetted.

# Building DeepXplore++
DeepXplore++ contains a docker file that allows for easier setup. The steps for running are the following:

1. Install [Docker](https://www.docker.com/)
2. Clone this repository to your machine
3. cd to the repository main directory
4. Run `docker build .` which will start building the image
5. Run `docker run -it [IMAGE ID]` where image ID is the hash of the image

For instructions on setting up the project locally, please see the original DeepXplore implementation. This project was run locally during testing, but this docker file is still included for convenience.

# Running DeepXplore++ with HiLight

While DeepXplore supports multiple types of models, HiLight as only been implemented for MNIST and ImageNet "blackout" modifications during the course of this project. All other tests are the existing DeepXplore++ tests.

At this point, you can enter any of the model directories which DeepXplore++ supports, (PDF, MNIST, Driving) and you can run the `gen_diff.py` file to generate adversarial examples and track coverage. 

`gen_diff.py` takes as input model specific arguments such as the type of input modification, lambda parameters, the number of gradient ascent iterations, and a coverage criteria that is being targetted. Examples for the supported models are shown below. See the `gen_diff.py` file for the specific model if you have any more questions.

MNIST - `KERAS_BACKEND=tensorflow python gen_diff.py blackout 1 0.1 10 20 1 0 nc`

ImageNet - `KERAS_BACKEND=tensorflow python gen_diff.py blackout 1 0.1 10 20 1 0 nc`

Note; see the HiLight results files for the specific hyperparameters used for each model

# HiLight Experimental Data

Experimental data for this project can be found in the `HiLight_report_results` directory. Files specify the model used, the hyperparameters, and the colleted results.  



