#!/bin/bash
mkdir -p official_report_results

# comment/uncomment lines as needed.
SELECTION="nc" # or snac
# MNIST
# (cd MNIST && python gen_diff.py light 1 0.1 10 500 20 0 $SELECTION) | tee official_report_results/mnist_500_${SELECTION}_selection_light.log
# (cd MNIST && python gen_diff.py occl 1 0.1 10 500 20 0 $SELECTION) | tee official_report_results/mnist_500_${SELECTION}_selection_occl.log
# (cd MNIST && python gen_diff.py blackout 1 0.1 10 500 20 0 $SELECTION) | tee official_report_results/mnist_500_${SELECTION}_selection_blackout.log

# # Driving
# (cd Driving && python gen_diff.py light 1 0.1 10 500 20 0 $SELECTION) | tee official_report_results/driving_500_${SELECTION}_selection_light.log
# (cd Driving && python gen_diff.py occl 1 0.1 10 500 20 0 $SELECTION) | tee official_report_results/driving_500_${SELECTION}_selection_occl.log
# (cd Driving && python gen_diff.py blackout 1 0.1 10 500 20 0 $SELECTION) | tee official_report_results/driving_500_${SELECTION}_selection_blackout.log 

# # PDF
(cd PDF && python gen_diff.py 2 0.1 0.1 500 20 0 $SELECTION) | tee official_report_results/pdf_500_${SELECTION}_selection.log
