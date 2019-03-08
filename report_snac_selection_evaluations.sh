#!/bin/bash
# for each iteration, prints DX-NC and True-NC (name??). Note that True-NC calculates individual neuron coverage based on the unscaled activation output.
mkdir -p official_report_results

# MNIST
echo "Warning: only running MNIST right now (we still need to update Driving + PDF with appropriate nc metrics)"
(cd MNIST && python gen_diff.py light 1 0.1 10 500 20 0) | tee official_report_results/mnist_500_snac_selection_light.log
(cd MNIST && python gen_diff.py occl 1 0.1 10 500 20 0) | tee official_report_results/mnist_500_snac_selection_occl.log
(cd MNIST && python gen_diff.py blackout 1 0.1 10 500 20 0) | tee official_report_results/mnist_500_snac_selection_blackout.log

# # Driving
# (cd Driving && python gen_diff.py light 1 0.1 10 500 20 0) | tee official_report_results/driving_500_snac_selection_light.log
# (cd Driving && python gen_diff.py occl 1 0.1 10 500 20 0) | tee official_report_results/driving_500_snac_selection_occl.log
# (cd Driving && python gen_diff.py blackout 1 0.1 10 500 20 0) | tee official_report_results/driving_500_snac_selection_blackout.log 

# # PDF
# (cd PDF && python gen_diff.py 2 0.1 0.1 500 20 0) | tee official_report_results/pdf_500_snac_selection.log
