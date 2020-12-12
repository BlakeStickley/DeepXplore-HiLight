import matplotlib.pyplot as plt

# Command for running gen_diff.py
# KERAS_BACKEND=tensorflow python gen_diff.py blackout 1 0.1 10 20 1 0 nc
# All results use these parameters unless otherwise stated

x1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]


# Baseline Random, limited to 20 blackouts (one change per iteration)
# - Diffs: 8/20
# - Size: 6x6
# - Blackouts: 20
# - Inner execution time (Milliseconds): 2.65
# - Total execution time (Milliseconds): 9816.019
# - Time to 90% neuron coverage (Milliseconds): 8848.523
reference_limit_20 = [60.0, 60.0, 60.0, 64.041, 64.041, 79.264, 80.856, 87.444, 87.4, 87.4, 88.268, 88.268, 88.268, 88.268, 88.268, 88.268, 89.816, 90.667, 90.7, 92.420]

# HiLight
# - Diffs: 20/20
# - Size: 6x6
# - Blackouts: 20
# - Inner execution time (Milliseconds): 11.943
# - Total execution time (Milliseconds): 13859.16
# - Time to 90% neuron coverage (Milliseconds): 4709.451
hilight = [63.944, 63.944, 82.673, 87.310, 87.310, 89.589, 89.589, 91.480, 92.528, 92.528, 92.528, 92.528, 92.528, 92.528, 92.816, 92.816, 92.954, 93.259, 93.259, 94.170]

# Darkest Pixel
# - Diffs: 20/20
# - Size: 8x8
# - Blackouts: 20
# - Inner execution time (Milliseconds): 8.011
# - Total execution time (Milliseconds): 14714.09
darkest = [63.171, 63.171, 76.973, 79.614, 79.614, 82.274, 82.274, 82.990, 83.979, 83.979, 83.979, 83.979, 83.979, 83.979, 84.117, 84.117, 84.326, 84.624, 84.624, 85.385]

# Mean Value Pixel 
# - Diffs: 18/20
# - Size: 8x8
# - Blackouts: 20
# - Inner execution time (Milliseconds): 74.822
# - Total execution time (Milliseconds): 17012.99
mean_value = [61.642, 61.642, 81.196, 86.295, 86.295, 89.958, 89.958, 92.096, 92.096, 92.096, 92.096, 92.096, 92.096, 92.096, 92.338, 92.338, 92.659, 93.103, 93.103, 93.834]

# Baseline results, allow up to 20 blackouts per image
# - Parameters: 1 0.1 10 20 20 0 nc
# - Diffs: 20/20
# - Size: 6x6
# - Blackouts: 62
# - Inner execution time (Milliseconds): 8.419
# - Total execution time (Milliseconds): 34768.212
# - Time to 90% neuron coverage (Milliseconds): 6257.322
reference = [63.9, 72.9, 84.1, 87.7, 87.9, 90.9, 91.2, 92.5, 93.3, 93.9, 94.2, 94.5, 94.9, 95.4, 95.5, 95.6, 95.8, 96.3, 96.3, 96.8]

# BELOW ARE TESTING RESULTS FROM OTHER METHODS
# - These tests were run with 20 gradient iteration steps, so they are no longer being used to compare against. 
# - Further, these approaches did not provide as promising results as the brightest pixel approach, so they were not used.
random_size_2 = [65.8, 73.4, 85.8, 88.4, 89.2, 91.7, 92.6, 93.1, 94.0, 94.4, 94.5, 94.9, 95.1, 95.4, 95.5, 95.6, 95.7, 95.9, 96.0, 96.4]
middle_target = [63.7, 70.5, 84.0, 87.1, 88.6, 90.2, 91.1, 92.2, 93.1, 93.4, 93.6, 93.6, 93.8, 93.8, 94.0, 94.3, 94.4, 94.6, 94.6, 95.0]
middle_target_rec_size_2 = [65.7, 73.0, 85.1, 88.3, 89.9, 91.8, 92.2, 93.2, 93.8, 93.9, 93.9, 94.0, 94.0, 94.1, 94.3, 94.4, 94.5, 94.7, 94.7, 95.1]
random_rec_size_growing = [65.8, 75.0, 86.6, 88.9, 89.7, 92.0, 92.9, 93.5, 94.2, 94.6, 94.7, 95.2, 95.2, 95.5, 95.6, 95.8, 95.9, 96.1, 96.1, 96.1]


# plt.plot(x1, reference, label = "Base Convergence")
# plt.plot(x1, reference_limit_20, label = "Random")
plt.plot(x1, hilight, label = "HiLight")
plt.plot(x1, darkest, label = "Darkest Pixel")
plt.plot(x1, mean_value, label = "Mean Value Pixel")
plt.ylabel('Neuron Coverage')
plt.xlabel('Changes Made to Image Set')
plt.title('DeepXplore "Blackout" Modification on MNIST')
plt.legend()
plt.show()
