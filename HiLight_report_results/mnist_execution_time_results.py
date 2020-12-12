# EXECUTION TIME BAR GRAPH
# This compares the totalt execution time of HiLight and the default (unrestricted) random appraoch.
# Here, HiLight is only generating one modification per image while the random appraoch still can
# generate up to 20 modifications per image. This comparison is valid since it goes to show the 
# advantages that an optimized single modifcation can have over the performance overhead of the 
# unoptimized random selection.
# This data can also be found in more detail in mnist_reference_plot.py

import matplotlib.pyplot as plt


names = ['Random', 'HiLight']
num_diffs = [34.77, 13.86]

name_pos = [i for i, _ in enumerate(names)]

plt.bar(name_pos, num_diffs, color=(0.2, 0.4, 0.6, 0.6), edgecolor='black')
plt.title("Time to Complete Input Generation on MNIST")
plt.ylabel("Seconds")

plt.xticks(name_pos, names)
plt.show()
