import matplotlib.pyplot as plt


names = ['Random', 'HiLight']
num_diffs = [8, 20]

name_pos = [i for i, _ in enumerate(names)]

plt.bar(name_pos, num_diffs, color='blue')
plt.title("Number of Misclassifications on MNIST")

plt.xticks(name_pos, names)
plt.show()
