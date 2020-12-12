import matplotlib.pyplot as plt


names = ['Random', 'HiLight']
speed_up = [1, 1.879]

name_pos = [i for i, _ in enumerate(names)]

plt.bar(name_pos, speed_up, color=(0.2, 0.4, 0.6, 0.6), edgecolor='black')
plt.title("Speed Up to Reach 90% Neuron Coverage on MNIST")
plt.ylabel("Realitive Speed Up")

plt.xticks(name_pos, names)
plt.show()
