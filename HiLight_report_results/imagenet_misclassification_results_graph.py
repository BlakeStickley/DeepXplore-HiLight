import matplotlib.pyplot as plt

hilight_diffs = [20, 20, 20, 20, 20, 20, 20]
random_diffs = [19, 17, 20, 19, 20, 20, 19, 20]
hilight_sum = 0
random_sum = 0
for index in range(7):
    hilight_sum += hilight_diffs[index]
    random_sum += random_diffs[index]
hilight_sum = hilight_sum / 7
random_sum = random_sum / 7
num_diffs = [random_sum, hilight_sum]

names = ['Random', 'HiLight']

name_pos = [i for i, _ in enumerate(names)]

plt.bar(name_pos, num_diffs, color='blue')
plt.title("Number of Misclassifications on ImageNet")

plt.xticks(name_pos, names)
plt.show()
