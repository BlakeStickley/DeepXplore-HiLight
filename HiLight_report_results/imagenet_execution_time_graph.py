import matplotlib.pyplot as plt

hilight_times = [680.41046, 927.943855, 881.757976, 956.03102, 733.391228, 1001.108861, 725.7053]
random_times = [1328.932073, 1889.395639, 1512.551768, 1380.256516, 1276.461631, 1369.970163, 985.751118, 1261.912603, 2097.107719, 1610.562215]
hilight_sum = 0
random_sum = 0

for index in range(7):
    hilight_sum += hilight_times[index]
    random_sum += random_times[index]
avg_times = []
avg_times.append((random_sum / 7)/60)
avg_times.append((hilight_sum / 7)/60)
names = ['Random', 'HiLight']
name_pos = [i for i, _ in enumerate(names)]

plt.bar(name_pos, avg_times, color=(0.2, 0.4, 0.6, 0.6), edgecolor='black')
plt.title("Time to complete Input Generation on ImageNet")
plt.ylabel("Minutes")

plt.xticks(name_pos, names)
plt.show()
