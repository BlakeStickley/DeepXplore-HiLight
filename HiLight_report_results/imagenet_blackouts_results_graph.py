import matplotlib.pyplot as plt

hilight_blackouts = [34, 58, 54, 58, 35, 63, 34]
random_blackouts = [81, 136, 111, 98, 90, 99, 65, 90, 165, 114]
hilight_sum = 0
random_sum = 0
for index in range (7):
    hilight_sum += hilight_blackouts[index]
    random_sum += random_blackouts[index]
hilight_sum = (hilight_sum / 7) # / 20
random_sum = (random_sum / 7) # / 20
num_blackouts = [random_sum, hilight_sum]

names = ['Random', 'HiLight']

name_pos = [i for i, _ in enumerate(names)]
plt.bar(name_pos, num_blackouts, color='blue')
plt.title("Average Number of Blackouts Applied to Image Set")

plt.xticks(name_pos, names)
plt.show()
