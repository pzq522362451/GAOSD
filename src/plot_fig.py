import matplotlib.pyplot as plt
import numpy as np

# fB=np.array([56.70,60.35,65.67,70.85,80.34,87.69,89.23,92.36,95.89,98.76,98.78,98.82,98.96])
# x=np.arange(13)
# plt.figure(1)
# plt.plot(fB)
# plt.savefig('../results/few-shot.png')
# plt.show()

#
# x=np.arange(13)
# #task 1
# fB=np.array([20.28,64.71,97.00,55.32,30.43,94.18,78.95,25.00,25.00,26.13,26.67,25.57,46.76])
# plt.bar(x,fB,width = 0.8, edgecolor = 'black', linewidth = 1, align = 'center', yerr = 0.5, ecolor = 'r')
# plt.xticks(np.arange(13))
# plt.savefig('../results/bar-task1.png')
# plt.show()

#task 2
# fB=np.array([74.69,41.42,95.74,36.65,35.98,96.88,97.03,33.43,33.33,34.01,38.01,33.96,34.15])
# plt.bar(x,fB,width = 0.8, edgecolor = 'black', linewidth = 1, align = 'center', yerr = 0.5, ecolor = 'r')
# plt.xticks(np.arange(13))
# plt.savefig('../results/bar-task2.png')
# plt.show()


#task 3
# x=np.arange(7)
# fB=np.array([49.09,95.74,49.87,51.24,6.03,8.22,70.79])
# plt.bar(x,fB,width = 0.8, edgecolor = 'black', linewidth = 2, align = 'center', yerr = 0.5, ecolor = 'r')
# plt.xticks(np.arange(7))
# plt.savefig('../results/bar-task3.png')
# plt.show()

#one shot
x=np.arange(7)
fB=np.array([92.08,96.35,99.94,100,100,100,100])
plt.plot(fB, 'r')
plt.savefig('../results/one-shot_var.png')
plt.show()







