import matplotlib.pyplot as plt
import numpy as np

x=np.array([99.96,45.50,41.50,3.62]).reshape(4,1)
x=np.array([99.96,100,100,100]).reshape(4,1)
fig1, ax1 = plt.subplots()
plt.figure(1)
ax1.boxplot(x,notch=False,
            patch_artist = False,
            medianprops={'color': 'red'},
            boxprops={'color': 'blue', 'linewidth': '1.0'},
            capprops={'color': 'black', 'linewidth': '1.0'})
# color = ['blue', 'orange', 'green','red','purple']  # 有多少box就对应设置多少颜色
plt.ylim(ymin=0,ymax=100)
plt.show()
# ax1.legend(['RF-GAN2'], loc='upper right')