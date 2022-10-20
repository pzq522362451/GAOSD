import numpy as np
import matplotlib.pyplot as plt
# randomseed==108
names=['f-Bi-GAN','f-WFE','f-SAE','f-DCAE']
boxplot_Bi=[92.76,92.28,92.19,92.80,92.76,93.76,92.86,92.46,92.81,92.67,
           92.42,92.27,92.38,92.17,92.17,92.71,91.74,93.49,92.76,92.76]
boxplot_WFE=[78.06,78.24,79.53,77.58,78.16,78.89,76.47,78.06,78.27,77.85,
             79.01,77.58,78.79,78.31,78.47,76.38,78.68,78.42,78.36,77.47]
boxplot_SAE=[91.68,91.84,90.37,90.39,89.40,90.29,90.59,91.45,90.57,90.49,
             89.53,90.67,91.67,90.50,90.48,90.41,91.47,90.53,90.38,90.57]
boxplot_DCAE=[51.23,51.79,51.48,50.19,51.57,50.78,51.25,51.52,51.58,50.51,
              50.45,51.57,50.38,51.37,51.53,51.71,51.34,50.39,51.65,51.87]
#BoxPlot for Fig_8 (average accuracy)
data=np.vstack(([boxplot_Bi, boxplot_WFE, boxplot_SAE, boxplot_DCAE]))

label='GAOSD','OSD-WFE','OSD-SAE','OSD-DCAE'

fig1, ax1 = plt.subplots()
plt.figure(1)
ax1.boxplot([data[0],data[1],data[2],data[3]],
            notch=False, labels = label,patch_artist = False, medianprops={'color':'red'},boxprops = {'color':'blue','linewidth':'1.0'},
            capprops={'color':'black','linewidth':'1.0'})
color = ['blue', 'orange', 'green','red','purple']  # 有多少box就对应设置多少颜色
# ax1.legend(['RF-GAN'], loc='upper right')
# plt.ylim(ymin=0,ymax=100)
plt.savefig('../results/boxplot.png')
plt.show()

