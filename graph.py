import matplotlib.pyplot as plt
import numpy as np


label = ['IMDB -> FineFood', 'Source -> FineFood']

N=3
index = np.arange(2)
bar_width = 0.9/N

#Color Scheme
colors = ['#fc9434', '#fccc94', '#148c94', '#4ba8a9','#98c8c8', '#2c4b4c']

#Results
p1_result = [56.8,56.8] #Source
p2_result = [21.2,21.2] #LNTENT
p3_result = [78.9,78.7] #TeTra

p1 = plt.bar(index, p1_result, 
             bar_width, 
             color=colors[0],
             hatch='O')

p2 = plt.bar(index+ [bar_width]*2, p2_result, 
             bar_width, 
             color=colors[1],
             hatch='//')

p3 = plt.bar(index+ [bar_width*2]*2, p3_result, 
             bar_width, 
             color=colors[2],
             hatch='xx')



plt.title('Results', fontsize=15)
plt.ylabel('Accuracy(%)', fontsize=10)

plt.yticks(range(0,101,20), fontsize=10)
plt.xticks(index+ [bar_width]*2, label, fontsize=10)

plt.legend((p1[0], p2[0], p3[0]), ('Source', 'LN-TENT', 'TeTra'), fontsize=10)
plt.savefig('figures/bar_plot.pdf')  
plt.show()