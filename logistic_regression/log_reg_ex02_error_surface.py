# In the name of Allah

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from LogRegModule import LogReg, read_data


x, y = read_data()

my_model = LogReg()
my_model.x = x
my_model.y = y
a, b, err_l = my_model.calc_err_landscape()


# PLOTTING

f = plt.figure()
ax = plt.gca()

xx1, xx2 = np.meshgrid(a, b)

contour = ax.contour(xx1, xx2, err_l.T, 100, cmap="RdBu")
ax_c = f.colorbar(contour)
plt.xlabel('Neural Networks Score Coeff')
plt.ylabel('Intercept')
plt.title('Cost in Param Space (Ext. Error Space) - (PR Coeff = 1)')


plt.show(block=False)

fig, ax = my_model.visualize_data()
fig, ax, probs, x1, x2 = my_model.visualize_probs(fig, ax)
plt.title('Current Cost: %3.2f' % my_model.calc_cost())
plt.show(block=True)

# value = input("Please enter a key\n")