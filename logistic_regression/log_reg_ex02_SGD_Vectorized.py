# In the name of Allah
import time

import matplotlib.pyplot as plt
import numpy as np
from LogRegModuleVectorized import LogReg, read_data


x, y = read_data()

my_model = LogReg()
my_model.coeff[0] = 1.4
my_model.coeff[1] = 0.1
my_model.intercept = -17
my_model.x = x
my_model.y = y


eta = 0.1
m = my_model.x.shape[1]
number_of_epochs = 100
#costs = np.zeros(number_of_epochs*m)
costs = np.zeros(number_of_epochs)
randomize = np.arange(m)
# cntr = 0
tic=time.time()
for epoch in range(number_of_epochs):
    eta = eta*0.9
    # for i in np.random.permutation(m):
    #     costs[cntr] = my_model.calc_cost()
    #     cntr += 1
    #     db, dw = my_model.get_loss_derivative(i)
    #     my_model.intercept = my_model.intercept - eta*db
    #     my_model.coeff = my_model.coeff - eta*dw[np.newaxis].T

    costs[epoch] = my_model.calc_cost()
    db, dw = my_model.get_loss_derivative()
    my_model.intercept = my_model.intercept - eta*db
    my_model.coeff = my_model.coeff - eta*dw[np.newaxis].T
    #randomize samples order
    np.random.shuffle(randomize)
    my_model.x=my_model.x[:,randomize]
    my_model.y=my_model.y[randomize]


# plt.plot(costs)

# my_model.coeff = np.array([0.94157241, 0.91666647]).T
# my_model.intercept = -22.93138926
toc=time.time()
print('Intercept (Theta 0: {}). Coefficients: {}'.format(my_model.intercept, my_model.coeff))
print('Elapsed Time:{}'.format((toc-tic)))

plt.show()
fig, ax = my_model.visualize_data()
fig, ax, probs, x1, x2 = my_model.visualize_probs(fig, ax)
plt.title('Current Cost: %3.2f' % my_model.calc_cost())
plt.show(block=True)
