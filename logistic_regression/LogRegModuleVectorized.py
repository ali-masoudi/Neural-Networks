# In the name of Allah


import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

# Partially extracted from this topic:
# https://stackoverflow.com/questions/52379210/python-logistic-regression-produces-wrong-coefficients


class LogReg:
    # Logistic Regression for binary classification
    # x and y fields contains training data, input and class labels
    # z = np.inner(x[:, i], coeff) + intercept
    # y_hat = sigmoid(z)
    # y_hat > 0 class are ONE classes
    # y_hat < 0 are ZERO classes
    def __init__(self):
        self.intercept = 0
        self.coeff = np.ones([2, 1])*0.1
        self.x = None
        self.y = None

    def __str__(self):
        return f"{self.intercept}({self.coeff})"

    def calc_cost(self):
        # calc total cost for current model parameters (two coeffs and one intercept) over training samples
        # non-vectorized version
        # cost = 0
        epsi = 1e-30
        # m = self.x.shape[1]
        # for i in range(m):
        #     z = np.inner(self.x[:, i], self.coeff.squeeze()) + self.intercept
        #     y_hat = self.sigmoid(z)
        #     l = -(self.y[i]*math.log2(y_hat + epsi) + (1-self.y[i])*math.log2(1-y_hat + epsi))
        #     # l = 0.5*(y_hat-y[i])**2
        #     cost += l
        z = np.inner(self.x.T, self.coeff.squeeze()) + self.intercept
        y_hat = self.sigmoid(z)
        cost = -(self.y*np.log2(y_hat + epsi) + (1-self.y)*np.log2(1-y_hat + epsi))
        cost=cost.mean()
        return cost

    def sigmoid(self, x):
        # return 1 / (1 + math.exp(-x))
        return 1 / (1 + np.exp(-x))

    def visualize_data(self):
        # visualize x data in 2D feature space with their corresponding labels (y)
        fig = plt.figure()
        ax = plt.gca()

        x1_min = np.min(self.x[0, :])
        x2_min = np.min(self.x[1, :])
        x1_max = np.max(self.x[0, :])
        x2_max = np.max(self.x[1, :])

        for outcome in [0, 1]:
            xo = 'yo' if outcome == 0 else 'k+'
            ind = self.y == outcome
            plt.plot(self.x.T[ind, 0], self.x.T[ind, 1], xo, mec='k')
        plt.xlim([x1_min, x1_max])
        plt.ylim([x2_min, x2_max])

        plt.xlabel('Neural Networks Score')
        plt.ylabel('Pattern Recognition Score')
        plt.title('Neural Networks & Pattern Recognition and admission outcome\n O samples not admitted, + are admitted')
        return fig, ax

    def get_output(self, x):
        # return y_hat = sigmoid(W'x + b)
        # z = np.inner(x, self.coeff.squeeze()) + self.intercept # x here is one (2,) sample so (2,)*(2,) works
        z = np.inner(x.T, self.coeff.squeeze()) + self.intercept # x here is all samples (2,100) with 2 features x.T is (100,2)
        y_hat = self.sigmoid(z)
        return y_hat

    def get_loss_derivative(self):
        # returns dJ/db = (y_hat - y), dJ/dw = (y_hat - y)*x
        x = self.x
        y_hat = self.get_output(x)
        dw=((y_hat - self.y)*x).mean(axis=1)
        db=(y_hat - self.y).mean()
        return db,dw

    def visualize_probs(self, fig, ax):
        # visualize y_hat as class ONE prob in the 2D feature space
        # extracted from:
        # https://stackoverflow.com/questions/52379210/python-logistic-regression-produces-wrong-coefficients

        x1_min = np.min(self.x[0, :])
        x2_min = np.min(self.x[1, :])
        x1_max = np.max(self.x[0, :])
        x2_max = np.max(self.x[1, :])

        x1 = np.arange(x1_min, x1_max+0.1, 1)
        x2 = np.arange(x2_min, x2_max+0.7, .8)

        xx1, xx2 = np.meshgrid(x1, x2)
        probs = np.zeros([x1.shape[0], x2.shape[0]])
        for i in range(x1.shape[0]):
            for j in range(x2.shape[0]):
                probs[i, j] = self.get_output(np.array([x1[i], x2[j]]))

        contour = ax.contourf(xx1, xx2, probs.T, 100, cmap="RdBu", vmin=0, vmax=1)
        ax_c = fig.colorbar(contour)
        ax_c.set_label("$P(y = 1)$")
        ax_c.set_ticks([0, .25, .5, .75, 1])

        # Decision Surface (Hyperplane)
        plt.contour(xx1, xx2, probs.T, [0.5], linewidths=1, colors='b', alpha=0.3)

        # plt.plot(xx1[probs > 0.5], xx2[probs > 0.5], '.b', alpha=0.3)
        return fig, ax, probs, x1, x2

    def calc_err_landscape(self):
        b = np.arange(-30, -15, 1)  # intercept
        a = np.arange(0.5, 2.5, .1)  # NN Coeff
        err = np.zeros([a.shape[0], b.shape[0]])
        self.coeff[1] = 1  # Pattern Coeff
        for i in range(a.shape[0]):
            for j in range(b.shape[0]):
                self.coeff[0] = a[i]
                self.intercept = b[j]
                err[i, j] = self.calc_cost()
        return a, b, err


def read_data():
    # A helper function to read admission.csv data
    # admission.csv is extracted from Deep Learning Specialization Course Assignment #2
    # The function returns normalized scores in [0, 20]
    ex2_folder = ''
    input_1 = pd.read_csv(ex2_folder + 'admission.csv', header=None)
    input_1 = input_1.to_numpy()
    input_1[:, 0:2] /= 5

    x = input_1[:, 0:2]
    y = input_1[:, 2]
    return x.transpose(), y

#
# # Visualize sklearn probs
# def visualize(reg_model, fig, axis):
#     xx1, xx2 = np.mgrid[6:21:2, 6:21:2]
#     grid = np.c_[xx1.ravel(), xx2.ravel()]
#     probs = reg_model.predict_proba(grid)[:, 1]
#     probs = probs.reshape(xx1.shape)
#
#     contour = axis.contourf(xx1, xx2, probs, 100, cmap="RdBu", vmin=0, vmax=1)
#     ax_c = fig.colorbar(contour)
#     ax_c.set_label("$P(y = 1)$")
#     ax_c.set_ticks([0, .25, .5, .75, 1])
#
#     # Decision Surface (Hyperplane)
#     plt.contour(xx1, xx2, probs, [0.5], linewidths=1, colors='b', alpha=0.3)
#
#     # plt.plot(xx1[probs > 0.5], xx2[probs > 0.5], '.b', alpha=0.3)


