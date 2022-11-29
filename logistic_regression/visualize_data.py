# In the name of Allah

import matplotlib.pyplot as plt
from LogRegModule import LogReg, read_data

# This code visualize admission.csv data in a 2D feature space

# Read admission.csv data then normalize it to [0, 20] scale
x, y = read_data()

# Instantiate a LogReg Model
my_model = LogReg()
my_model.x = x
my_model.y = y

fig, ax = my_model.visualize_data()
plt.show(block=True)
