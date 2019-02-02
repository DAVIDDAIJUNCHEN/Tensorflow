import matplotlib.pyplot as plt
from train import *
from data import *

# Show the original data #
plt.plot(Train_X, Train_Y, 'ro', label='Original data')
plt.legend()
plt.show()


# Show the fitting curve and training curve #

