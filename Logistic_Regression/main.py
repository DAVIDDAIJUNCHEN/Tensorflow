import matplotlib.pyplot as plt
from train import *
from data import *

# Show the original data #
plt.figure(1)
plt.subplot(121)
plt.plot(Train_X, Train_Y, 'ro', label='Original data')
plt.plot(Train_X, w2*Train_X+b2, label='Fitted data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Original data vs. Fitted data')

# Show the fitting curve and training curve #
plt.subplot(122)
plt.plot(plotdata['batchsize'], plotdata['loss'], 'b--', label='loss tends')
plt.xlabel('Minibatch number')
plt.ylabel('Loss')
plt.legend()
plt.title('Minibatch run vs. Training loss')

plt.show()

