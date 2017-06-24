# Support Vector Machine
# The code has been taken from https://github.com/llSourcell/Classifying_Data_Using_a_Support_Vector_Machine

# Math library
import numpy as np

# to plot our data and visualise it
from matplotlib import pyplot as plt


# Define our Data

# Input Data - [X value, Y value, Bias Term]
X = np.array([
    [-2, 4, -1],
    [4, 1, -1],
    [1, 6, -1],
    [2, 4, -1],
    [6, 2, -1],
])

# Associated output labels - First 2 examples are label '-1' and last 3 are '1'
Y = np.array([-1, -1, 1, 1, 1])


# Plot the data in a 2D graph
def plotX(X):
    for d, sample in enumerate(X):
        # Plot the negative samples
        if d < 2:
            plt.scatter(sample[0], sample[1], s=120, marker='_', linewidths=2)
        # Plot the positive samples (last 3)
        else:
            plt.scatter(sample[0], sample[1], s=120, marker='+', linewidths=2)
            pass

    plt.plot([-2, 6], [6, 0.5])
    plt.show()

plotX(X)

# stochastic gradient descent to learn the seperating hyperplane between both classes

def svm_sgd_plot(X, Y):
    # Initialize our SVM's weight vector with zeros
    w = np.zeros(len(X[0]))

    # The learning rate
    eta = 1

    # How many iterations to train for
    epochs = 100000

    # store misclassifications so we can plot how they change over time
    errors = []

    # training part, gradient descent part
    for epoch in range(1, epochs):
        error = 0

        for i, x in enumerate(X):
            # misapplication
            if (Y[i]*np.dot(X[i], w)) < 1:
                w = w + eta * ((X[i] * Y[i]) + (-2 * (1/epoch) * w))
                error = 1
            else:
                w = w + eta * (-2 * (1/epoch) * w)
            errors.append(error)

    # lets plot the rate of classification errors during the training
    plt.plot(errors, '|')
    plt.ylim(0.5, 1.5)
    plt.axes().set_yticklabels([])
    plt.xlabel('Epoch')
    plt.xlabel('Misclassified')
    plt.show()

    return w



w = svm_sgd_plot(X, Y)

for d, sample in enumerate(X):
    # Plot the negative samples
    if d < 2:
        plt.scatter(sample[0], sample[1], s=120, marker='_', linewidths=2)
    # Plot the positive samples
    else:
        plt.scatter(sample[0], sample[1], s=120, marker='+', linewidths=2)

# Add our test samples
plt.scatter(2,2, s=120, marker='_', linewidths=2, color='yellow')
plt.scatter(4,3, s=120, marker='+', linewidths=2, color='blue')

# Print the hyperplane calculated by svm_sgd()
x2=[w[0],w[1],-w[1],w[0]]
x3=[w[0],w[1],w[1],-w[0]]

x2x3 =np.array([x2,x3])
X,Y,U,V = zip(*x2x3)
ax = plt.gca()
ax.quiver(X,Y,U,V,scale=1, color='blue')
plt.show()










