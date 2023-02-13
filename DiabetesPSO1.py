import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyswarms as ps
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import metrics
import matplotlib.pyplot as plt
from pyswarms.utils.plotters import plot_contour, plot_surface
from pyswarms.utils.plotters.formatters import Designer
from pyswarms.utils.plotters.formatters import Mesher
from pyswarms.utils.functions import single_obj as fx
#from sklearn.datasets import load_iris

#data_ = genfromtxt('pima-indians-diabetes.csv', delimiter=',', skip_header = 1)
# data = load_iris()
# X = data.data
# y = data.target

data = pd.read_csv("pima-indians-diabetes.csv")
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

d = preprocessing.normalize(X)
X = pd.DataFrame(d)
# X.head()
# y.head()

# data = []
# target = []
# for row in data_:
#     data.append(list(row[:-1]))
#     target.append(int(row[-1]))

X = X.to_numpy()
y = y.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=20, random_state=4)

n_inputs = 8
n_hidden = 20
n_classes = 2

num_samples = 767

def logits_function(p):
    """ Calculate roll-back the weights and biases

    Inputs
    ------
    p: np.ndarray
        The dimensions should include an unrolled version of the
        weights and biases.

    Returns
    -------
    numpy.ndarray of logits for layer 2

    """
    # Roll-back the weights and biases
    W1 = p[0:160].reshape((n_inputs,n_hidden))
    b1 = p[160:180].reshape((n_hidden,))
    W2 = p[180:220].reshape((n_hidden,n_classes))
    b2 = p[220:222].reshape((n_classes,))

    # Perform forward propagation
    z1 = X.dot(W1) + b1  # Pre-activation in Layer 1
    a1 = np.tanh(z1)     # Activation in Layer 1
    logits = a1.dot(W2) + b2 # Pre-activation in Layer 2
    return logits          # Logits for Layer 2

# Forward propagation
def forward_prop(params):
    """Forward propagation as objective function

    This computes for the forward propagation of the neural network, as
    well as the loss.

    Inputs
    ------
    params: np.ndarray
        The dimensions should include an unrolled version of the
        weights and biases.

    Returns
    -------
    float
        The computed negative log-likelihood loss given the parameters
    """

    logits = logits_function(params)

    # Compute for the softmax of the logits
    exp_scores = np.exp(logits)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # Compute for the negative log likelihood

    corect_logprobs = -np.log(probs[range(num_samples), y])
    loss = np.sum(corect_logprobs) / num_samples

    return loss

def f(x):
    """Higher-level method to do forward_prop in the
    whole swarm.

    Inputs
    ------
    x: numpy.ndarray of shape (n_particles, dimensions)
        The swarm that will perform the search

    Returns
    -------
    numpy.ndarray of shape (n_particles, )
        The computed loss for each particle
    """
    n_particles = x.shape[0]
    j = [forward_prop(x[i]) for i in range(n_particles)]
    return np.array(j)

def predict(X, p):
    """
    Use the trained weights to perform class predictions.

    Inputs
    ------
    X: numpy.ndarray
        Input Iris dataset
    pos: numpy.ndarray
        Position matrix found by the swarm. Will be rolled
        into weights and biases.
    """
    # Neural network architecture
    n_inputs = 8
    n_hidden = 20
    n_classes = 2

    # Roll-back the weights and biases
    W1 = p[0:160].reshape((n_inputs,n_hidden))
    b1 = p[160:180].reshape((n_hidden,))
    W2 = p[180:220].reshape((n_hidden,n_classes))
    b2 = p[220:222].reshape((n_classes,))

    # Perform forward propagation
    z1 = X.dot(W1) + b1  # Pre-activation in Layer 1
    a1 = np.tanh(z1)     # Activation in Layer 1
    z2 = a1.dot(W2) + b2 # Pre-activation in Layer 2
    logits = z2          # Logits for Layer 2

    y_pred = np.argmax(logits, axis=1)
    return y_pred

# Initialize swarm
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}

# Call instance of PSO
dimensions = (n_inputs * n_hidden) + (n_hidden * n_classes) + n_hidden + n_classes
optimizer = ps.single.GlobalBestPSO(n_particles=150, dimensions=dimensions, options=options)

# Perform optimization
cost, pos = optimizer.optimize(f, iters=500)

predicted_values = predict(X_test, pos)
actual_values = y_test
confusion_matrix = metrics.confusion_matrix(actual_values, predicted_values)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.show()
print("Accuracy score",accuracy_score(actual_values, predicted_values))
print("Precision score",precision_score(actual_values, predicted_values, pos_label=1))
print("Recall score",recall_score(actual_values, predicted_values))
print("F1 score",f1_score(actual_values, predicted_values))

m = Mesher(func=f)
# Make animation
animation = plot_contour(pos_history=optimizer.pos_history,
                         mesher=m,
                         mark=(0,0)) # Mark minima
animation.save('mymovie.mp4')
