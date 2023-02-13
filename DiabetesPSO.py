import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyswarms as ps
from numpy import genfromtxt
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

data = pd.read_csv("pima-indians-diabetes.csv")
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

d = preprocessing.normalize(X)
X = pd.DataFrame(d)

X = X.to_numpy()
y = y.to_numpy()

n_inputs = 8
n_hidden = 20
n_classes = 2

num_samples = 767

def logits_function(p):
    # Roll-back the weights and biases
    W1 = p[0:160].reshape((n_inputs,n_hidden))
    b1 = p[160:180].reshape((n_hidden,))
    W2 = p[180:580].reshape((n_hidden,n_hidden))
    b2 = p[580:600].reshape((n_hidden,))
    W3 = p[600:640].reshape((n_hidden,n_classes))
    b3 = p[640:642].reshape((n_classes,))

    # Perform forward propagation
    z1 = X.dot(W1) + b1  # Pre-activation in Layer 1
    a1 = np.tanh(z1)     # Activation in Layer 1
    logits = a1.dot(W2) + b2 # Pre-activation in Layer 2
    z2 = logits.dot(W2) + b2
    a2 = np.tanh(z2)
    logits = a2.dot(W3) + b3
    return logits          # Logits for Layer 2

# Forward propagation
def forward_prop(params):
    logits = logits_function(params)

    # Compute for the softmax of the logits
    exp_scores = np.exp(logits)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # Compute for the negative log likelihood

    corect_logprobs = -np.log(probs[range(num_samples), y])
    loss = np.sum(corect_logprobs) / num_samples

    return loss

def f(x):
    n_particles = x.shape[0]
    j = [forward_prop(x[i]) for i in range(n_particles)]
    return np.array(j)

def predict(pos):
    logits = logits_function(pos)
    y_pred = np.argmax(logits, axis=1)
    return y_pred

# Initialize swarm
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}

# Call instance of PSO
dimensions = (n_inputs * n_hidden) + (n_hidden * n_hidden) + (n_hidden * n_classes) + n_hidden + n_hidden + n_classes
optimizer = ps.single.GlobalBestPSO(n_particles=150, dimensions=dimensions, options=options)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=20, random_state=4)
# Perform optimization
cost, pos = optimizer.optimize(f, iters=1000)

predicted = predict(pos)
print((predicted == y).mean())

fp = 0
fn = 0

tp = 0
tn = 0

actual_values, predicted_values = y, predicted

for actual_value, predicted_value in zip(actual_values, predicted_values):
    # let's first see if it's a true (t) or false prediction (f)
    if predicted_value == actual_value: # t?
        if predicted_value == 1: # tp
            tp += 1
        else: # tn
            tn += 1
    else: # f?
        if predicted_value == 1: # fp
            fp += 1
        else: # fn
            fn += 1
print(fp,fn,tp,tn)
confusion_matrix = metrics.confusion_matrix(actual_values, predicted_values)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
plt.show()

print("Accuracy score",accuracy_score(actual_values, predicted_values))
print("Precision score",precision_score(actual_values, predicted_values, pos_label=1))
print("Precision score",recall_score(actual_values, predicted_values))
print("F1 score",f1_score(actual_values, predicted_values))
