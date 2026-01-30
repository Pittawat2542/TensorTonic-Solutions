import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    weights, biases = np.zeros(X.shape[1]), 0.0

    for i in range(steps):
        z = X @ weights + biases
        y_pred = _sigmoid(z)
        loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        gradient_weight = X.T @ (y_pred - y) / y.shape[0]
        gradient_biases = np.mean(y_pred - y)
        weights, biases = weights - lr * gradient_weight, biases - lr * gradient_biases
    
    return weights, biases