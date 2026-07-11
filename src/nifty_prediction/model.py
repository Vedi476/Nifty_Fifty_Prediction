import numpy as np


class LinearWindowModel:
    def __init__(self, n_features, n_outputs):
        self.W = np.zeros((n_features, n_outputs))
        self.b = np.zeros(n_outputs)

    def forward(self, X):
        return X @ self.W + self.b

    def train(self, X, Y, epochs=100, lr=1e-5, verbose=True):
        loss_history = []
        for i in range(epochs):
            preds = self.forward(X)
            error = preds - Y
            loss = np.mean(error ** 2)
            if verbose:
                print(f"Epoch {i + 1}, loss={loss}")

            self.W = self.W - lr * ((X.T @ error) / X.shape[0])
            self.b = self.b - lr * np.mean(error, axis=0)
            loss_history.append(loss)
        return loss_history
