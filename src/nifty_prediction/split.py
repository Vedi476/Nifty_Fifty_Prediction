
def train_test_split_by_time(X, Y, train_fraction=0.8):
    count = X.shape[0]
    index = count * train_fraction
    X_train = X[:int(index)]
    X_test = X[int(index):]
    Y_train = Y[:int(index)]
    Y_test = Y[int(index):]
    return X_train, X_test, Y_train, Y_test
