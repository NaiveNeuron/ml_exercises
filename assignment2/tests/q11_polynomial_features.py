from otter.test_files import test_case

OK_FORMAT = False

name = "q11_polynomial_features"
points = 1

@test_case(points=None, hidden=False)
def test_consistent_features(np, X_train, X_val, X_test, theta):
    assert X_train.shape[1] > 9, 'add new features to X_train (e.g. polynomial ones)'
    assert X_val.shape[1] == X_test.shape[1] == X_train.shape[1] == np.shape(theta)[0], 'X_val and X_test need the same features as X_train'

@test_case(points=None, hidden=False)
def test_errors(np, X_train, y_train, X_val, y_val, theta):
    mse = lambda X, y: np.sum((X @ theta - y) ** 2) / (2 * len(y))
    assert mse(X_train, y_train) < 40, f'training error is too high: {mse(X_train, y_train)}'
    assert mse(X_val, y_val) < 60, f'validation error is too high: {mse(X_val, y_val)}'

