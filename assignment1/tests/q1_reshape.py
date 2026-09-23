from otter.test_files import test_case

OK_FORMAT = False

name = "q1_reshape"
points = 1

@test_case(points=None, hidden=False)
def test_shape(X_reshaped):
    assert X_reshaped.shape == (100, 240, 240, 3), f'X_reshaped has shape {X_reshaped.shape}, expected (100, 240, 240, 3)'

@test_case(points=None, hidden=False)
def test_keeps_order(np, X, X_reshaped):
    assert np.array_equal(np.ravel(X_reshaped)[:1000], X[:1000]), 'X_reshaped should contain the values of X, in the same order'

