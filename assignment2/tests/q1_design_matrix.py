from otter.test_files import test_case

OK_FORMAT = False

name = "q1_design_matrix"
points = 0.5

@test_case(points=None, hidden=False)
def test_shape(X):
    assert X.shape == (97, 2), f'X has shape {X.shape}, expected (97, 2)'

@test_case(points=None, hidden=False)
def test_columns(np, X, data):
    assert np.allclose(X[:, 0], 1), 'the first column of X should be the intercept term (all ones)'
    assert np.allclose(X[:, 1], data[:, 0]), 'the second column of X should be the population'

