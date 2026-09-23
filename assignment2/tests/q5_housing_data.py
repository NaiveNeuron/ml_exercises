from otter.test_files import test_case

OK_FORMAT = False

name = "q5_housing_data"
points = 0.5

@test_case(points=None, hidden=False)
def test_shapes(X, y):
    assert X.shape == (47, 2), f'X has shape {X.shape}, expected (47, 2)'
    assert y.shape == (47,), f'y has shape {y.shape}, expected (47,)'

@test_case(points=None, hidden=False)
def test_same_examples(np, X, y, data):
    rows = lambda A: A[np.lexsort(A.T[::-1])]
    assert np.allclose(rows(np.c_[X, y]), rows(data)), 'X and y should contain exactly the examples from data (possibly shuffled)'

