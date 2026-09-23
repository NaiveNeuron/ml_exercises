from otter.test_files import test_case

OK_FORMAT = False

name = "q6_normalization"
points = 1

@test_case(points=None, hidden=False)
def test_shapes(np, X, mean, std):
    assert X.shape == (47, 3), f'X has shape {X.shape}, expected (47, 3): two normalized features plus the intercept'
    assert np.shape(mean) == (2,) and np.shape(std) == (2,), 'mean and std should hold one value per feature'

@test_case(points=None, hidden=False)
def test_normalized(np, X, y, mean, std, data):
    rows = lambda A: A[np.lexsort(A.T[::-1])]
    assert np.allclose(X[:, 0], 1), 'the first column of X should be the intercept term (all ones)'
    assert np.allclose(X[:, 1:].mean(axis=0), 0, atol=1e-08), 'normalized features should have zero mean'
    assert np.allclose(X[:, 1:].std(axis=0), 1, atol=0.02), 'normalized features should have unit standard deviation'
    assert np.allclose(mean, data[:, :2].mean(axis=0)), 'mean should be the per-feature mean of the housing data'
    assert np.allclose(std, data[:, :2].std(axis=0), rtol=0.02), 'std should be the per-feature standard deviation of the housing data'
    assert np.allclose(rows(X[:, 1:] * std + mean), rows(data[:, :2])), 'X should contain the normalized housing examples'
    assert np.allclose(np.sort(y), np.sort(data[:, 2])), 'y (the prices) should not be normalized'

