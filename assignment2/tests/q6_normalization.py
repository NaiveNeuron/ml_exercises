from otter.test_files import test_case

OK_FORMAT = False

name = "q6_normalization"
points = 1

@test_case(points=None, hidden=False)
def test_shapes(np, X, mean, std):
    assert X.shape == (47, 3), f'X has shape {X.shape}, expected (47, 3): two normalized features plus the intercept'
    assert np.shape(mean) == (2,) and np.shape(std) == (2,), 'mean and std should hold one value per feature'

