from otter.test_files import test_case

OK_FORMAT = False

name = "q2_mean"
points = 1

@test_case(points=None, hidden=False)
def test_answers_present(np, Z_mean, Z_mean_shape):
    assert isinstance(Z_mean, np.ndarray), 'Z_mean should be a numpy array'
    assert isinstance(Z_mean_shape, tuple), 'Z_mean_shape should be a tuple (the dimensions of Z_mean)'

