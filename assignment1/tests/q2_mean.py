from otter.test_files import test_case

OK_FORMAT = False

name = "q2_mean"
points = 1

@test_case(points=None, hidden=False)
def test_values(np, Z, Z_mean, Z_mean_shape):
    assert np.shape(Z_mean) == (10, 15) and np.allclose(Z_mean, Z.mean(axis=1)), 'Z_mean is not the mean of Z over the second dimension (numpy numbers dimensions from 0)'
    assert tuple(Z_mean_shape) == np.shape(Z_mean), 'Z_mean_shape should be the dimensions of Z_mean'

