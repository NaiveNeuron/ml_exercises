from otter.test_files import test_case

OK_FORMAT = False

name = "q4_zero_every_second"
points = 1

@test_case(points=None, hidden=False)
def test_shape(R):
    assert R.shape == (10, 10, 3), f'R has shape {R.shape}, expected (10, 10, 3)'

@test_case(points=None, hidden=False)
def test_zero_pattern(np, R):
    assert np.all(R[:, 1, :] == 0) and np.all(R[:, 3, :] == 0), 'elements with index 1, 3, 5, ... in the second dimension should be zero'
    assert np.any(R[:, 0, :] != 0) and np.any(R[:, 2, :] != 0), 'elements with index 0, 2, 4, ... in the second dimension should stay random'

