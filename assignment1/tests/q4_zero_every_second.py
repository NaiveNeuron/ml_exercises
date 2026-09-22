from otter.test_files import test_case

OK_FORMAT = False

name = "q4_zero_every_second"
points = 1

@test_case(points=None, hidden=False)
def test_shape(R):
    assert R.shape == (10, 10, 3), f'R has shape {R.shape}, expected (10, 10, 3)'

