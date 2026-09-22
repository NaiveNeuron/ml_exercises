from otter.test_files import test_case

OK_FORMAT = False

name = "q2_mean"
points = 1

@test_case(points=None, hidden=False)
def test_shape(Z_mean):
    assert Z_mean.shape == (10, 15), f'Z_mean has shape {Z_mean.shape}, expected (10, 15)'

