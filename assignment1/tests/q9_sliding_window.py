from otter.test_files import test_case

OK_FORMAT = False

name = "q9_sliding_window"
points = 2

@test_case(points=None, hidden=False)
def test_shape(cuts):
    assert cuts.shape == (108, 80, 80, 3), f'cuts has shape {cuts.shape}, expected (108, 80, 80, 3)'

