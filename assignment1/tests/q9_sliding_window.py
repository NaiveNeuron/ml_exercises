from otter.test_files import test_case

OK_FORMAT = False

name = "q9_sliding_window"
points = 2

@test_case(points=None, hidden=False)
def test_shape(cuts):
    assert cuts.shape == (108, 80, 80, 3), f'cuts has shape {cuts.shape}, expected (108, 80, 80, 3)'

@test_case(points=None, hidden=False)
def test_contains_windows(np, img, cuts):

    def has(window):
        return any((np.array_equal(w, window) for w in cuts))
    assert has(img[:80, :80]), 'cuts should contain the top-left 80x80 window of the image'
    assert has(img[80:160, 160:240]), 'cuts should contain the 80x80 window starting at row 80, column 160; plain reshape mixes pixels from different parts of the image'

