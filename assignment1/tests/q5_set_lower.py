from otter.test_files import test_case

OK_FORMAT = False

name = "q5_set_lower"
points = 1

@test_case(points=None, hidden=False)
def test_example(np, set_lower):
    A = np.array([[1, 3, 5], [3, 5, 1], [1, 2, 5]])
    set_lower(A, 2.5, 5000)
    assert np.array_equal(A, [[5000, 3, 5], [3, 5, 5000], [5000, 5000, 5]]), f'got\n{A}'

