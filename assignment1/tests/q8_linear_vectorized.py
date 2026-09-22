from otter.test_files import test_case

OK_FORMAT = False

name = "q8_linear_vectorized"
points = 1

@test_case(points=None, hidden=False)
def test_example(np, operate_vectorized):
    X = np.arange(6.0).reshape(2, 3)
    W = np.arange(12.0).reshape(3, 4)
    b = np.array([1.0, -1.0, 2.0, 0.5])
    assert np.isclose(operate_vectorized(X, W, b), 399.0), 'operate_vectorized returns a wrong value for the example input'

