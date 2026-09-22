from otter.test_files import test_case

OK_FORMAT = False

name = "q7_linear_loops"
points = 1

@test_case(points=None, hidden=False)
def test_example(np, operate_loops):
    X = np.arange(6.0).reshape(2, 3)
    W = np.arange(12.0).reshape(3, 4)
    b = np.array([1.0, -1.0, 2.0, 0.5])
    assert np.isclose(operate_loops(X, W, b), 399.0), 'operate_loops returns a wrong value for the example input'

