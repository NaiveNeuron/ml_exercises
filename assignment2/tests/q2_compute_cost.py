from otter.test_files import test_case

OK_FORMAT = False

name = "q2_compute_cost"
points = 1

@test_case(points=None, hidden=False)
def test_example(np, compute_cost):
    X = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    y = np.array([1.0, 2.0, 3.0])
    assert np.isclose(compute_cost(X, y, np.array([0.0, 0.0])), 14 / 6), 'wrong cost for theta = [0, 0]'
    assert np.isclose(compute_cost(X, y, np.array([0.0, 1.0])), 0), 'wrong cost for theta = [0, 1]'

