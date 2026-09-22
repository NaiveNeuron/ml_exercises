from otter.test_files import test_case

OK_FORMAT = False

name = "q3_grad_descent"
points = 1

@test_case(points=None, hidden=False)
def test_cost_decreases(np, grad_descent, compute_cost):
    import contextlib, io
    x = np.linspace(-1, 1, 50)
    X, y = (np.c_[np.ones(50), x], 2 + 3 * x)
    with contextlib.redirect_stdout(io.StringIO()):
        theta, history = grad_descent(X, y, np.zeros(2), 0.1)
    assert history[-1] < compute_cost(X, y, np.zeros(2)), 'the cost should go down during gradient descent'
    assert np.all(np.diff(history) <= 1e-09), 'the cost should never increase'

