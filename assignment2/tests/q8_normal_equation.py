from otter.test_files import test_case

OK_FORMAT = False

name = "q8_normal_equation"
points = 1

@test_case(points=None, hidden=False)
def test_example(np, compute_theta_norm_eq):
    X = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    y = np.array([1.0, 3.0, 5.0])
    assert np.allclose(compute_theta_norm_eq(X, y), [1, 2]), 'wrong theta for a simple line y = 1 + 2x'

@test_case(points=None, hidden=False)
def test_price_roughly(np, price):
    assert np.ndim(price) == 0 and abs(float(price) - 290000) / 290000 < 0.03, f'the price should be around 290000, got {price}; did you normalize the new house with the stored mean and std?'

