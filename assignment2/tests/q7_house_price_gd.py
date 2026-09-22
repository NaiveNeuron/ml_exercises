from otter.test_files import test_case

OK_FORMAT = False

name = "q7_house_price_gd"
points = 0.5

@test_case(points=None, hidden=False)
def test_numbers(np, house, price):
    assert np.shape(house) == (3,), 'house should be the normalized example, including the intercept term'
    assert np.ndim(price) == 0, 'price should be a single number'

