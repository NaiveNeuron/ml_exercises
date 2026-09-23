from otter.test_files import test_case

OK_FORMAT = False

name = "q7_house_price_gd"
points = 0.5

@test_case(points=None, hidden=False)
def test_numbers(np, house, price):
    assert np.shape(house) == (3,), 'house should be the normalized example, including the intercept term'
    assert np.ndim(price) == 0, 'price should be a single number'

@test_case(points=None, hidden=False)
def test_price_roughly(np, price):
    assert np.ndim(price) == 0 and abs(float(price) - 290000) / 290000 < 0.03, f'the price should be around 290000, got {price}; did you normalize the new house with the stored mean and std?'

