from otter.test_files import test_case

OK_FORMAT = False

name = "q4_food_truck_predictions"
points = 0.5

@test_case(points=None, hidden=False)
def test_numbers(np, profit_35k, profit_70k):
    assert np.ndim(profit_35k) == 0 and np.ndim(profit_70k) == 0, 'profit_35k and profit_70k should be single numbers'
    (float(profit_35k), float(profit_70k))

@test_case(points=None, hidden=False)
def test_units(profit_35k, profit_70k):
    assert abs(float(profit_35k)) < 20 and abs(float(profit_70k)) < 20, 'predictions should be in $10 000s (like the data) and the population should be in 10 000s too (35 000 people is 3.5)'
    assert float(profit_70k) > float(profit_35k), 'a bigger city should have a bigger predicted profit'

