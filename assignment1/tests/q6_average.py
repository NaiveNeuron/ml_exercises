from otter.test_files import test_case

OK_FORMAT = False

name = "q6_average"
points = 1

@test_case(points=None, hidden=False)
def test_example(np, operate_loops_avg, operate_vectorized_avg):
    assert np.isclose(operate_loops_avg(np.arange(50)), 24.5), 'operate_loops_avg(np.arange(50)) should be 24.5'
    assert np.isclose(operate_vectorized_avg(np.arange(50)), 24.5), 'operate_vectorized_avg(np.arange(50)) should be 24.5'

