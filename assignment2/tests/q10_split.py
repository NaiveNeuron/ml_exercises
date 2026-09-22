from otter.test_files import test_case

OK_FORMAT = False

name = "q10_split"
points = 0.5

@test_case(points=None, hidden=False)
def test_shapes(X_train, y_train, X_val, y_val, X_test, y_test):
    assert X_train.shape == (618, 8) and y_train.shape == (618,), 'the training set should have 618 examples'
    assert X_val.shape == (206, 8) and y_val.shape == (206,), 'the validation set should have 206 examples'
    assert X_test.shape == (206, 8) and y_test.shape == (206,), 'the test set should have 206 examples'

