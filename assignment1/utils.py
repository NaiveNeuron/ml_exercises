import time
import functools
from matplotlib import pyplot as plt
import numpy as np


def timeit(func):
    """
    Profiling function to measure time it takes to finish function.

    Args:
        func(*function): Function to meassure

    Returns:
        (*function) New wrapped function with meassurment
    """
    @functools.wraps(func)
    def newfunc(*args, **kwargs):
        start_time = time.time()
        out = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        ftime = elapsed_time
        msg = "function [{}] finished in {} ms"
        print(msg.format(func.__name__, ftime))
        return out
    return newfunc


def show_cut_image(x):
    f, axarr = plt.subplots(4, 4)
    rnd = x[np.random.randint(x.shape[0], size=16)]
    for i in range(4):
        for j in range(4):
            axarr[i, j].imshow(rnd[i + j * 4])
            axarr[i, j].axis('off')
    plt.show()
